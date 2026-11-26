import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data import FastVLMDataset, collate
from protocol import IMAGE_TOKEN_INDEX

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def expand_image_placeholder(tensor, input_ids, visual_tokens, fill_value):
    expanded = []
    for row, ids in zip(tensor, input_ids):
        positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
        if len(positions) != 1:
            raise ValueError("each sample must contain exactly one image placeholder")
        position = int(positions[0])
        image_values = torch.full(
            (visual_tokens,), fill_value, dtype=row.dtype, device=row.device)
        expanded.append(torch.cat([row[:position], image_values, row[position + 1:]]))
    return torch.stack(expanded)


class SectionalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sections = inputs.pop("loss_sections")
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]
        images = inputs["images"]
        visual_tokens = (images.shape[-2] // 64) * (images.shape[-1] // 64)
        labels = expand_image_placeholder(labels, input_ids, visual_tokens, -100)
        sections = expand_image_placeholder(sections, input_ids, visual_tokens, 0)

        logits = model(**inputs).logits[:, :-1].float()
        labs = labels[:, 1:]
        secs = sections[:, 1:]
        if logits.shape[:2] != labs.shape:
            raise RuntimeError(f"logits/labels misaligned: {logits.shape} vs {labs.shape}")
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labs.clamp_min(0).reshape(-1),
            reduction="none",
        ).view(labs.shape)
        ce = ce * (labs != -100)
        loss = ce.new_zeros(())
        for section_id, weight in ((3, 1.0), (4, 1.0)):
            mask = secs == section_id
            if mask.any():
                loss = loss + weight * ce[mask].sum() / mask.sum()
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--image-root")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in ("model", "data", "output"):
        if getattr(args, key):
            cfg[key] = getattr(args, key)
    if args.image_root:
        cfg["image_root"] = args.image_root
    if cfg["batch_size"] != 1:
        raise ValueError("variable-resolution training currently requires batch_size=1")

    torch.manual_seed(cfg["seed"])
    dtype = torch.bfloat16 if cfg["dtype"] == "bf16" else torch.float16
    model_config = AutoConfig.from_pretrained(cfg["model"], trust_remote_code=True)
    model_config.mm_vision_tower = f"mobileclip_l_{cfg['max_side']}"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        config=model_config,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora)
    base_model = model.get_base_model()
    base_model.get_vision_tower().requires_grad_(False)
    base_model.get_model().mm_projector.requires_grad_(cfg["train_projector"])
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = FastVLMDataset(
        cfg["data"],
        cfg["image_root"],
        tokenizer,
        model.get_vision_tower().image_processor,
        cfg["max_side"],
        cfg["min_side"],
    )
    if args.dry_run:
        dataset.records = dataset.records[:8]
    if not dataset.records:
        raise ValueError("no training records found")
    print(f"training sequences: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=cfg["output"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if args.dry_run else cfg["grad_accum"],
        learning_rate=cfg["lr"],
        num_train_epochs=1 if args.dry_run else cfg["epochs"],
        lr_scheduler_type="cosine",
        bf16=cfg["dtype"] == "bf16",
        fp16=cfg["dtype"] == "fp16",
        logging_steps=1 if args.dry_run else cfg["logging_steps"],
        save_strategy="no" if args.dry_run else "steps",
        save_steps=cfg["save_steps"],
        report_to=[],
        seed=cfg["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0 if args.dry_run else cfg["num_workers"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=2 if args.dry_run else -1,
    )
    trainer = SectionalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda features: collate(features, tokenizer.pad_token_id),
    )
    trainer.train()

    if not args.dry_run:
        output = Path(cfg["output"])
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        run_config = dict(cfg)
        run_config["prompt"] = __import__("protocol").PROMPT_DIRECT
        (output / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"adapter saved to {output}")


if __name__ == "__main__":
    main()
