import argparse
import json
import re

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from protocol import build_prompt_ids, point_to_original, resized_dims, visual_token_count


def parse_point(text):
    match = re.search(
        r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]", text)
    return [float(match.group(1)), float(match.group(2))] if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--min-side", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.mm_vision_tower = f"mobileclip_l_{args.max_side}"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.generation_config.do_sample = False

    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    resized_size = resized_dims(*original_size, args.max_side, args.min_side)
    image = image.resize(resized_size)
    pixel_values = model.get_vision_tower().image_processor(
        images=image,
        do_resize=False,
        do_center_crop=False,
        return_tensors="pt",
    )["pixel_values"].to(model.device, dtype=model.dtype)
    input_ids, attention_mask = build_prompt_ids(tokenizer, args.instruction)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = attention_mask.unsqueeze(0).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    raw = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    point = parse_point(raw)
    result = {
        "raw": raw,
        "original_size": list(original_size),
        "resized_size": list(resized_size),
        "visual_tokens": visual_token_count(*resized_size),
        "point_in_resized_image": point,
        "point_in_original_image": point_to_original(point, original_size, resized_size)
        if point is not None else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
