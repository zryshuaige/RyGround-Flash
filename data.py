import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from protocol import build_prompt_ids, encode_target, resized_dims, target_point


class FastVLMDataset(Dataset):
    def __init__(self, jsonl_path, image_root, tokenizer, image_processor,
                 max_side=2048, min_side=1024):
        self.records = [
            json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_side = max_side
        self.min_side = min_side

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        width, height = original_size
        x1, y1, x2, y2 = (float(value) for value in record["bbox"])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"invalid bbox in {record.get('id', index)}")

        resized_size = resized_dims(width, height, self.max_side, self.min_side)
        image = image.resize(resized_size)
        pixel_values = self.image_processor(
            images=image,
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt",
        )["pixel_values"][0]

        point = target_point(record["bbox"], original_size, resized_size)
        prompt_ids, attention_mask = build_prompt_ids(
            self.tokenizer, record["instruction"])
        target_ids, target_sections = encode_target(self.tokenizer, point)
        return {
            "input_ids": torch.cat([prompt_ids, target_ids]),
            "labels": torch.cat([
                torch.full_like(prompt_ids, -100),
                target_ids,
            ]),
            "loss_sections": torch.cat([
                torch.zeros_like(prompt_ids),
                target_sections,
            ]),
            "attention_mask": torch.cat([
                attention_mask,
                torch.ones_like(target_ids),
            ]),
            "images": pixel_values,
        }


def collate(features, pad_id):
    if len(features) != 1:
        raise ValueError("variable-resolution training requires batch_size=1")
    feature = features[0]
    return {
        "input_ids": feature["input_ids"].unsqueeze(0),
        "labels": feature["labels"].unsqueeze(0),
        "loss_sections": feature["loss_sections"].unsqueeze(0),
        "attention_mask": feature["attention_mask"].unsqueeze(0),
        "images": feature["images"].unsqueeze(0),
    }
