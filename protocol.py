import torch

IMAGE_TOKEN_INDEX = -200
PATCH_SIZE = 64
PROMPT_DIRECT = (
    "Based on the screenshot of the page, I give a text description and you give "
    "its corresponding location. The coordinate represents a clickable location "
    "[x, y] for an element."
)


def resized_dims(width, height, max_side=2048, min_side=1024):
    if max_side % PATCH_SIZE or min_side % PATCH_SIZE:
        raise ValueError("max_side and min_side must be multiples of 64")
    if min_side < 1024 or max_side < min_side:
        raise ValueError("require max_side >= min_side >= 1024 for FastViTHD")
    scale = max(max_side / max(width, height), min_side / min(width, height))
    resized = tuple(
        max(PATCH_SIZE, round(side * scale / PATCH_SIZE) * PATCH_SIZE)
        for side in (width, height)
    )
    return resized


def visual_token_count(width, height):
    if width % PATCH_SIZE or height % PATCH_SIZE:
        raise ValueError("image dimensions must be multiples of 64")
    return (width // PATCH_SIZE) * (height // PATCH_SIZE)


def target_point(bbox, original_size, resized_size):
    x1, y1, x2, y2 = bbox
    width, height = original_size
    resized_width, resized_height = resized_size
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (
        round(center_x * resized_width / width),
        round(center_y * resized_height / height),
    )


def point_to_original(point, original_size, resized_size):
    width, height = original_size
    resized_width, resized_height = resized_size
    return [
        point[0] * width / resized_width,
        point[1] * height / resized_height,
    ]


def build_prompt_ids(tokenizer, instruction):
    content = f"<image>\n{PROMPT_DIRECT}\n{instruction}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )
    pre, post = rendered.split("<image>", 1)
    pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids[0]
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids[0]
    image_id = torch.tensor([IMAGE_TOKEN_INDEX], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, image_id, post_ids])
    return input_ids, torch.ones_like(input_ids)


def encode_target(tokenizer, point):
    coordinate = f"({point[0]}, {point[1]})"
    coordinate_ids = tokenizer(
        coordinate, return_tensors="pt", add_special_tokens=False).input_ids[0]
    terminator_ids = tokenizer(
        "<|im_end|>\n", return_tensors="pt", add_special_tokens=False).input_ids[0]
    target_ids = torch.cat([coordinate_ids, terminator_ids])
    sections = torch.cat([
        torch.full_like(coordinate_ids, 3),
        torch.full_like(terminator_ids, 4),
    ])
    return target_ids, sections
