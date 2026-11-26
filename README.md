# RyGround Flash

**Resolution-preserving GUI grounding with FastVLM-0.5B.**

RyGround Flash adapts FastVLM to direct coordinate prediction through parameter-efficient supervised fine-tuning. It combines high-resolution, aspect-ratio-preserving visual inputs with a simple point-prediction objective—no region heads, reward models, or multi-stage decoding.

## Highlights

- **Native aspect ratios.** Target a 2048-pixel long side with 64-pixel alignment. No center cropping, stretching, or letterboxing.
- **High-resolution GUI inputs.** A minimum 1024-pixel short side preserves detail while supporting 16:9 displays, 32:9 ultrawide screens, and portrait mobile UIs.
- **Direct grounding.** Predict `(x, y)` from the GT box center using answer-only cross-entropy with explicit end-of-turn supervision.
- **Parameter-efficient training.** Freeze FastViTHD and the multimodal projector; train only language-model LoRA adapters.
- **Consistent coordinates.** Recompute targets from original bounding boxes under the exact same resize protocol used at inference.

## Training

Configure the model, dataset, and image-root paths in `configs/sft.yaml`. Each JSONL record must provide `image`, `instruction`, and an absolute-pixel `bbox`.

```bash
python train.py --config configs/sft.yaml --dry-run
python train.py --config configs/sft.yaml
```

The default recipe uses 8K samples, one epoch, batch size 1, gradient accumulation 48, and LoRA rank 8 with alpha 16.

## Inference

```bash
python infer.py \
  --model /path/to/FastVLM-0.5B \
  --adapter /path/to/direct_lora \
  --image /path/to/screenshot.png \
  --instruction "click the save button"
```

The output includes the raw response, visual-token count, point in the resized frame, and point mapped back to the original screenshot.

## Validation

Verified with real FastVLM weights: an 8-sample two-update dry-run, 16:9 and 32:9 forward/backward passes, coordinate round-trip checks, and adapter save/reload inference.

Full 8K training and ScreenSpot benchmark results are **TBD**.

## TODO

- [ ] Add Zoom-In (under consideration).
