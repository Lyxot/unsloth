"""Reproduce unslothai/unsloth#15473: Gemma 4 on MLX answering an image prompt with <unused6226>.

Loads the model the way Unsloth Studio does (FastMLXModel.from_pretrained), optionally LoRA-trains it
for a few steps, then runs a text prompt and an image prompt greedily and reports whether the first
token's logprobs are NaN.
"""

import argparse
import importlib.metadata as md
import json
import math
import os
import platform
import sys
import tempfile

import mlx.core as mx
from PIL import Image, ImageDraw


def versions():
    out = {}
    for p in ("mlx", "mlx-metal", "mlx-vlm", "mlx-lm", "transformers", "unsloth_zoo"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def shapes_image(path):
    im = Image.new("RGB", (640, 640), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 120, 180, 440], fill = (228, 52, 36))
    d.ellipse([400, 120, 720, 440], fill = (0, 0, 255))
    im.save(path)
    return path


def run_prompt(model, processor, text, image):
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(processor, model.config, text, num_images = 1 if image else 0)
    pieces, first_lp, n = [], None, 0
    for r in stream_generate(
        model, processor, prompt, image = [image] if image else None, max_tokens = 40, temperature = 0.0
    ):
        if first_lp is None and getattr(r, "logprobs", None) is not None:
            lp = r.logprobs.astype(mx.float32)
            first_lp = {
                "nan": int(mx.isnan(lp).sum().item()),
                "inf": int(mx.isinf(lp).sum().item()),
                "size": int(lp.size),
            }
        pieces.append(r.text)
        n += 1
    out = "".join(pieces)
    bad = "<unused" in out or out.strip().startswith("<pad>") or (first_lp and first_lp["nan"])
    return {
        "prompt": text,
        "image": bool(image),
        "output": out,
        "first_token_logprobs": first_lp,
        "tokens": n,
        "broken": bool(bad),
    }


def lora_train(model, processor, steps, out_dir):
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model = FastMLXModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0)
    data = [
        {"text": f"### Question: what is {i} plus {i}?\n### Answer: {2 * i}."} for i in range(24)
    ]
    trainer = MLXTrainer(
        model = model,
        tokenizer = processor,
        train_dataset = data,
        args = MLXTrainingConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            max_steps = steps,
            warmup_steps = 1,
            learning_rate = 2e-4,
            logging_steps = 1,
            output_dir = out_dir,
            seed = 3407,
            report_to = "none",
        ),
    )
    trainer.train()
    losses = list(getattr(trainer, "_train_loss_history", []))
    model.eval()
    return model, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "mlx-community/gemma-4-e4b-it-4bit")
    ap.add_argument("--dtype", default = "float16", help = "float16, bfloat16, float32, or auto")
    ap.add_argument("--train-steps", type = int, default = 0)
    ap.add_argument("--report", default = "report.json")
    args = ap.parse_args()

    from unsloth_zoo.mlx.loader import FastMLXModel

    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device_info": {k: str(v) for k, v in mx.device_info().items()},
        "metal_available": mx.metal.is_available(),
        "versions": versions(),
        "args": vars(args),
    }
    print(json.dumps(info, indent = 2), flush = True)

    model, processor = FastMLXModel.from_pretrained(
        args.model, max_seq_length = 2048, dtype = None if args.dtype == "auto" else args.dtype
    )
    info["loaded_dtype"] = str(
        model.language_model.model.embed_tokens.scales.dtype
        if hasattr(model.language_model.model.embed_tokens, "scales")
        else model.language_model.model.embed_tokens.weight.dtype
    )
    info["pooler_fp32_patch"] = bool(getattr(model, "_unsloth_gemma4_vision_pooler_fp32", False))

    image = shapes_image(os.path.join(tempfile.mkdtemp(), "shapes.png"))
    results = []
    if args.train_steps:
        model, losses = lora_train(model, processor, args.train_steps, tempfile.mkdtemp())
        info["train_losses"] = losses
        info["train_losses_finite"] = all(math.isfinite(x) for x in losses)
    for text, img in (
        ("Write one sentence about the ocean.", None),
        ("Describe this image", image),
    ):
        res = run_prompt(model, processor, text, img)
        print(json.dumps(res, indent = 2), flush = True)
        results.append(res)
    info["results"] = results
    info["reproduced"] = any(r["broken"] for r in results)
    with open(args.report, "w") as f:
        json.dump(info, f, indent = 2)
    print(f"REPRODUCED={info['reproduced']}")
    # A red job marks the reproducing cell.
    sys.exit(3 if info["reproduced"] else 0)


if __name__ == "__main__":
    main()
