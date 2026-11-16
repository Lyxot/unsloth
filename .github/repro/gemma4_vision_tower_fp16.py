"""Gemma 4 vision tower alone in float16: are its soft tokens finite? (unslothai/unsloth#15473)

Only the shard holding vision_tower / embed_vision is downloaded, so the 26B tower runs where the
full checkpoint does not fit. Importing unsloth_zoo's MLX loader applies its Gemma 4 fixups the way a
model load does, so the same script tells a fixed unsloth-zoo from an unfixed one.
"""

import argparse
import json
import os
import sys
import tempfile

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

PREFIXES = ("vision_tower.", "embed_vision.")


def shapes_image():
    im = Image.new("RGB", (640, 640), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 120, 180, 440], fill = (228, 52, 36))
    d.ellipse([400, 120, 720, 440], fill = (0, 0, 255))
    return im


def vision_weights(repo):
    index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({f for k, f in index.items() if k.startswith(PREFIXES)})
    weights = {}
    for shard in shards:
        for k, v in mx.load(hf_hub_download(repo, shard)).items():
            if k.startswith(PREFIXES):
                weights[k] = v
    return weights, shards


def build(repo, weights):
    from mlx_vlm.models.gemma4.config import TextConfig, VisionConfig
    from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder
    from mlx_vlm.models.gemma4.vision import VisionModel

    raw = json.load(open(hf_hub_download(repo, "config.json")))
    vision_cfg = VisionConfig.from_dict(raw["vision_config"])
    text_cfg = TextConfig.from_dict(raw["text_config"])

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = VisionModel(vision_cfg)
            self.embed_vision = MultimodalEmbedder(
                embedding_dim = vision_cfg.hidden_size,
                text_hidden_size = text_cfg.hidden_size,
                eps = vision_cfg.rms_norm_eps,
            )

        def __call__(self, pixels):
            tower = self.vision_tower(pixels)
            return tower, self.embed_vision(tower)

    model = Vision()
    q = raw.get("quantization") or {}
    if q:
        nn.quantize(
            model,
            group_size = q["group_size"],
            bits = q["bits"],
            class_predicate = lambda p, m: f"{p}.scales" in weights,
        )
    if not getattr(vision_cfg, "use_clipped_linears", False):
        weights = {
            k: v
            for k, v in weights.items()
            if not any(s in k for s in ("input_max", "input_min", "output_max", "output_min"))
        }
    model.load_weights(list(weights.items()), strict = False)
    return model, vision_cfg


def pixels(repo):
    from mlx_vlm.utils import load_processor

    path = os.path.dirname(hf_hub_download(repo, "config.json"))
    for f in (
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ):
        hf_hub_download(repo, f)
    processor = load_processor(path, add_detokenizer = False)
    out = processor.image_processor([shapes_image()])
    return mx.array((out[0] if isinstance(out, tuple) else out)["pixel_values"])


def stats(x):
    x32 = x.astype(mx.float32)
    finite = mx.isfinite(x32)
    return {
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "nonfinite": int((~finite).sum().item()),
        "absmax_finite": float(mx.max(mx.where(finite, mx.abs(x32), 0)).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "mlx-community/gemma-4-26b-a4b-it-4bit")
    ap.add_argument("--report", default = "tower_report.json")
    ap.add_argument("--expect", choices = ("reproduced", "clean"), default = None)
    args = ap.parse_args()

    import unsloth_zoo
    import unsloth_zoo.mlx.loader as loader

    fix = getattr(loader, "_fix_gemma4_vision_pooler_fp32", None)
    applied = bool(fix and fix())
    weights, shards = vision_weights(args.model)
    model, _ = build(args.model, weights)
    px = pixels(args.model)

    info = {
        "model": args.model,
        "shards": shards,
        "unsloth_zoo": unsloth_zoo.__version__,
        "pooler_fix_present": fix is not None,
        "pooler_fix_applied": applied,
        "device": mx.device_info().get("device_name"),
    }
    ref = None
    for dtype in (mx.float32, mx.float16):
        model.set_dtype(dtype)
        tower, soft = model(px.astype(dtype))
        mx.eval(tower, soft)
        info[str(dtype)] = {"tower": stats(tower), "soft_tokens": stats(soft)}
        if dtype == mx.float32:
            ref = soft.astype(mx.float32)
        else:
            s = soft.astype(mx.float32)
            ok = mx.all(mx.isfinite(s)).item()
            info[str(dtype)]["cosine_vs_fp32"] = (
                float((mx.sum(s * ref) / (mx.linalg.norm(s) * mx.linalg.norm(ref))).item())
                if ok
                else None
            )
    info["reproduced"] = info[str(mx.float16)]["soft_tokens"]["nonfinite"] > 0
    print(json.dumps(info, indent = 2))
    json.dump(info, open(args.report, "w"), indent = 2)
    print(f"REPRODUCED={info['reproduced']}")
    if args.expect:
        sys.exit(0 if info["reproduced"] == (args.expect == "reproduced") else 1)
    sys.exit(3 if info["reproduced"] else 0)


if __name__ == "__main__":
    main()
