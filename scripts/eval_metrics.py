# scripts/eval_metrics.py
"""
Summarize Keras model architectures (LSTM/MLP/others) into paper-style tables.

Usage examples:
  python scripts/eval_metrics.py \
    --crop_model models/weights/wcp_LSTM_model_fs_model.h5 \
    --rc_model   models/weights/rc_LSTM_model_fs_model.h5 \
    --ghc_model  models/weights/ghc_mlp_model_fs_model.h5 \
    --out results/metrics/model_architectures.md

You can pass any subset; only provided models are summarized.
"""

import argparse
import os
from typing import Optional
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model  # optional; not required
from tensorflow.keras import Model

def _fmt_shape(shape) -> str:
    # Keras returns tuples; make them look like (None, 288, 16) etc.
    if shape is None:
        return "None"
    return "(" + ", ".join("None" if s is None else str(s) for s in shape) + ")"

def summarize_model(model: Model, title: str) -> str:
    """Return a markdown block with a table like those in the paper."""
    lines = []
    lines.append(f"{title}")
    lines.append(f"")
    lines.append(f"Layer (type)\tOutput Shape\tParam #")
    total_params = int(model.count_params())
    trainable_params = 0
    non_trainable_params = 0

    # Walk layers
    for layer in model.layers:
        name = layer.__class__.__name__
        # Output shape: many layers expose .output_shape; fallback to getattr
        out_shape = getattr(layer, "output_shape", None)
        # When model is not built, Keras may defer shapes; force build if needed
        try:
            if out_shape is None:
                # attempt to infer from _inbound_nodes; if not available, leave blank
                out_shape = getattr(layer, "output_shape", None)
        except Exception:
            out_shape = None

        out_str = _fmt_shape(out_shape if out_shape is not None else None)

        # Params for this layer
        params = 0
        try:
            params = int(layer.count_params())
        except Exception:
            params = 0

        lines.append(f"{name}\t{out_str}\t{params}")

        # Track trainable/non-trainable
        if layer.trainable:
            trainable_params += params
        else:
            non_trainable_params += params

    # Footer (totals)
    lines.append(f"Total params: {total_params}")
    lines.append(f"Trainable params: {trainable_params}")
    lines.append(f"Non-trainable params: {non_trainable_params}")
    lines.append("")  # blank line
    return "\n".join(lines)

def maybe_load(path: Optional[str]) -> Optional[Model]:
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"WARNING: model not found: {path}")
        return None
    try:
        return load_model(path)
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop_model", help="Path to crop-parameter estimator (.h5)", default=None)
    ap.add_argument("--rc_model",   help="Path to resource-consumption estimator (.h5)", default=None)
    ap.add_argument("--ghc_model",  help="Path to greenhouse-climate estimator (.h5)", default=None)
    ap.add_argument("--out",        help="Markdown output file", default="results/metrics/model_architectures.md")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    blocks = []

    crop = maybe_load(args.crop_model)
    if crop is not None:
        blocks.append(summarize_model(crop, "Table: Crop parameter (LSTM) model architecture."))

    rc = maybe_load(args.rc_model)
    if rc is not None:
        blocks.append(summarize_model(rc, "Table: Resource consumption (LSTM/MLP) model architecture."))

    ghc = maybe_load(args.ghc_model)
    if ghc is not None:
        blocks.append(summarize_model(ghc, "Table: Greenhouse climate (MLP) model architecture."))

    if not blocks:
        print("No models summarized (no valid paths provided).")
        return

    md = "\n".join(blocks)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n===== SUMMARY =====\n")
    print(md)
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
