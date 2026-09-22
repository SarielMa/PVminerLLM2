"""
Compatibility helpers shared by the SFT / preference / inference scripts.

Exists because adding Qwen3.5-9B and Qwen3.8-27B broke three assumptions that
were safe for the Llama-3.x / Qwen2.5 models:

1. They ship as `Qwen3_5ForConditionalGeneration` -- a vision-language wrapper
   around a text decoder -- so `AutoModelForCausalLM` cannot instantiate them.

2. Only 1 layer in 4 is `full_attention`; the other 3 are `linear_attention`
   and expose `in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj`
   instead of `q_proj/k_proj/v_proj/o_proj`. The old hardcoded LoRA target
   lists match 12.9% (SFT) and 51.6% (PO) of the language tower, so most of
   the network would silently never train.

3. They are hybrid reasoning models: the chat template appends a bare
   `<think>\n` unless `enable_thinking=False` is passed, which would put
   reasoning text in front of the JSON the pipeline has to parse.

Everything here is backward compatible: defaults reproduce the old behaviour
for the existing models.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import transformers


# ---------------------------------------------------------------------------
# transformers 4.x / 5.x keyword differences
# ---------------------------------------------------------------------------
def _major() -> int:
    try:
        return int(transformers.__version__.split(".")[0])
    except Exception:
        return 4


#: `torch_dtype=` was renamed to `dtype=` in transformers 5.
DTYPE_KW: str = "dtype" if _major() >= 5 else "torch_dtype"


def dtype_kwargs(dtype) -> Dict[str, object]:
    """Return {'dtype': x} or {'torch_dtype': x} for the installed version."""
    return {DTYPE_KW: dtype}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
#: Tried in order. CausalLM first so the existing models keep their exact
#: load path; the VLM wrapper classes are only reached if that fails.
_AUTO_CLASSES: Sequence[str] = (
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
)


def load_decoder_lm(model_path: str, **kwargs):
    """Load a decoder LM, falling back across auto classes.

    Qwen3.5/3.8 need AutoModelForImageTextToText; everything already in this
    project loads via AutoModelForCausalLM exactly as before.
    """
    errors: List[str] = []
    for cls_name in _AUTO_CLASSES:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_path, **kwargs)
            print(f"[pv_model_compat] loaded via {cls_name}")
            return model
        except Exception as e:  # noqa: BLE001 - we want the next candidate
            errors.append(f"{cls_name}: {type(e).__name__}: {e}")
    raise RuntimeError(
        "Could not load model with any auto class:\n  " + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------
#: Substrings identifying the vision tower. Present only in the Qwen3.5 family;
#: harmless for text-only models.
_VISION_HINTS = ("visual", "vision", "image", "patch_embed", "merger")

#: What the scripts used before this module existed.
LEGACY_TARGETS_FULL = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")
LEGACY_TARGETS_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def auto_lora_targets(model) -> List[str]:
    """Every nn.Linear leaf name in the language tower.

    The vision tower uses disjoint names (qkv, proj, linear_fc1, linear_fc2),
    so excluding it needs no filtering beyond the path check below -- there are
    no collisions with language-tower names.
    """
    import torch.nn as nn

    names = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        low = name.lower()
        if any(h in low for h in _VISION_HINTS):
            continue
        leaf = name.split(".")[-1]
        if leaf in ("lm_head", "score", "classifier"):
            continue
        names.add(leaf)
    return sorted(names)


def resolve_lora_targets(spec: Optional[str], model, default: Sequence[str]) -> List[str]:
    """Turn --lora_target_modules into a concrete list.

    spec is None      -> `default` (unchanged legacy behaviour)
    spec == "auto"    -> every language-tower Linear leaf name in `model`
    spec == "a,b,c"   -> exactly those names
    """
    if spec is None or spec == "":
        targets = list(default)
        source = "default"
    elif spec.strip().lower() == "auto":
        targets = auto_lora_targets(model)
        source = "auto (inspected model)"
    else:
        targets = [t.strip() for t in spec.split(",") if t.strip()]
        source = "explicit"

    print(f"[pv_model_compat] LoRA targets ({source}): {targets}")
    _report_coverage(model, targets)
    return targets


def _report_coverage(model, targets: Sequence[str]) -> None:
    """Warn loudly if the target list misses most of the language tower."""
    try:
        import torch.nn as nn
    except Exception:
        return

    total = matched = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        low = name.lower()
        if any(h in low for h in _VISION_HINTS):
            continue
        leaf = name.split(".")[-1]
        if leaf in ("lm_head", "score", "classifier"):
            continue
        total += 1
        if leaf in targets:
            matched += 1

    if not total:
        return
    pct = 100.0 * matched / total
    print(f"[pv_model_compat] language-tower Linear coverage: "
          f"{matched}/{total} ({pct:.1f}%)")
    if pct < 90.0:
        print(f"[pv_model_compat] WARNING: {total - matched} language-tower "
              f"Linear layers are NOT being adapted. Pass "
              f"--lora_target_modules auto if that is not intended.")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
def supports_thinking(tok) -> bool:
    """True if this tokenizer's chat template has a thinking mode."""
    tpl = getattr(tok, "chat_template", None) or ""
    return "enable_thinking" in tpl


def render_prompt(tok, query: str, system_prompt: Optional[str] = None,
                  mode: str = "raw", enable_thinking: bool = False) -> str:
    """Build the generation prompt.

    mode="raw"  -> `query` unchanged (what the existing models were trained on)
    mode="chat" -> chat template with add_generation_prompt=True, and
                   enable_thinking forwarded when the template supports it.
    """
    if mode == "raw":
        return query

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": query})

    kwargs = {}
    if supports_thinking(tok):
        kwargs["enable_thinking"] = enable_thinking

    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, **kwargs
    )


def strip_thinking(text: str) -> str:
    """Drop a leading <think>...</think> block if the model emitted one."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip("\n")
    return text
