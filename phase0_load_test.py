#!/usr/bin/env python
"""
Phase 0 go/no-go test for adding Qwen3.5-9B / Qwen3.8-27B to the PV pipeline.

Answers three questions, cheapest first:

  --stage config   (no GPU, no weight download)
      Does transformers understand `qwen3_5`? How does the chat template render
      for the training path vs the inference path, with thinking on and off?

  --stage modules  (no GPU, no weight download -- meta device)
      What are the real nn.Linear module names? The hybrid linear/full attention
      layers do NOT all expose q_proj/k_proj/v_proj/o_proj, so the hardcoded LoRA
      target lists in sft_peft_ddp.py:156 and train_preference.py:71 will silently
      miss most of the network. This prints what to target instead.

  --stage vllm     (needs GPU + downloads weights)
      Can vLLM actually serve the architecture? This is the hard gate: if it
      cannot, the lm_eval / infer_vllm_and_confusion.py path does not work.

Usage:
    python phase0_load_test.py --model Qwen/Qwen3.5-9B --stage config
    python phase0_load_test.py --model Qwen/Qwen3.5-9B --stage modules
    python phase0_load_test.py --model Qwen/Qwen3.5-9B --stage vllm --tp 2
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import traceback


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def info(msg: str) -> None:
    print(f"         {msg}")


# ----------------------------------------------------------------------------
# Stage 1: config + chat template   (no GPU, no weights)
# ----------------------------------------------------------------------------
def stage_config(model_id: str) -> bool:
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    hr(f"STAGE config -- {model_id}")
    print(f"  transformers {transformers.__version__}")

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        ok(f"AutoConfig loaded: {cfg.__class__.__name__}")
    except Exception as e:
        bad(f"AutoConfig failed: {e}")
        info("=> transformers does not know this architecture yet. Upgrade further.")
        return False

    arch = getattr(cfg, "architectures", None)
    info(f"architectures     : {arch}")
    info(f"model_type        : {getattr(cfg, 'model_type', None)}")

    tcfg = getattr(cfg, "text_config", None)
    if tcfg is not None:
        info(f"text_config type  : {getattr(tcfg, 'model_type', None)}")
        info(f"hidden_size       : {getattr(tcfg, 'hidden_size', None)}")
        info(f"num_hidden_layers : {getattr(tcfg, 'num_hidden_layers', None)}")
        info(f"vocab_size        : {getattr(tcfg, 'vocab_size', None)}")
        lt = getattr(tcfg, "layer_types", None)
        if lt:
            counts = collections.Counter(lt)
            info(f"layer_types       : {dict(counts)}  (total {len(lt)})")
            info("  ^ only the full_attention layers carry standard q/k/v/o_proj")

    # ---- chat template: the part that actually bites -----------------------
    hr("chat template rendering")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        ok("tokenizer loaded")
    except Exception as e:
        bad(f"tokenizer failed: {e}")
        return False

    if not getattr(tok, "chat_template", None):
        bad("no chat_template on this tokenizer")
        return False

    sys_p = "You are a helpful assistant."
    user_p = "Extract the patient voice annotations."
    answer = '{"results": [{"Code": "X", "Sub-code": "y", "Span": "z"}]}'

    # (a) INFERENCE path -- what lm_eval / vLLM send. enable_thinking matters here.
    for flag in (None, True, False):
        kw = {} if flag is None else {"enable_thinking": flag}
        label = "default (unset)" if flag is None else f"enable_thinking={flag}"
        try:
            out = tok.apply_chat_template(
                [{"role": "system", "content": sys_p},
                 {"role": "user", "content": user_p}],
                tokenize=False, add_generation_prompt=True, **kw,
            )
        except Exception as e:
            bad(f"[gen prompt] {label}: {e}")
            continue
        tail = out[-90:].replace("\n", "\\n")
        thinking_open = out.rstrip().endswith("<think>")
        state = "THINKING ON" if thinking_open else "thinking off"
        print(f"  [gen prompt] {label:22s} -> {state}")
        print(f"               ...{tail}")

    # (b) TRAINING path -- train_preference.py:165 style, explicit assistant turn.
    try:
        full = tok.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": user_p},
             {"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False,
        )
        empty_think = "<think>\n\n</think>" in full or "<think></think>" in full
        print(f"\n  [training]   explicit assistant turn -> "
              f"{'empty think block (non-thinking format)' if empty_think else 'NO empty think block'}")
        idx = full.find("assistant")
        print(f"               ...{full[idx:idx + 120]!r}")
        if not empty_think:
            info("  ^ CHECK THIS: training and inference formats may disagree.")
    except Exception as e:
        bad(f"[training] render failed: {e}")

    return True


# ----------------------------------------------------------------------------
# Stage 2: module inventory via meta device   (no GPU, no weights)
# ----------------------------------------------------------------------------
def stage_modules(model_id: str) -> bool:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig

    hr(f"STAGE modules -- {model_id}")
    print("  Instantiating on meta device: no weights downloaded, no GPU used.")

    cfg = AutoConfig.from_pretrained(model_id)

    model = None
    tried = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM"):
        try:
            import transformers
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                tried.append(f"{cls_name}: not present in this transformers")
                continue
            with torch.device("meta"):
                model = cls.from_config(cfg)
            ok(f"instantiated via {cls_name}")
            break
        except Exception as e:
            tried.append(f"{cls_name}: {type(e).__name__}: {e}")

    if model is None:
        bad("could not instantiate from config with any auto class")
        for t in tried:
            info(t)
        return False

    for t in tried:
        info(f"(also tried) {t}")

    # Collect every Linear leaf, split language tower vs vision tower.
    lang, vision, other = collections.Counter(), collections.Counter(), collections.Counter()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        low = name.lower()
        if any(k in low for k in ("visual", "vision", "image", "patch_embed", "merger")):
            vision[leaf] += 1
        elif any(k in low for k in ("language_model", "model.layers", "text_model", "decoder")):
            lang[leaf] += 1
        else:
            other[leaf] += 1

    def dump(title, counter):
        hr(title)
        if not counter:
            print("  (none)")
            return
        for leaf, n in counter.most_common():
            print(f"  {leaf:28s} x{n}")

    dump("LANGUAGE tower nn.Linear leaf names  <- LoRA targets come from here", lang)
    dump("VISION tower nn.Linear leaf names    <- exclude these for text-only", vision)
    dump("OTHER / unclassified", other)

    current = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    hr("verdict vs the hardcoded LoRA target lists")
    print(f"  train_preference.py:71 targets : {current}")
    print(f"  sft_peft_ddp.py:156 targets    : {current[:4]}")
    matched = {k: lang[k] for k in current if lang.get(k)}
    missed = sorted(set(lang) - set(current))
    total = sum(lang.values())
    cov = sum(matched.values())
    print(f"\n  language-tower Linear layers total : {total}")
    print(f"  covered by current target list     : {cov}"
          f"  ({100.0 * cov / total:.1f}%)" if total else "")
    if missed:
        print(f"\n  NOT TARGETED (add these):")
        for m in missed:
            print(f"    {m:28s} x{lang[m]}")
        print("\n  => add a --lora_target_modules flag and pass the full list.")
    else:
        ok("current target list covers every language-tower Linear")
    return True


# ----------------------------------------------------------------------------
# Stage 3: vLLM serve   (GPU + weight download)  -- THE HARD GATE
# ----------------------------------------------------------------------------
def stage_vllm(model_id: str, tp: int, max_model_len: int) -> bool:
    hr(f"STAGE vllm -- {model_id}  (tp={tp})")
    try:
        import vllm
        print(f"  vllm {vllm.__version__}")
        from vllm import LLM, SamplingParams
    except Exception as e:
        bad(f"vllm import failed: {e}")
        return False

    try:
        llm = LLM(model=model_id, tensor_parallel_size=tp,
                  max_model_len=max_model_len, trust_remote_code=True)
        ok("vLLM loaded the model -- eval path is viable")
    except Exception as e:
        bad(f"vLLM could not load: {type(e).__name__}: {e}")
        traceback.print_exc()
        info("=> HARD GATE FAILED. lm_eval + infer_vllm_and_confusion.py cannot")
        info("   run this architecture on this vllm build. Stop and rethink the")
        info("   eval path before touching any pipeline code.")
        return False

    tok = llm.get_tokenizer()
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content":
             'Return only JSON: {"results": [{"Code": "A", "Sub-code": "b", "Span": "c"}]}'}]

    for flag in (True, False):
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=flag)
        out = llm.generate(
            [prompt], SamplingParams(temperature=0.0, max_tokens=256))[0].outputs[0].text
        hr(f"generation with enable_thinking={flag}")
        print(f"  chars={len(out)}  contains <think>={'<think>' in out}")
        print("  " + out[:400].replace("\n", "\n  "))
        if flag is False and "<think>" in out:
            bad("thinking leaked through despite enable_thinking=False")

    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["config", "modules", "vllm", "all"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    args = ap.parse_args()

    results = {}
    if args.stage in ("config", "all"):
        results["config"] = stage_config(args.model)
    if args.stage in ("modules", "all"):
        results["modules"] = stage_modules(args.model)
    if args.stage in ("vllm", "all"):
        results["vllm"] = stage_vllm(args.model, args.tp, args.max_model_len)

    hr("SUMMARY")
    for k, v in results.items():
        print(f"  {k:10s} {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
