#!/usr/bin/env python
"""
PV extraction evaluation with vLLM directly -- no lm_eval, no FinBen.

Reproduces what `lm_eval --tasks PvExtraction_full --num_fewshot 0
--apply_chat_template` did (FinBen tasks/pv_miner/PvExtraction.yaml):

  prompt   : chat template over a single user turn containing ex["query"]
  decoding : greedy, max 1024 new tokens, stop on "JSON_END" or EOS
  scoring  : pv_utils.evaluate_eppc_agg (code / sub-code micro F1, relaxed span F1)

Two modes:

  generate + score (GPU):
    python evaluate_pv.py --model <merged model dir> --out_dir <dir> --tp 2

  score only (CPU) -- rescore a preds.jsonl written by this script, or an
  lm_eval samples_*.jsonl written with --log_samples:
    python evaluate_pv.py --score_only <file.jsonl> --out_dir <dir>

Writes <out_dir>/preds.jsonl and <out_dir>/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import pv_utils
from pv_model_compat import render_prompt, strip_thinking

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.normpath(
    os.path.join(HERE, "..", "benckmark", "PV_benckmark", "split_out", "test"))

# PvExtraction.yaml generation_kwargs
STOP = ["JSON_END"]
MAX_NEW_TOKENS = 1024


def load_scored_pairs(path: str) -> List[Dict[str, Any]]:
    """Read (target, prediction) rows from our preds.jsonl or lm_eval samples."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "filtered_resps" in r:  # lm_eval --log_samples format
                rows.append({"doc_id": r["doc_id"], "target": r["target"],
                             "pred": r["filtered_resps"][0]})
            else:
                rows.append(r)
    return rows


def score(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return pv_utils.evaluate_eppc_agg([[r["target"], r["pred"]] for r in rows])


def encode(tok, prompt: str) -> List[int]:
    # Same rule as lm_eval's vLLM backend: a chat-rendered prompt that already
    # starts with BOS must not get a second one; otherwise use tokenizer defaults.
    bos = tok.bos_token
    if bos and prompt.startswith(bos):
        return tok(prompt, add_special_tokens=False).input_ids
    return tok(prompt).input_ids


def generate(args) -> List[Dict[str, Any]]:
    from datasets import load_from_disk
    from vllm import LLM, SamplingParams

    ds = load_from_disk(args.data)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()

    prompts = [render_prompt(tok, ex["query"], system_prompt=args.system_prompt,
                             mode=args.prompt_mode,
                             enable_thinking=args.enable_thinking)
               for ex in ds]
    token_ids = [encode(tok, p) for p in prompts]

    # Keep room for the full generation budget, truncating the prompt from the
    # left if needed (lm_eval does the same).
    max_prompt = args.max_model_len - args.max_new_tokens
    n_trunc = sum(len(t) > max_prompt for t in token_ids)
    token_ids = [t[-max_prompt:] for t in token_ids]
    if n_trunc:
        print(f"WARNING: {n_trunc} prompts truncated to {max_prompt} tokens", file=sys.stderr)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=STOP + [tok.decode(tok.eos_token_id)],
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )

    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": t} for t in token_ids], sp)
    print(f"generated {len(outs)} in {time.time() - t0:.1f}s", file=sys.stderr)

    rows = []
    for i, (ex, prompt, out) in enumerate(zip(ds, prompts, outs)):
        raw = out.outputs[0].text if out.outputs else ""
        rows.append({
            "doc_id": i,
            "prompt": prompt,
            "raw": raw,
            "pred": strip_thinking(raw),
            "target": ex["answer"],
            "finish_reason": out.outputs[0].finish_reason if out.outputs else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF model dir or hub id (generate mode)")
    ap.add_argument("--score_only", help="Rescore this jsonl instead of generating")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA, help="load_from_disk test split")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")

    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--prompt_mode", choices=["chat", "raw"], default="chat",
                    help="chat = lm_eval --apply_chat_template (all existing evals)")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Off by default. Qwen3.5 thinking output breaks JSON parsing.")
    ap.add_argument("--system_prompt", default=None)
    args = ap.parse_args()

    if bool(args.model) == bool(args.score_only):
        ap.error("pass exactly one of --model or --score_only")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_scored_pairs(args.score_only) if args.score_only else generate(args)

    if args.model:
        with open(os.path.join(args.out_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = score(rows)
    summary = {
        "metrics": metrics,
        "n": len(rows),
        "source": args.score_only or args.model,
        "config": {k: v for k, v in vars(args).items() if k not in ("score_only", "model")},
    }
    if args.model:
        summary["n_hit_max_tokens"] = sum(r.get("finish_reason") == "length" for r in rows)
        import vllm, transformers
        summary["versions"] = {"vllm": vllm.__version__, "transformers": transformers.__version__}

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
