"""Merge a LoRA adapter into the base model once, so vLLM can serve plain weights."""
import argparse
import os
import time

from src.common.timing import RunRecorder


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--run_id", default="merge")
    return p.parse_args(argv)


def merge(model_name, adapter, out):
    """Merge in float32 on CPU, then save as float16."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model = model.to(torch.float16)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)


def main(argv=None):
    args = parse_args(argv)
    rec = RunRecorder(results_dir=args.results_dir, run_id=args.run_id,
                      stage="merge", tag="merge",
                      params={"adapter": args.adapter, "out": args.out})
    t0 = time.time()
    with rec.interval("processing"):
        merge(args.model_name, args.adapter, args.out)
    print(f"merged {args.adapter} -> {args.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
