"""Smoke test: can vLLM serve an 8B judge in 8-bit on ONE 16 GB V100?

Proposal link: the judging stage uses 1-GPU replicas only if 8-bit is
validated; otherwise 2-GPU pipeline-parallel replicas.  Scope: this script
only answers "does it load, fit, and generate sensibly?"  It is not a
benchmark and not a quality evaluation.

Usage:  python smoke_8bit.py MODEL QUANT
  MODEL  Hugging Face id, e.g. Qwen/Qwen3-8B
  QUANT  a vLLM quantization method name (e.g. int8_per_channel_weight_only)
         or 'none' (fp16 control).  bitsandbytes is NOT accepted by vLLM 0.30.0.
"""
import subprocess
import sys
import time


def gpu_mem_mib():
    """Used GPU memory in MiB as reported by nvidia-smi (all processes)."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True)
    return int(out.strip().splitlines()[0])


def main():
    model, quant = sys.argv[1], sys.argv[2]
    result = {"model": model, "quant": quant, "loaded": False}
    try:
        from vllm import LLM, SamplingParams

        kwargs = dict(model=model, dtype="float16",   # V100: no bf16
                      max_model_len=4096,
                      gpu_memory_utilization=0.90)
        if quant != "none":                       # any vLLM method name
            kwargs["quantization"] = quant

        t0 = time.perf_counter()
        llm = LLM(**kwargs)
        result["load_s"] = round(time.perf_counter() - t0, 1)
        result["loaded"] = True
        result["gpu_mib_after_load"] = gpu_mem_mib()

        prompts = [
            "Rate this dental hygiene feedback from 1 to 5 and explain in "
            "one sentence: 'Brush twice a day and floss daily.'",
            "What is 17 * 23? Answer with just the number.",
        ]
        sp = SamplingParams(temperature=0.0, max_tokens=64)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        result["gen_s"] = round(time.perf_counter() - t0, 2)
        for o in outs:
            print("PROMPT:", o.prompt[:60])
            print("OUTPUT:", o.outputs[0].text.strip())
        result["sane"] = "391" in outs[1].outputs[0].text
    except Exception as e:                       # report, don't hide
        result["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
    print("RESULT", result)


if __name__ == "__main__":
    main()