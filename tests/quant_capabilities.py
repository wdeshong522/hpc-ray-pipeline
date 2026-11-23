"""List which vLLM quantization methods accept a V100 (compute capability 70).

Needs no GPU: only reads each method's declared minimum capability.
Proposal link: decides whether any 8-bit route exists for 1-GPU judging.
"""
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS, get_quantization_config)

for name in QUANTIZATION_METHODS:
    try:
        cap = get_quantization_config(name).get_min_capability()
        note = "OK on V100" if cap <= 70 else ""
        print(f"{name:32s} min_capability={cap:3d}  {note}")
    except Exception as e:                      # report, don't hide
        print(f"{name:32s} could not read: {type(e).__name__}: {str(e)[:80]}")
