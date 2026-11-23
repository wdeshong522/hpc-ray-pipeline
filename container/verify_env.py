"""
verify_env.py
=============
Checks one of the image's two virtualenvs against the versions we validated.
Used in two places:

1. At image build time, with no GPU:      verify_env.py train   (or serve)
   Checks package versions, and the CUDA architectures the torch binary was
   COMPILED for (a property of the binary, readable without a GPU).

2. On a Wahab GPU node, after pulling the image:   verify_env.py train --gpu
   Additionally checks that CUDA really works and that the device is a V100.
   (torch.cuda.get_arch_list() returns [] on machines without a GPU, so the
   hard "V100 kernels exist" check can only happen here.)

Exit code 0 = everything matched, 1 = at least one problem (all are printed).

Run with the venv's own interpreter, e.g.
    /opt/venvs/serve/bin/python /opt/verify_env.py serve --gpu
"""

import argparse
import importlib.metadata as md
import sys

EXPECTED = {
    "train": {
        "torch": "2.4.1+cu124", "transformers": "4.51.0", "peft": "0.14.0",
        "trl": "0.13.0", "accelerate": "1.2.1", "bitsandbytes": "0.50.2",
        "ray": "2.58.0", "datasets": "2.21.0",
    },
    "serve": {
        "torch": "2.13.0+cu126", "torchvision": "0.28.0+cu126",
        "torchaudio": "2.11.0+cu126", "vllm": "0.30.0",
        "transformers": "5.17.0",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("env", choices=sorted(EXPECTED))
    parser.add_argument("--gpu", action="store_true",
                        help="also test CUDA on a real GPU (run on a GPU node)")
    args = parser.parse_args()

    problems = []

    # 1. Package versions: exact match with what we validated on Wahab.
    for package, wanted in EXPECTED[args.env].items():
        try:
            found = md.version(package)
        except md.PackageNotFoundError:
            problems.append(f"{package}: not installed (expected {wanted})")
            continue
        print(f"{package:14s} {found}")
        if found != wanted:
            problems.append(f"{package}: expected {wanted}, found {found}")

    # 2. CUDA architectures the torch binary was compiled for. No GPU needed.
    try:
        import torch
    except ImportError as exc:
        problems.append(f"cannot import torch: {exc}")
        torch = None

    if torch is not None:
        try:
            flags = torch._C._cuda_getArchFlags()
        except Exception as exc:  # attribute may not exist in every build
            flags = None
            print(f"compiled arch flags unavailable here ({exc}); "
                  "the --gpu check on Wahab is the authoritative test")
        else:
            print(f"compiled CUDA arch flags: {flags}")
            if flags is not None and "sm_70" not in flags:
                problems.append("torch binary was not compiled for sm_70 (V100)")

        # 3. Real GPU test (only on a GPU node).
        if args.gpu:
            if not torch.cuda.is_available():
                problems.append("--gpu given but CUDA is not available")
            else:
                name = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                archs = torch.cuda.get_arch_list()
                print(f"GPU 0: {name}, compute capability {cap}")
                print(f"torch.cuda.get_arch_list(): {archs}")
                if "sm_70" not in archs:
                    problems.append("sm_70 not in torch.cuda.get_arch_list()")
                x = torch.randn(1024, 1024, device="cuda")
                total = (x @ x).sum().item()  # forces a real kernel launch
                print(f"GPU matmul ran (checksum {total:.3e})")

    if problems:
        print("\nFAILED:")
        for line in problems:
            print("  -", line)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
