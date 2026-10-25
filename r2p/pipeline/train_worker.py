"""Launched only after lease validation; enforce the leased VRAM cap before training."""

import os
import runpy
import sys


def main():
    argv = sys.argv[1:]
    device = argv[argv.index("--device") + 1] if "--device" in argv else "cpu"
    if device != "cpu" and (not os.getenv("LANDLORD_LEASE_ID") or not os.getenv("R2P_LEASE_VRAM_GB")):
        raise RuntimeError("Accelerator training requires a verified landlord lease")
    if device == "cuda":
        import torch
        cap = float(os.environ["R2P_LEASE_VRAM_GB"]) * 1024 ** 3
        for index in range(torch.cuda.device_count()):
            fraction = min(1.0, cap / torch.cuda.get_device_properties(index).total_memory)
            torch.cuda.set_per_process_memory_fraction(fraction, index)
            if index == 0:
                # B's trainer applies the same cap again; pass the derived lease fraction explicitly.
                os.environ["LANDLORD_MEMORY_FRACTION"] = str(fraction)
                argv += ["--memory-fraction", str(fraction)]
    sys.argv = ["r2p.policy.train", *argv]
    runpy.run_module("r2p.policy.train", run_name="__main__")


if __name__ == "__main__":
    main()
