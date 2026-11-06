#!/usr/bin/env python3
# tools/mm3_dflash2_fixture.py - generate the synthetic DFlash2 draft fixture the Metal
# mm3 DFlash2 gates run against (metal_mm3_dflash2_test): a tiny random-weights draft in
# the real checkpoint layout (config.json + model.safetensors under <out>/dflash2), the
# target emb/head pair the test hands the draft, and the torch reference's expected
# context/proposal outputs (expected.bin / expected.json).
#
# The generator itself is the reference implementation, tools/mm3/dflash2_ref.py;
# this is only the cmake-facing entry point (ctest calls it with --out).
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "mm3"))

import dflash2_ref

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dflash2_ref.fixture(args.out)
