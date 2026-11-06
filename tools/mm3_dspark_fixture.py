#!/usr/bin/env python3
# tools/mm3_dspark_fixture.py - generate the synthetic DSpark draft fixture the Metal
# mm3 DSpark gates run against (metal_mm3_dspark_test, metal_mm3_divergence_test):
# a tiny random-weights draft in the real safetensors layout (config.json +
# model-dspark-00001.safetensors under <out>/dspark), the target emb/head pair the
# test hands the draft, and the torch reference's expected context/proposal outputs
# (expected.bin / expected.json).
#
# The generator itself is the reference implementation, tools/mm3/dspark_ref.py;
# this is only the cmake-facing entry point (ctest calls it with --out).
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "mm3"))

import dspark_ref

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dspark_ref.fixture(args.out)
