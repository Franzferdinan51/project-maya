#!/usr/bin/env python3
"""tools/hy4_metal_fixture.py - the Hy4 Metal port's fixture: a synthetic Hy4-flash GGUF set,
packed by the real packer (the port plan's stage 1).

The set is the packer test's own (tools/test_hy4_pack_gguf.py's model(): 3 layers + the MTP
block in the set, layer 0 dense and indexed, 1 and 2 sliding-window, Engram at layer 1, toy
widths), with one change: expert_latent_length 128 -> 256. The C++ loader
(src/hy4/hy4_pack.hpp) refuses a latent width that is not a whole number of 256s - a rule the
packer's own geometry check does not have, and the real model's width (1536) satisfies; the
packer test's toy 128 does not, so a pack of the unmodified set loads in Python and is refused
by the engine. Everything else is the test's set, byte for byte.

    python tools/hy4_metal_fixture.py --out <model dir>     # the pack lands in <out>/maya-hy4
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="the model folder to write the set into")
    a = ap.parse_args()
    root = pathlib.Path(a.out)

    spec = importlib.util.spec_from_file_location("hy4_pack_gguf_test", HERE / "test_hy4_pack_gguf.py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)

    test.model(root, **{test.ARCH + ".expert_latent_length": (4, 256)})

    r = subprocess.run([sys.executable, str(HERE / "hy4" / "pack_gguf.py"), "--model", str(root)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return r.returncode
    pack = root / "maya-hy4"
    for name in ("hy4.txt", "tensors.txt", "engram.bin"):
        if not (pack / name).is_file():
            print("%s: the packer wrote no %s" % (pack, name), file=sys.stderr)
            return 1
    print("%s: fixture pack ready" % pack)
    return 0


if __name__ == "__main__":
    sys.exit(main())
