#!/usr/bin/env python3
"""tools/hy4_metal_e2e.py - stage 4's gate for maya-metal-hy4: greedy end-to-end on the
stage-3 block fixture pack.

  1. `--ids` on the Metal forward and on the scalar CPU reference (--cpu) must continue
     the prompt with the SAME token ids (token-for-token, greedy).
  2. `--serve` must speak the serve protocol (INFO/READY, PP/T/DONE) and its greedy GEN
     must produce the same chain; a follow-up GEN that extends the conversation must
     report the prefix reuse in DONE and still produce the reference's continuation.

Usage: hy4_metal_e2e.py <maya-metal-hy4> <pack-dir>
"""
import subprocess
import sys

PROMPT = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# (not the block test's prompt: its position-10 router scores are a proven dead tie -
# stage 3's gate prints the flip with gaps 0.0006/0.0000 - and a free-running chain past
# a tie compares two different valid continuations.  This prompt's chain is tie-free,
# so the comparison below is exact, token for token.)
N = 24


def run_ids(exe, pack, cpu):
    cmd = [exe, "--pack", pack, "--ids", ",".join(map(str, PROMPT)), "--n", str(N)]
    if cpu:
        cmd.append("--cpu")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise SystemExit(f"--ids {'--cpu ' if cpu else ''}failed ({p.returncode}):\n{p.stderr[-2000:]}")
    for line in p.stdout.splitlines():
        if line.startswith("tokens:"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    raise SystemExit(f"no tokens: line in --ids output:\n{p.stdout}\n{p.stderr[-1000:]}")


class Serve:
    def __init__(self, exe, pack):
        self.p = subprocess.Popen([exe, "--pack", pack, "--serve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.ready = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("engine exited before READY")
            if line.startswith("READY"):
                self.ready = line.split()
                return

    def gen(self, max_new, ids, keys=""):
        self.p.stdin.write(f"GEN {max_new}{keys} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        toks, done = [], None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("engine exited mid-request")
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                done = line.split()
                return toks, done
            elif line.startswith("ERR"):
                raise SystemExit(f"engine error: {line.strip()}")

    def quit(self):
        self.p.stdin.write("QUIT\n")
        self.p.stdin.flush()
        rc = self.p.wait(timeout=60)
        if rc != 0:
            raise SystemExit(f"engine exit code {rc} after QUIT")


def main():
    exe, pack = sys.argv[1], sys.argv[2]
    gpu = run_ids(exe, pack, cpu=False)
    ref = run_ids(exe, pack, cpu=True)
    if gpu != ref:
        raise SystemExit(f"greedy E2E mismatch:\n  metal: {gpu}\n  cpu:   {ref}")
    print(f"ids e2e: metal == cpu reference over {len(gpu)} greedy tokens: {gpu}")

    s = Serve(exe, pack)
    if len(s.ready) < 2 or int(s.ready[1]) <= 0 or "stop" not in s.ready[2:]:
        raise SystemExit(f"bad READY line: {s.ready}")
    toks, done = s.gen(N, PROMPT)
    want = ref[:len(toks)]
    if toks != want:
        raise SystemExit(f"serve GEN mismatch:\n  serve: {toks}\n  ids:   {want}")
    if done[5] == "length" and len(toks) != N:
        raise SystemExit(f"finish=length but only {len(toks)} tokens")
    if done[5] == "eos" and toks[-1] != 257:
        raise SystemExit(f"finish=eos but last token is {toks[-1]}, not the pack's eos 257")
    print(f"serve gen: {len(toks)} tokens, finish={done[5]}, reused={done[8]}")

    # a follow-up GEN extending the conversation by the first 4 emitted tokens: DONE must
    # report the reuse the prefix logic gives (the whole follow-up prompt, unless GEN1's
    # unfed last token bounds it), and the continuation must match the ids chain
    cont_prompt = PROMPT + toks[:4]
    k = min(4, len(toks))
    want_reuse = min(len(cont_prompt), len(PROMPT) + len(toks) - 1)
    toks2, done2 = s.gen(4, cont_prompt)
    if int(done2[8]) != want_reuse:
        raise SystemExit(f"follow-up GEN reused {done2[8]}, want {want_reuse}")
    if toks2 != gpu[k:k + 4]:
        raise SystemExit(f"follow-up GEN mismatch:\n  serve: {toks2}\n  ids:   {gpu[k:k + 4]}")
    print(f"serve follow-up: reused={done2[8]}, continuation {toks2} == the ids chain")
    s.quit()
    print("E2E PASSED")


if __name__ == "__main__":
    main()
