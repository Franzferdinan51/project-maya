"""tools/test_mm3_serve_restore.py - maya-metal-mm3 --serve restore points (P5, ported from the GLM
engine) on the synthetic fixture: a request can record a prefix state at shared=/marks= points, and
a later request that starts with that prefix resumes from it (DONE's reused field is the point, not
0) and writes exactly the tokens the uncached path writes.  A 39-of-40-token near miss reuses
nothing.  The same battery runs twice: drafting off (spec=1) and drafting on (spec=2) - the synth
fixture loads no DSpark draft, so the spec-on pass proves the keys and the restore compose; the
real-model DSpark identity gate is the live A/B in the nightly record.

    python tools/test_mm3_serve_restore.py MAYA_METAL_MM3 GGUF

Run it directly against a built maya-metal-mm3 (the mm3 synth GGUF is the metal_mm3_gen fixture).
"""
import subprocess
import sys

S = list(range(11, 51))            # the shared preamble: 40 tokens
P1 = S + list(range(101, 125))     # conversation 1 behind S
P2 = S + list(range(151, 175))     # conversation 2 behind S (diverges at 40)
P3 = S + list(range(201, 225))     # conversation 3 (the marks= recorder)
P4 = S + list(range(231, 255))     # conversation 4 (resumes from the marks= entry, carries no keys)
P5 = S[:39] + [99] + list(range(101, 121))   # shares 39 of the 40: must reuse nothing
P6 = S[:30] + list(range(251, 285))          # branches at 30: the marks=20 entry is its deepest match
U = [301] + list(range(302, 340))  # an unrelated prompt: flushes the snapshot and the entries
NEW = 16


class Engine:
    def __init__(self, exe: str, gguf: str):
        self.p = subprocess.Popen([exe, "--serve", "--gguf", gguf, "--max-context", "256"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for line in self.p.stdout:
            if line.startswith("READY"):
                self.flags = line.split()[2:]
                return
        raise SystemExit("maya-metal-mm3 exited before READY")

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def gen(self, ids: list, keys: str = "", spec: int = 1):
        """(tokens, DONE fields) - greedy, so only the prefix machinery can differ"""
        self.send(f"GEN {NEW} spec={spec} {keys} {','.join(map(str, ids))}".replace("  ", " "))
        toks = []
        for line in self.p.stdout:
            if line.startswith("T "):
                toks.append(int(line[2:]))
            elif line.startswith("DONE"):
                f = line.split()
                assert f[5] in ("length", "stop"), f   # the synth model may end a prompt early; identity is what is gated
                return toks, f
            elif line.startswith("ERR"):
                raise SystemExit(line.strip())
        raise SystemExit("maya-metal-mm3 exited mid-request")

    def close(self):
        self.send("QUIT")
        self.p.wait(timeout=60)


def reused(done) -> int:
    return int(done[8])


def battery(e: Engine, spec: int) -> None:
    # the uncached control for P1, then a flush so the recording request starts cold
    a0, d = e.gen(P1, spec=spec)
    assert reused(d) == 0, d
    e.gen(U, spec=spec)

    # recording shared=40 writes the same tokens as the control, and starts cold
    a, d = e.gen(P1, "shared=40", spec=spec)
    assert a == a0, "recording a restore point changed the tokens"
    assert reused(d) == 0, d

    # conversation 2 behind the same preamble: the snapshot (P1 less its last token) cannot match,
    # so a reuse of exactly 40 is the recorded entry resuming it - the prefill of the preamble is skipped
    b, d = e.gen(P2, "shared=40", spec=spec)
    assert reused(d) == 40, f"the branched request reused {reused(d)} tokens, want 40"
    print(f"test_mm3_serve_restore: shared point resumed (reused {reused(d)} of {len(P2) - 1} prefill tokens)")

    # a near miss (39 of 40 shared) reuses nothing
    _, d = e.gen(P5, spec=spec)
    assert reused(d) == 0, f"a 39-of-40 near miss reused {reused(d)} tokens"

    # the same conversation 2 with no cache anywhere: the restored run's tokens, token for token
    e.gen(U, spec=spec)
    c, d = e.gen(P2, spec=spec)
    assert reused(d) == 0, d
    assert c == b, "the restored request wrote other tokens than the uncached path"

    # marks= records too, and a later request with no keys at all resumes from the entry
    _, d = e.gen(P3, "marks=40", spec=spec)
    assert reused(d) == 0, d
    m2, d = e.gen(P4, spec=spec)
    assert reused(d) == 40, f"the marks entry resumed {reused(d)} tokens, want 40"
    e.gen(U, spec=spec)
    m3, d = e.gen(P4, spec=spec)
    assert reused(d) == 0 and m3 == m2, "the marks-restored tokens differ from the uncached path"

    # two points in one request: a branch at 30 falls back to the shallower entry (deepest match wins)
    e.gen(U, spec=spec)
    _, d = e.gen(P1, "marks=20,40", spec=spec)
    assert reused(d) == 0, d
    x, d = e.gen(P6, spec=spec)
    assert reused(d) == 20, f"the branched request reused {reused(d)} tokens, want the 20-deep entry"
    e.gen(U, spec=spec)
    x0, d = e.gen(P6, spec=spec)
    assert reused(d) == 0 and x0 == x, "the shallow-restored tokens differ from the uncached path"


def main() -> int:
    exe, gguf = sys.argv[1], sys.argv[2]
    e = Engine(exe, gguf)
    assert "stop" in e.flags and "pause" in e.flags, e.flags
    assert "restore" in e.flags, f"READY does not advertise restore: {e.flags}"

    battery(e, spec=1)   # drafting off
    battery(e, spec=2)   # drafting on (no draft on the synth fixture: keys compose, tokens hold)

    e.close()
    print("test_mm3_serve_restore: PASS (shared + marks entries resume at their exact points, "
          "tokens identical to the uncached path, near miss reuses 0, spec off and on)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
