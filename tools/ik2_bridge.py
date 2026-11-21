#!/Users/duckets/maya-venv/bin/python
"""ik2 bridge: a Maya engine for Inkling 2 Mini on the Mac.

Maya's serve stack talks to its engines over stdin/stdout (`GEN <max_new>
[key=value ...] <token ids>` in, `T <id>` lines and a final
`DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish>` out).
The native maya-ik2 engine is CUDA-only, so on the Mac this bridge stands
in its place: serve spawns it like any other engine, and it drives a
llama-server child (ggml/Metal, the Q4_K_M GGUF) over HTTP, translating
between the two protocols.

Inkling writes typed segments (<|content_thinking|>, <|content_text|>,
<|end_message|>). This bridge serves only the content_text segment to
Maya: the thinking happens, it just is not shown, and tool-call segments
are dropped - Maya's own frontend has no Inkling segment parser. Plain
chat through Maya's regular web UI is the job here.

Stdlib only. Args (after serve's own --serve): --llama-bin, --gguf,
--llama-port (the child's private port), --ctx, --alias.
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
END = "<|end_message|>"
HOLDBACK = 6          # token ids kept un-emitted at the tail (BPE can still merge there)
OUT = threading.Lock()


def emit(line: str) -> None:
    with OUT:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class Llama:
    def __init__(self, bin_path: str, gguf: str, port: int, ctx: int, alias: str,
                 extra: list | None = None):
        self.base = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [bin_path, "-m", gguf, "--host", "127.0.0.1", "--port", str(port),
             "--ctx-size", str(ctx), "--n-gpu-layers", "999",
             "--alias", alias, "--jinja"] + (extra or []),
            stdout=subprocess.DEVNULL, stderr=sys.stderr)
        deadline = time.time() + 240
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not become healthy in 240 s")

    def post(self, path: str, payload: dict, timeout: float = 600):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def tokenize(self, text: str) -> list:
        with self.post("/tokenize", {"content": text}) as r:
            return json.load(r)["tokens"]

    def detokenize(self, ids: list) -> str:
        with self.post("/detokenize", {"tokens": ids}) as r:
            return json.load(r)["content"]

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()



class Generation:
    def __init__(self, llama: Llama):
        self.llama = llama
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.close_sent = False

    def start(self, max_new: int, sampling: dict, ids: list) -> None:
        self.stop.clear()
        self.close_sent = False
        self.thread = threading.Thread(target=self._run, args=(max_new, sampling, ids),
                                       daemon=True)
        self.thread.start()

    def _run(self, max_new: int, sampling: dict, ids: list) -> None:
        llama = self.llama
        try:
            if llama.proc.poll() is not None:
                emit("ERR the ik2 bridge's llama-server is gone")
                return
            prompt = llama.detokenize(ids)
            body = {"prompt": prompt, "n_predict": max_new, "stream": True,
                    "cache_prompt": False, "return_tokens": True}
            for key in ("temperature", "top_p", "top_k", "seed"):
                if key in sampling:
                    body[key] = sampling[key]
            # Segment by token id: llama-server's text stream strips the
            # special segment tokens, but return_tokens carries them.
            # 200008 opens the thinking segment, 200004 the text segment;
            # any other special closes/opens something we do not serve
            # (tool calls, the message end). Only text-segment ids go out,
            # preceded once by a literal </think> when thinking came first:
            # Maya's OutputParser opens in the reasoning state and that text
            # is what closes it (the repo frontend's InklingRewriter makes
            # the same translation from <|content_text|>).
            state, saw_thinking, served, final = "pre", False, 0, {}
            with llama.post("/completion", body) as resp:
                for line in resp:
                    if self.stop.is_set():
                        break
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:])
                    for tid in data.get("tokens", []):
                        if tid == 200008:
                            state, saw_thinking = "thinking", True
                        elif tid == 200004:
                            if saw_thinking and not self.close_sent:
                                for cid in llama.tokenize("</think>"):
                                    emit(f"T {cid}")
                                self.close_sent = True
                            state = "text"
                        elif tid >= 200000:
                            state = "other"
                        elif state == "text":
                            emit(f"T {tid}")
                            served += 1
                    if data.get("stop"):
                        final = data
                        break
            timings = final.get("timings", {})
            generated = int(final.get("tokens_predicted", 0)) or served
            prompt_tokens = int(timings.get("prompt_n", len(ids)))
            finish = "length" if final.get("stopped_limit") else "stop"
            emit(f"DONE {generated} {prompt_tokens} "
                 f"{timings.get('prompt_ms', 0):.0f} {timings.get('predicted_ms', 0):.0f} "
                 f"{finish}")
        except Exception as e:                              # noqa: BLE001 - reported to serve
            emit(f"ERR ik2 bridge: {e}")


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--serve":
        args = args[1:]
    opts = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            opts[args[i][2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    # the official config's own args name the model and its context the
    # engine's way (--model, --max-context); the bridge's are --gguf/--ctx
    ctx = int(opts.get("ctx") or opts.get("max-context", "32768"))
    gguf = opts.get("gguf") or opts.get("model")
    # any --llama-<flag> <value> config arg is passed to llama-server as
    # --<flag> <value> (speculative decoding, cache types, ...)
    extra = []
    for k, v in opts.items():
        if k.startswith("llama-") and k not in ("llama-bin", "llama-port"):
            extra += ["--" + k[len("llama-"):], v]
    try:
        llama = Llama(opts["llama-bin"], gguf,
                      int(opts.get("llama-port", "18085")), ctx,
                      opts.get("alias", "inkling-2-mini"), extra)
    except Exception as e:                                  # noqa: BLE001 - serve reads READY or dies
        print(f"ik2 bridge: {e}", file=sys.stderr, flush=True)
        return 1
    emit("NOTE ik2 bridge: Inkling 2 Mini via llama-server (ggml/Metal)")
    emit("INFO engine=ik2-bridge-1.0")
    emit(f"READY {ctx} stop")
    gen = Generation(llama)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line.startswith("GENI"):
                # an image request: serve hands engines an embeddings file,
                # which llama-server cannot consume (it encodes pixels
                # itself). Fail the request cleanly instead of hanging it.
                emit("ERR ik2 bridge: images are not supported through the bridge")
            elif line.startswith("GEN "):
                if gen.thread and gen.thread.is_alive():
                    gen.thread.join()
                parts = line.split(" ")
                max_new = int(parts[1])
                sampling = {}
                for kv in parts[2:-1]:
                    k, _, v = kv.partition("=")
                    try:
                        sampling[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
                ids = [int(t) for t in parts[-1].split(",") if t]
                gen.start(max_new, sampling, ids)
            elif line == "STOP":
                gen.stop.set()
            elif line == "QUIT":
                break
            # anything else (WRAP, PAUSE, ...): not advertised in READY, ignored
    finally:
        gen.stop.set()
        llama.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
