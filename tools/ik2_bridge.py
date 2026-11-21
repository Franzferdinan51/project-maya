#!/Users/duckets/maya-venv/bin/python
"""ik2 bridge: a Maya engine for Inkling 2 Mini on the Mac.

Maya's serve stack talks to its engines over stdin/stdout (`GEN <max_new>
[key=value ...] <token ids>` in, `T <id>` lines and a final
`DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish>` out).
The native maya-ik2 engine is CUDA-only, so on the Mac this bridge stands
in its place: serve spawns it like any other engine, and it drives one of
two local backends, translating between the two protocols:

- llama (default): a llama-server child (ggml/Metal) over HTTP.
- mlx (--backend mlx): mlx-vlm in-process (the inkling2-mlx fork),
  running netprojects' MLX build of the model with its bundled DSpark
  drafter. Faster prefill and decode on Apple Silicon, and the model
  carries its vision encoder. The bridge re-execs itself under the MLX
  venv's python for this backend; the llama backend stays stdlib-only
  under the maya venv.

Inkling writes typed segments (<|content_thinking|>, <|content_text|>,
<|end_message|>). This bridge serves only the content_text segment to
Maya: the thinking happens, it just is not shown, and tool-call segments
are dropped - Maya's own frontend has no Inkling segment parser. Plain
chat through Maya's regular web UI is the job here.

Args (after serve's own --serve): --llama-bin, --gguf, --llama-port (the
child's private port), --ctx, --alias; or --backend mlx, --mlx-model,
--mlx-draft-block, --mlx-python, --max-context.
"""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
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


class SegmentServer:
    """Feeds a stream of generated token ids to serve.  Shared by both
    backends.  Content ids of the thinking and text segments go out,
    plus the end-of-sampling id (200006), the one id serve ends a
    request on for this tokenizer; every other segment marker stays
    here (they would leak into the answer as text, and the message
    end would cut the request at the end of its thinking).  A tool
    call's invoke_tool_json segment is translated into the <tool_call>
    text serve's parser turns into structured calls.  Maya's
    OutputParser opens in the reasoning state, so the thinking ids are
    the request's reasoning (the Monitor's live rates and phase read
    them too), and a literal </think> emitted once when the text
    segment follows thinking is what closes it (the repo frontend's
    InklingRewriter makes the same translation from
    <|content_text|>).
    """

    def __init__(self, tokenize, detokenize=None):
        self.tokenize = tokenize
        self.detokenize = detokenize
        self.state = "pre"
        self.saw_thinking = False
        self.close_sent = False
        self.served = 0
        self.name_buf = []      # plain ids in "other" state (a tool's name precedes its segment)
        self.tool_buf = None    # plain ids inside an invoke_tool_json segment

    def _emit_text(self, text: str) -> None:
        for cid in self.tokenize(text):
            emit(f"T {cid}")
            self.served += 1

    def _close_thinking(self) -> None:
        if self.saw_thinking and not self.close_sent:
            self._emit_text("</think>")
            self.close_sent = True

    def _flush_tool(self) -> None:
        """A finished invoke_tool_json segment becomes the <tool_call> text
        serve's parser builds structured calls from (Qwen's form).  The model
        writes the call as: the tool's name as plain text, the segment open
        (200049), then {"name": ..., "args": {...}} as JSON text."""
        buf, self.tool_buf = self.tool_buf, None
        if buf is None or self.detokenize is None:
            return
        try:
            call = json.loads(self.detokenize(buf).strip())
            name = str(call.get("name") or self.detokenize(self.name_buf).strip())
            args = call.get("args") or {}
            if not name or not isinstance(args, dict):
                return

            def esc(s) -> str:
                return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            parts = [f"<tool_call><function={esc(name)}>"]
            for k, v in args.items():
                val = v if isinstance(v, str) else json.dumps(v)
                parts.append(f"<parameter={esc(k)}>{esc(val)}</parameter>")
            parts.append("</function></tool_call>")
            text = "".join(parts)
        except Exception:  # noqa: BLE001 - an unparseable call is dropped, never mangled into the answer
            return
        self._close_thinking()
        self.state = "text"
        self._emit_text(text)

    def flush(self) -> None:
        self._flush_tool()

    def feed(self, tid: int) -> None:
        if tid == 200008:                       # <|content_thinking|>
            self._flush_tool()
            self.state, self.saw_thinking = "thinking", True
            self.name_buf = []
        elif tid == 200004:                     # <|content_text|>
            self._flush_tool()
            self._close_thinking()
            self.state = "text"
            self.name_buf = []
        elif tid == 200049:                     # <|content_invoke_tool_json|>
            self.tool_buf = []
            self.state = "tool"
        elif tid == 200006:                     # <|content_model_end_sampling|>
            self._flush_tool()
            # Only the end of sampling crosses the pipe: it is the one id
            # that always ends the model's turn, and serve ends the request
            # on it (its stop_ids name it for this tokenizer).  The message
            # end (200010) must NOT cross: a thinking segment is a message
            # of its own, and serve would read its end as the turn's end
            # and cut the request before the answer.  Dropping both made
            # every request finish as "length".
            emit(f"T {tid}")
            self.served += 1
        elif tid >= 200000:                     # any other marker: a segment boundary
            self._flush_tool()
            self.state = "other"
            self.name_buf = []
        elif self.state in ("thinking", "text"):
            emit(f"T {tid}")
            self.served += 1
        elif self.state == "tool":
            self.tool_buf.append(tid)
        elif self.state == "other":
            self.name_buf.append(tid)


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
            seg = SegmentServer(llama.tokenize, llama.detokenize)
            final = {}
            with llama.post("/completion", body) as resp:
                for line in resp:
                    if self.stop.is_set():
                        break
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:])
                    for tid in data.get("tokens", []):
                        seg.feed(tid)
                    if data.get("stop"):
                        final = data
                        break
            seg.flush()
            timings = final.get("timings", {})
            generated = int(final.get("tokens_predicted", 0)) or seg.served
            prompt_tokens = int(timings.get("prompt_n", len(ids)))
            finish = "length" if final.get("stopped_limit") else "stop"
            emit(f"DONE {generated} {prompt_tokens} "
                 f"{timings.get('prompt_ms', 0):.0f} {timings.get('predicted_ms', 0):.0f} "
                 f"{finish}")
        except Exception as e:                              # noqa: BLE001 - reported to serve
            emit(f"ERR ik2 bridge: {e}")


class Mlx:
    """In-process mlx-vlm backend (inkling2-mlx fork): the netprojects
    MLX build of Inkling 2 Mini plus its bundled DSpark drafter."""

    def __init__(self, model_path: str, draft_block: int):
        from mlx_vlm import load
        from mlx_vlm.speculative.drafters import load_drafter
        self.model, self.processor = load(model_path)
        self.draft, self.kind = load_drafter(model_path)
        self.draft_block = draft_block
        tok = getattr(self.processor, "tokenizer", self.processor)
        self.tok = tok

    def tokenize(self, text: str) -> list:
        return self.tok.encode(text)

    def detokenize(self, ids: list) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


def read_pictures(path: str) -> list:
    """serve's GENI file (the MMI1 layout serve/ik2_vision.py writes via mm3_vision): int32 magic,
    int32 count, then per picture int64 start, int32 width, int32 height, height*width*3 RGB bytes.
    -> [(width, height, rgb bytes)]"""
    with open(path, "rb") as f:
        data = f.read()
    magic, count = struct.unpack_from("<4si", data, 0)
    if magic != b"MMI1":
        raise ValueError(f"ik2 bridge: not a picture file ({magic!r})")
    off, pics = 8, []
    for _ in range(count):
        _start, w, h = struct.unpack_from("<qii", data, off)
        off += 16
        n = w * h * 3
        pics.append((w, h, data[off:off + n]))
        off += n
    return pics


def save_picture(w: int, h: int, rgb: bytes) -> str:
    """One picture's RGB bytes -> a temp image file mlx-vlm's processor reads (PNG via Pillow,
    which the MLX venv carries; a stdlib BMP if Pillow is somehow missing)."""
    try:
        from PIL import Image
        fd, name = tempfile.mkstemp(suffix=".png", prefix="ik2-pic-")
        os.close(fd)
        Image.frombytes("RGB", (w, h), rgb).save(name)
        return name
    except ImportError:
        pass
    fd, name = tempfile.mkstemp(suffix=".bmp", prefix="ik2-pic-")
    row = w * 3
    pad = (-row) % 4
    with os.fdopen(fd, "wb") as f:
        f.write(struct.pack("<2sIHHI", b"BM", 54 + (row + pad) * h, 0, 0, 54))
        f.write(struct.pack("<IiiHHIIiiII", 40, w, -h, 1, 24, 0, (row + pad) * h,
                            2835, 2835, 0, 0))
        for y in range(h):
            f.write(rgb[y * row:(y + 1) * row])
            f.write(b"\0" * pad)
    return name


class MlxGeneration:
    def __init__(self, mlx: Mlx):
        self.mlx = mlx
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self, max_new: int, sampling: dict, ids: list) -> None:
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, args=(max_new, sampling, ids),
                                       daemon=True)
        self.thread.start()

    def start_image(self, max_new: int, sampling: dict, path: str, ids: list) -> None:
        self.stop.clear()
        self.thread = threading.Thread(target=self._run_image,
                                       args=(max_new, sampling, path, ids),
                                       daemon=True)
        self.thread.start()

    def _run(self, max_new: int, sampling: dict, ids: list) -> None:
        try:
            self._stream(max_new, sampling, self.mlx.detokenize(ids), len(ids))
        except Exception as e:                              # noqa: BLE001 - reported to serve
            emit(f"ERR ik2 bridge: {e}")

    def _run_image(self, max_new: int, sampling: dict, path: str, ids: list) -> None:
        tmp = []
        try:
            tmp = [save_picture(w, h, rgb) for w, h, rgb in read_pictures(path)]
            # serve expanded the template's one image placeholder into the picture's whole span
            # of image tokens; mlx-vlm's processor expands a single placeholder itself, from the
            # picture's grid - collapse the run back to the templated form.
            prompt = self.mlx.detokenize(ids)
            prompt = re.sub(r"(<\|unused_200054\|>)(?:<\|unused_200054\|>)+", r"\1", prompt)
            self._stream(max_new, sampling, prompt, len(ids), image=tmp)
        except Exception as e:                              # noqa: BLE001 - reported to serve
            emit(f"ERR ik2 bridge: {e}")
        finally:
            for name in tmp:
                try:
                    os.unlink(name)
                except OSError:
                    pass

    def _stream(self, max_new: int, sampling: dict, prompt: str, n_ids: int,
                image: list | None = None) -> None:
        from mlx_vlm import stream_generate
        mlx = self.mlx
        kwargs = {"max_tokens": max_new,
                  "draft_model": mlx.draft, "draft_kind": mlx.kind,
                  "draft_block_size": mlx.draft_block}
        if "temperature" in sampling:
            kwargs["temp"] = sampling["temperature"]
        if "top_p" in sampling:
            kwargs["top_p"] = sampling["top_p"]
        if "top_k" in sampling:
            kwargs["top_k"] = sampling["top_k"]
        if image:
            kwargs["image"] = image
        seg = SegmentServer(mlx.tokenize, mlx.detokenize)
        last = None
        t0 = time.time()
        stream = stream_generate(mlx.model, mlx.processor, prompt, **kwargs)
        try:
            for r in stream:
                if self.stop.is_set():
                    break
                if r.token is not None:
                    seg.feed(r.token)
                last = r
        finally:
            stream.close()
        seg.flush()
        wall_ms = (time.time() - t0) * 1000
        if last is None:
            emit("ERR ik2 bridge: mlx produced no output")
            return
        generated = int(last.generation_tokens) or seg.served
        prompt_tokens = int(last.prompt_tokens) or n_ids
        prompt_ms = (prompt_tokens / last.prompt_tps * 1000
                     if last.prompt_tps else 0)
        decode_ms = (last.generation_tokens / last.generation_tps * 1000
                     if last.generation_tps else wall_ms)
        finish = "length" if last.finish_reason == "length" else "stop"
        emit(f"DONE {generated} {prompt_tokens} "
             f"{prompt_ms:.0f} {decode_ms:.0f} {finish}")


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
    if opts.get("backend") == "mlx":
        return run_mlx(opts, ctx)
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
    emit("INFO engine=ik2-bridge-1.0"
         + (" dspark=1" if opts.get("llama-spec-type") == "draft-dspark" else ""))
    emit(f"READY {ctx} stop")
    gen = Generation(llama)
    try:
        serve_loop(gen)
    finally:
        gen.stop.set()
        llama.kill()
    return 0


def run_mlx(opts: dict, ctx: int) -> int:
    # mlx-vlm lives in its own venv; re-exec under its python once.
    venv_py = opts.get("mlx-python", "/Users/duckets/mlx-venv/bin/python")
    if (not os.environ.get("IK2_BRIDGE_MLX")
            and os.path.abspath(sys.executable) != os.path.abspath(venv_py)):
        os.environ["IK2_BRIDGE_MLX"] = "1"
        os.execv(venv_py, [venv_py, os.path.abspath(__file__)] + sys.argv[1:])
    try:
        mlx = Mlx(opts["mlx-model"], int(opts.get("mlx-draft-block", "4")))
    except Exception as e:                                  # noqa: BLE001 - serve reads READY or dies
        print(f"ik2 bridge: {e}", file=sys.stderr, flush=True)
        return 1
    emit("NOTE ik2 bridge: Inkling 2 Mini via MLX (DSpark)")
    if opts.get("mlx-vision") == "off":
        emit("INFO engine=ik2-bridge-1.0 vision=off dspark=1")
    else:
        # the MLX build carries the model's vision tower: serve resizes the pictures and sends
        # their pixels with GENI (serve/ik2_vision.py); the bridge runs the tower in-process.
        emit("INFO engine=ik2-bridge-1.0 vision=ik2 dspark=1 vision_patch=20 vision_merge=2 "
             "image_token=200054 vision_min_pixels=1600 vision_max_pixels=4194304")
    emit(f"READY {ctx} stop")
    gen = MlxGeneration(mlx)
    try:
        serve_loop(gen)
    finally:
        gen.stop.set()
    return 0


def _sampling_of(parts: list) -> dict:
    sampling = {}
    for kv in parts:
        k, _, v = kv.partition("=")
        try:
            sampling[k] = float(v) if "." in v else int(v)
        except ValueError:
            pass
    return sampling


def serve_loop(gen) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.startswith("GENI "):
            # an image request: GENI <max_new> [key=value ...] <pictures file> <token ids>.
            # Only a backend with start_image (the MLX one) can take it - serve sends the
            # resized pixels in the file; the llama backend fails it cleanly instead of hanging.
            if gen.thread and gen.thread.is_alive():
                gen.thread.join()
            if not hasattr(gen, "start_image"):
                emit("ERR ik2 bridge: images are not supported through the bridge")
                continue
            parts = line.split(" ")
            ids = [int(t) for t in parts[-1].split(",") if t]
            gen.start_image(int(parts[1]), _sampling_of(parts[2:-2]), parts[-2], ids)
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


if __name__ == "__main__":
    sys.exit(main())
