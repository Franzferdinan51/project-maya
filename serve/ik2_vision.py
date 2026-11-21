"""serve/ik2_vision.py - pictures for Inkling 2 Mini on the Mac (the ik2 bridge's MLX backend runs the
vision tower itself).

The server's half, shaped like serve.mm3_vision (this Vision subclasses it, so the server's ENGINE_VISION
isinstance checks include it): a picture is resized (aspect kept, bicubic) to sides that are multiples of
40 (Inkling's 20px native patch, merged 2x2) with at most 2048 x 2048 pixels, and the one
<|unused_200054|> the template writes per picture becomes the span of its image tokens. The engine gets
the resized pixels in a GENI file (the same MMI1 layout as MiniMax's: int32 magic, int32 count, then per
picture int64 start, int32 width, int32 height, height x width x 3 RGB bytes); the bridge runs the tower
(the MLX build carries the model's own vision encoder). EXIF orientation is applied (the picture as
viewers see it) and transparent areas become white, as for the server's other models.
"""
from __future__ import annotations

from pathlib import Path

from serve import mm3_vision
from serve.mm3_vision import prepare  # noqa: F401  (re-exported for readers of this module)


class Picture(mm3_vision.Picture):
    @property
    def tokens(self) -> int:
        """The span's tokens: Inkling's span is its image tokens alone (no start/end tokens around
        them - the template's own markers stand outside the placeholder)."""
        return self.n


class Geometry(mm3_vision.Geometry):
    """Inkling 2's grid: a 20px native patch merged 2x2 (a 40px factor), its image token, and the
    picture bounds this server offers. The engine's `INFO vision=ik2 vision_patch=... ...` facts win
    over these defaults."""

    @classmethod
    def from_info(cls, info: dict) -> "Geometry":
        def num(k, d):
            v = info.get(k)
            return int(v) if isinstance(v, int) else d
        return cls(patch=num("vision_patch", 20), merge=num("vision_merge", 2),
                   min_pixels=num("vision_min_pixels", 40 * 40),
                   max_pixels=num("vision_max_pixels", 2048 * 2048),
                   image_token=num("image_token", 200054),
                   image_start=num("image_start", 200054), image_end=num("image_end", 200054))


class Vision(mm3_vision.Vision):
    """The server's side of the ik2 bridge's pictures: mm3's resize and GENI file, with Inkling's span
    shape - the template's picture block already carries its own markers (<|content_image|> before the
    placeholder, <|end_message|> after), so the placeholder alone becomes the picture's image tokens."""

    def picture(self, source: str) -> Picture:
        pic = super().picture(source)
        return Picture(pic.rgb, pic.width, pic.height, pic.n)

    def expand(self, ids: list[int], pictures: list[Picture]) -> tuple[list[int], list[int]]:
        """Each <|unused_200054|> of the rendered prompt (one per picture, in order) -> its picture's
        image tokens; -> (ids, the positions of the spans' first image tokens)."""
        g = self.g
        out, starts, k = [], [], 0
        for t in ids:
            if t == g.image_token:
                if k >= len(pictures):
                    raise ValueError("the prompt and its images do not match")
                starts.append(len(out))
                out += [g.image_token] * pictures[k].n
                k += 1
            else:
                out.append(t)
        if k != len(pictures):
            raise ValueError("the prompt and its images do not match")
        return out, starts
