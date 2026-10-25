# Copyright 2026 Jigyasu Juneja
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-Stdlib PNG/PPM Codec, Dual-Buffer Photometry, CIE Lab Similarity, and Auto-Saliency Contact Sheets.

Implements the dual-buffer visual recognition pipeline with zero required external dependencies:
  - Binary PNG (signature, IHDR, IDAT scanline filters 0..4, RGB/RGBA/Gray, IEND, IEEE CRC32) and PPM (P6/P3)
    codecs in pure Python stdlib (`zlib`, `struct`), verifying chunk framing and CRC32 before any optional
    Pillow fast-path.
  - Stage-2A unsharpened Shades-of-Gray Minkowski p=6 white balance (`shades_of_gray_white_balance`),
    inner-80% CIE L*a*b* Delta-E similarity (`inner_crop_cie_lab_similarity`), and residual InfoNCE MLP
    adapter `norm(q + alpha * W2 * GELU(W1 * q))` (`apply_residual_mlp_adapter`).
  - Stage-2B composite contact-sheet compiler (`compile_dual_buffer_contact_sheet`) keeping `stage2a_query_buffer`
    strictly unsharpened while building local-contrast-adjusted comparison columns with `2x` zoomed inter-candidate
    variance saliency strips (`extract_inter_candidate_saliency_band`), `2x` bottom-30% pack-weight zoom strips,
    and optional `1.15x` cylindrical horizontal unwarping (`curved_container=True`).
"""

from __future__ import annotations

import io
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import SchemaContractViolationError
from .shortlist import split_identifier_tokens

try:
    from PIL import Image as _PILImage  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - pure stdlib is primary
    _PILImage = None

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SRGB_TO_LIN_LUT: tuple[float, ...] = tuple(
    (c / 255.0) / 12.92 if (c / 255.0) <= 0.04045 else (((c / 255.0) + 0.055) / 1.055) ** 2.4
    for c in range(256)
)

_FONT_5X7: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111),
    "3": (0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
}


@dataclass(frozen=True)
class ImageBuffer:
    """Immutable 8-bit packed RGB image buffer (`len(rgb) == width * height * 3`)."""

    width: int
    height: int
    rgb: bytes

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(f"ImageBuffer dimensions must be >= 1x1, got {self.width}x{self.height}")
        if not isinstance(self.rgb, (bytes, bytearray)):
            raise ValueError("ImageBuffer.rgb must be bytes")
        if isinstance(self.rgb, bytearray):
            object.__setattr__(self, "rgb", bytes(self.rgb))
        expected = self.width * self.height * 3
        if len(self.rgb) != expected:
            raise ValueError(f"ImageBuffer expected {expected} RGB bytes, got {len(self.rgb)}")

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise IndexError(f"pixel ({x}, {y}) outside {self.width}x{self.height}")
        off = (y * self.width + x) * 3
        return self.rgb[off], self.rgb[off + 1], self.rgb[off + 2]

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> ImageBuffer:
        cx0 = max(0, min(self.width - 1, x0))
        cy0 = max(0, min(self.height - 1, y0))
        cx1 = max(cx0 + 1, min(self.width, x1))
        cy1 = max(cy0 + 1, min(self.height, y1))
        cw = cx1 - cx0
        ch = cy1 - cy0
        out = bytearray(cw * ch * 3)
        row_bytes = cw * 3
        for dy in range(ch):
            src_off = ((cy0 + dy) * self.width + cx0) * 3
            dst_off = dy * row_bytes
            out[dst_off : dst_off + row_bytes] = self.rgb[src_off : src_off + row_bytes]
        return ImageBuffer(cw, ch, bytes(out))

    def resize_bilinear(self, new_width: int, new_height: int) -> ImageBuffer:
        if new_width < 1 or new_height < 1:
            raise ValueError(f"resize target must be >= 1x1, got {new_width}x{new_height}")
        if new_width == self.width and new_height == self.height:
            return ImageBuffer(self.width, self.height, bytes(self.rgb))

        out = bytearray(new_width * new_height * 3)
        src = self.rgb
        sw = self.width
        sh = self.height
        x_scale = float(sw - 1) / float(max(1, new_width - 1)) if new_width > 1 else 0.0
        y_scale = float(sh - 1) / float(max(1, new_height - 1)) if new_height > 1 else 0.0

        dst_idx = 0
        for ny in range(new_height):
            fy = ny * y_scale if new_height > 1 else 0.5 * float(sh - 1)
            y0 = int(fy)
            y1 = min(sh - 1, y0 + 1)
            wy = fy - y0
            w_y0 = 1.0 - wy
            row0 = y0 * sw * 3
            row1 = y1 * sw * 3
            for nx in range(new_width):
                fx = nx * x_scale if new_width > 1 else 0.5 * float(sw - 1)
                x0 = int(fx)
                x1 = min(sw - 1, x0 + 1)
                wx = fx - x0
                w00 = w_y0 * (1.0 - wx)
                w01 = w_y0 * wx
                w10 = wy * (1.0 - wx)
                w11 = wy * wx
                i00 = row0 + x0 * 3
                i01 = row0 + x1 * 3
                i10 = row1 + x0 * 3
                i11 = row1 + x1 * 3
                for c in range(3):
                    val = (
                        w00 * src[i00 + c]
                        + w01 * src[i01 + c]
                        + w10 * src[i10 + c]
                        + w11 * src[i11 + c]
                    )
                    out[dst_idx] = min(255, max(0, int(round(val))))
                    dst_idx += 1
        return ImageBuffer(new_width, new_height, bytes(out))


@dataclass(frozen=True)
class CompiledContactSheet:
    """Output of dual-buffer Stage-2A / Stage-2B contact-sheet compilation."""

    png_bytes: bytes
    stage2a_query_buffer: ImageBuffer
    saliency_band: tuple[int, int]
    variance_profile: list[float]
    diff_hints: list[str]
    panel_labels: list[str]


def _png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    return struct.pack(">I", len(chunk_data)) + chunk_type + chunk_data + struct.pack(">I", crc)


def encode_png(buf: ImageBuffer) -> bytes:
    """Encode an `ImageBuffer` as valid 8-bit RGB PNG bytes (`IHDR`, `IDAT`, `IEND`)."""
    ihdr = struct.pack(">IIBBBBB", buf.width, buf.height, 8, 2, 0, 0, 0)
    row_stride = buf.width * 3
    raw_scanlines = bytearray(buf.height * (1 + row_stride))
    for y in range(buf.height):
        dst_off = y * (1 + row_stride)
        src_off = y * row_stride
        # Use Sub filter (1) on interior rows for lossless compression and None (0) on row 0.
        if y == 0:
            raw_scanlines[dst_off] = 0
            raw_scanlines[dst_off + 1 : dst_off + 1 + row_stride] = buf.rgb[src_off : src_off + row_stride]
        else:
            raw_scanlines[dst_off] = 1
            row_bytes = buf.rgb[src_off : src_off + row_stride]
            for i in range(row_stride):
                left = row_bytes[i - 3] if i >= 3 else 0
                raw_scanlines[dst_off + 1 + i] = (row_bytes[i] - left) & 0xFF

    compressed = zlib.compress(bytes(raw_scanlines), level=6)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png(data: bytes) -> ImageBuffer:
    """Decode PNG bytes (`color_type` 0, 2, or 6; filters 0..4), strictly verifying IEEE CRC32 on every chunk."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 33:
        raise ValueError("invalid or truncated PNG byte stream")
    raw = bytes(data)
    if raw[:8] != _PNG_SIGNATURE:
        raise ValueError("invalid PNG signature")

    pos = 8
    seen_ihdr = False
    seen_iend = False
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    interlace = 0
    idat_parts: list[bytes] = []

    while pos < len(raw):
        if pos + 8 > len(raw):
            raise ValueError("truncated PNG chunk header")
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        ctype = raw[pos + 4 : pos + 8]
        pos += 8
        if pos + length + 4 > len(raw):
            raise ValueError(f"truncated PNG chunk payload for {ctype!r}")
        cdata = raw[pos : pos + length]
        expected_crc = struct.unpack(">I", raw[pos + length : pos + length + 4])[0]
        pos += length + 4

        actual_crc = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(
                f"PNG CRC32 mismatch on chunk {ctype!r}: expected 0x{expected_crc:08x}, got 0x{actual_crc:08x}"
            )

        if not seen_ihdr:
            if ctype != b"IHDR" or len(cdata) != 13:
                raise ValueError("PNG must begin with a 13-byte IHDR chunk")
            width, height, bit_depth, color_type, comp_m, filt_m, interlace = struct.unpack(
                ">IIBBBBB", cdata
            )
            if width < 1 or height < 1:
                raise ValueError(f"invalid PNG dimensions {width}x{height}")
            if comp_m != 0 or filt_m != 0:
                raise ValueError("unsupported PNG compression or filter method")
            seen_ihdr = True
        elif ctype == b"IHDR":
            raise ValueError("duplicate IHDR chunk in PNG stream")
        elif ctype == b"IDAT":
            idat_parts.append(cdata)
        elif ctype == b"IEND":
            if len(cdata) != 0:
                raise ValueError(f"IEND chunk must have 0 payload bytes, got {len(cdata)}")
            seen_iend = True
            break

    if not seen_ihdr or not seen_iend or not idat_parts:
        raise ValueError("incomplete PNG stream: missing IHDR, IDAT, or IEND")

    if bit_depth == 8 and interlace == 0 and color_type in (0, 2, 6):
        bpp = {0: 1, 2: 3, 6: 4}[color_type]
        try:
            decompressed = zlib.decompress(b"".join(idat_parts))
        except zlib.error as exc:
            raise ValueError(f"corrupt PNG IDAT deflate stream: {exc}") from exc

        row_bytes = width * bpp
        expected_len = height * (1 + row_bytes)
        if len(decompressed) != expected_len:
            raise ValueError(
                f"PNG decompressed byte count mismatch: expected {expected_len}, got {len(decompressed)}"
            )

        prev_row = bytearray(row_bytes)
        rgb_out = bytearray(width * height * 3)
        out_idx = 0

        for y in range(height):
            off = y * (1 + row_bytes)
            ftype = decompressed[off]
            scan = decompressed[off + 1 : off + 1 + row_bytes]
            recon = bytearray(row_bytes)

            if ftype == 0:
                recon[:] = scan
            elif ftype == 1:
                for i in range(row_bytes):
                    a = recon[i - bpp] if i >= bpp else 0
                    recon[i] = (scan[i] + a) & 0xFF
            elif ftype == 2:
                for i in range(row_bytes):
                    recon[i] = (scan[i] + prev_row[i]) & 0xFF
            elif ftype == 3:
                for i in range(row_bytes):
                    a = recon[i - bpp] if i >= bpp else 0
                    b = prev_row[i]
                    recon[i] = (scan[i] + ((a + b) >> 1)) & 0xFF
            elif ftype == 4:
                for i in range(row_bytes):
                    a = recon[i - bpp] if i >= bpp else 0
                    b = prev_row[i]
                    c = prev_row[i - bpp] if i >= bpp else 0
                    recon[i] = (scan[i] + _paeth_predictor(a, b, c)) & 0xFF
            else:
                raise ValueError(f"unsupported PNG scanline filter type {ftype} at row {y}")

            if color_type == 2:
                rgb_out[out_idx : out_idx + width * 3] = recon
                out_idx += width * 3
            elif color_type == 6:
                for px in range(width):
                    r_v = recon[px * 4]
                    g_v = recon[px * 4 + 1]
                    b_v = recon[px * 4 + 2]
                    a_v = recon[px * 4 + 3]
                    if a_v == 255:
                        rgb_out[out_idx] = r_v
                        rgb_out[out_idx + 1] = g_v
                        rgb_out[out_idx + 2] = b_v
                    else:
                        inv_a = 255 - a_v
                        rgb_out[out_idx] = (r_v * a_v + 255 * inv_a + 127) // 255
                        rgb_out[out_idx + 1] = (g_v * a_v + 255 * inv_a + 127) // 255
                        rgb_out[out_idx + 2] = (b_v * a_v + 255 * inv_a + 127) // 255
                    out_idx += 3
            else:
                for px in range(width):
                    g_v = recon[px]
                    rgb_out[out_idx] = g_v
                    rgb_out[out_idx + 1] = g_v
                    rgb_out[out_idx + 2] = g_v
                    out_idx += 3

            prev_row = recon

        return ImageBuffer(width, height, bytes(rgb_out))

    if _PILImage is not None:
        with _PILImage.open(io.BytesIO(raw)) as pil_img:
            conv = pil_img.convert("RGB")
            return ImageBuffer(conv.width, conv.height, conv.tobytes())

    raise ValueError(
        f"unsupported PNG format (bit_depth={bit_depth}, color_type={color_type}, interlace={interlace})"
    )


def encode_ppm(buf: ImageBuffer) -> bytes:
    """Encode an `ImageBuffer` as binary P6 PPM bytes."""
    header = f"P6\n{buf.width} {buf.height}\n255\n".encode("ascii")
    return header + buf.rgb


def _read_ppm_tokens(raw: bytes, count: int, start: int = 0) -> tuple[list[bytes], int]:
    tokens: list[bytes] = []
    n = len(raw)
    i = start
    while i < n and len(tokens) < count:
        b_ch = raw[i]
        if b_ch in (9, 10, 13, 32):
            i += 1
            continue
        if b_ch == 35:  # '#'
            while i < n and raw[i] != 10:
                i += 1
            continue
        t0 = i
        while i < n and raw[i] not in (9, 10, 13, 32, 35):
            i += 1
        tokens.append(raw[t0:i])
    return tokens, i


def decode_ppm(data: bytes) -> ImageBuffer:
    """Decode binary (`P6`) or ASCII (`P3`) PPM bytes into an `ImageBuffer`."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 7:
        raise ValueError("invalid or truncated PPM byte stream")
    raw = bytes(data)
    tokens, pos = _read_ppm_tokens(raw, 4, 0)
    if len(tokens) < 4:
        raise ValueError("incomplete PPM header")
    magic = tokens[0]
    if magic not in (b"P6", b"P3"):
        raise ValueError(f"unsupported PPM magic {magic!r}; expected b'P6' or b'P3'")
    try:
        width = int(tokens[1])
        height = int(tokens[2])
        maxval = int(tokens[3])
    except ValueError as exc:
        raise ValueError(f"invalid PPM header integers: {exc}") from exc

    if width < 1 or height < 1 or maxval < 1 or maxval > 65535:
        raise ValueError(f"invalid PPM header geometry {width}x{height} maxval={maxval}")

    expected_samples = width * height * 3
    if magic == b"P6":
        if pos >= len(raw) or raw[pos] not in (9, 10, 13, 32):
            raise ValueError("PPM P6 header missing single whitespace separator")
        pos += 1
        if maxval == 255:
            payload = raw[pos : pos + expected_samples]
            if len(payload) != expected_samples:
                raise ValueError(f"truncated P6 payload: expected {expected_samples}, got {len(payload)}")
            return ImageBuffer(width, height, payload)
        if maxval < 256:
            payload = raw[pos : pos + expected_samples]
            if len(payload) != expected_samples:
                raise ValueError("truncated P6 payload")
            scaled = bytes(min(255, int(round((b * 255.0) / float(maxval)))) for b in payload)
            return ImageBuffer(width, height, scaled)
        byte_need = expected_samples * 2
        payload16 = raw[pos : pos + byte_need]
        if len(payload16) != byte_need:
            raise ValueError("truncated 16-bit P6 payload")
        out16 = bytearray(expected_samples)
        for idx in range(expected_samples):
            val16 = (payload16[2 * idx] << 8) | payload16[2 * idx + 1]
            out16[idx] = min(255, int(round((val16 * 255.0) / float(maxval))))
        return ImageBuffer(width, height, bytes(out16))

    sample_tokens, _ = _read_ppm_tokens(raw, expected_samples, pos)
    if len(sample_tokens) != expected_samples:
        raise ValueError(f"truncated P3 payload: expected {expected_samples} samples, got {len(sample_tokens)}")
    out_p3 = bytearray(expected_samples)
    for idx, tok in enumerate(sample_tokens):
        val = int(tok)
        if val < 0 or val > maxval:
            raise ValueError(f"PPM P3 sample {val} out of [0, {maxval}]")
        out_p3[idx] = val if maxval == 255 else min(255, int(round((val * 255.0) / float(maxval))))
    return ImageBuffer(width, height, bytes(out_p3))


def _lin_to_srgb_byte(val: float) -> int:
    clamped = min(1.0, max(0.0, val))
    if clamped <= 0.0031308:
        srgb = 12.92 * clamped
    else:
        srgb = 1.055 * (clamped ** (1.0 / 2.4)) - 0.055
    return min(255, max(0, int(round(srgb * 255.0))))


def shades_of_gray_white_balance(buf: ImageBuffer, p: float = 6.0) -> ImageBuffer:
    """Apply Stage-2A unsharpened Minkowski p-norm Shades-of-Gray illuminant normalization."""
    if not math.isfinite(p) or p < 1.0:
        raise ValueError(f"Minkowski exponent p must be finite and >= 1.0, got {p!r}")

    lut = _SRGB_TO_LIN_LUT
    pow_lut = [v**p for v in lut]
    rgb = buf.rgb
    n_px = float(buf.width * buf.height)

    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    for off in range(0, len(rgb), 3):
        sum_r += pow_lut[rgb[off]]
        sum_g += pow_lut[rgb[off + 1]]
        sum_b += pow_lut[rgb[off + 2]]

    inv_p = 1.0 / p
    mu_r = (sum_r / n_px) ** inv_p
    mu_g = (sum_g / n_px) ** inv_p
    mu_b = (sum_b / n_px) ** inv_p
    if max(mu_r, mu_g, mu_b) <= 1e-8:
        return ImageBuffer(buf.width, buf.height, bytes(rgb))

    mu_bar = (mu_r + mu_g + mu_b) / 3.0
    gain_r = mu_bar / max(mu_r, 1e-6)
    gain_g = mu_bar / max(mu_g, 1e-6)
    gain_b = mu_bar / max(mu_b, 1e-6)

    map_r = [_lin_to_srgb_byte(lut[c] * gain_r) for c in range(256)]
    map_g = [_lin_to_srgb_byte(lut[c] * gain_g) for c in range(256)]
    map_b = [_lin_to_srgb_byte(lut[c] * gain_b) for c in range(256)]

    out = bytearray(len(rgb))
    for off in range(0, len(rgb), 3):
        out[off] = map_r[rgb[off]]
        out[off + 1] = map_g[rgb[off + 1]]
        out[off + 2] = map_b[rgb[off + 2]]
    return ImageBuffer(buf.width, buf.height, bytes(out))


def _lab_f(t: float) -> float:
    delta = 6.0 / 29.0
    if t > delta**3:
        return t ** (1.0 / 3.0)
    return (t / (3.0 * delta * delta)) + (4.0 / 29.0)


def mean_inner_crop_cie_lab(buf: ImageBuffer, crop_fraction: float = 0.80) -> tuple[float, float, float]:
    """Compute mean D65 CIE L*a*b* coordinates over the inner `crop_fraction` of an `ImageBuffer`."""
    if not math.isfinite(crop_fraction) or crop_fraction <= 0.0 or crop_fraction > 1.0:
        raise ValueError(f"crop_fraction must lie in (0, 1], got {crop_fraction!r}")

    margin_x = int(round((1.0 - crop_fraction) * 0.5 * buf.width))
    margin_y = int(round((1.0 - crop_fraction) * 0.5 * buf.height))
    x0 = max(0, min(buf.width - 1, margin_x))
    y0 = max(0, min(buf.height - 1, margin_y))
    x1 = max(x0 + 1, buf.width - margin_x)
    y1 = max(y0 + 1, buf.height - margin_y)
    inner = buf.crop(x0, y0, x1, y1)

    lut = _SRGB_TO_LIN_LUT
    rgb = inner.rgb
    n_px = float(inner.width * inner.height)
    sum_l = 0.0
    sum_a = 0.0
    sum_b = 0.0

    for off in range(0, len(rgb), 3):
        r_lin = lut[rgb[off]]
        g_lin = lut[rgb[off + 1]]
        b_lin = lut[rgb[off + 2]]

        x_val = (0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin) / 0.95047
        y_val = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
        z_val = (0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin) / 1.08883

        fx = _lab_f(x_val)
        fy = _lab_f(y_val)
        fz = _lab_f(z_val)

        sum_l += 116.0 * fy - 16.0
        sum_a += 500.0 * (fx - fy)
        sum_b += 200.0 * (fy - fz)

    return sum_l / n_px, sum_a / n_px, sum_b / n_px


def _is_non_str_seq(val: Any) -> bool:
    return isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray))


def _require_finite_num(val: Any, label: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise SchemaContractViolationError(
            f"{label} must be a finite int or float, got {type(val).__name__}"
        )
    out = float(val)
    if not math.isfinite(out):
        raise SchemaContractViolationError(f"{label} must be finite, got {val!r}")
    return out


def inner_crop_cie_lab_similarity(
    buf_a: ImageBuffer,
    buf_b: ImageBuffer,
    crop_fraction: float = 0.80,
    scale: float = 25.0,
) -> tuple[float, float]:
    """Compute `(exp(-delta_e / scale), delta_e)` between inner-80% CIE L*a*b* means of two buffers."""
    f_scale = _require_finite_num(scale, "scale")
    if f_scale <= 0.0:
        raise SchemaContractViolationError(f"scale must be finite and > 0, got {scale!r}")
    la, aa, ba = mean_inner_crop_cie_lab(buf_a, crop_fraction=crop_fraction)
    lb, ab, bb = mean_inner_crop_cie_lab(buf_b, crop_fraction=crop_fraction)
    delta_e = math.hypot(la - lb, aa - ab, ba - bb)
    return math.exp(-delta_e / f_scale), delta_e


def _gelu_exact(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def apply_residual_mlp_adapter(
    query_vec: Sequence[float],
    w1: Sequence[Sequence[float]],
    w2: Sequence[Sequence[float]],
    alpha: float = 0.35,
) -> list[float]:
    """Project embedding via `norm(q + alpha * W2 * GELU(W1 * q))` to separate sibling variants."""
    if not _is_non_str_seq(query_vec) or not query_vec:
        raise SchemaContractViolationError("query_vec must be a non-empty numeric sequence")
    f_alpha = _require_finite_num(alpha, "alpha")

    q = [_require_finite_num(v, "query_vec") for v in query_vec]
    dim = len(q)

    if not _is_non_str_seq(w1) or not w1 or any(not _is_non_str_seq(row) or len(row) != dim for row in w1):
        raise SchemaContractViolationError(f"w1 must be a non-empty (H x {dim}) numeric matrix")
    hidden_dim = len(w1)
    if not _is_non_str_seq(w2) or len(w2) != dim or any(not _is_non_str_seq(row) or len(row) != hidden_dim for row in w2):
        raise SchemaContractViolationError(f"w2 must be a ({dim} x {hidden_dim}) numeric matrix")

    w1_clean = [[_require_finite_num(w_val, "w1") for w_val in row] for row in w1]
    w2_clean = [[_require_finite_num(w_val, "w2") for w_val in row2] for row2 in w2]

    if f_alpha == 0.0:
        combined = list(q)
    else:
        hidden: list[float] = []
        for row_f in w1_clean:
            acc = sum(fw * q_val for fw, q_val in zip(row_f, q))
            if not math.isfinite(acc):
                raise SchemaContractViolationError("residual MLP w1 projection overflowed float64")
            hidden.append(_gelu_exact(acc))

        combined = []
        for i, row2_f in enumerate(w2_clean):
            proj = sum(fw * h_val for fw, h_val in zip(row2_f, hidden))
            val = q[i] + f_alpha * proj
            if not math.isfinite(val):
                raise SchemaContractViolationError("residual MLP combined vector overflowed float64")
            combined.append(val)

    max_abs = max(abs(v) for v in combined)
    if max_abs <= 0.0:
        return [0.0] * dim
    scaled = [v / max_abs for v in combined]
    unit_hypot = math.hypot(*scaled)
    return [s / unit_hypot for s in scaled]


def extract_inter_candidate_saliency_band(
    candidates: Sequence[ImageBuffer],
    band_height_fraction: float | None = None,
) -> tuple[int, int, list[float]]:
    """Compute per-row inter-candidate RGB variance V(y) and return `(y_start, y_end, variance_profile)`."""
    if not candidates:
        raise ValueError("candidates must contain at least one ImageBuffer")
    if band_height_fraction is not None and (
        not math.isfinite(band_height_fraction) or band_height_fraction <= 0.0 or band_height_fraction > 1.0
    ):
        raise ValueError(f"band_height_fraction must lie in (0, 1], got {band_height_fraction!r}")

    target_w = candidates[0].width
    target_h = candidates[0].height
    aligned = [
        c if (c.width == target_w and c.height == target_h) else c.resize_bilinear(target_w, target_h)
        for c in candidates
    ]

    k_cnt = len(aligned)
    row_stride = target_w * 3
    variances: list[float] = []

    if k_cnt == 1:
        rgb0 = aligned[0].rgb
        for y in range(target_h):
            off = y * row_stride
            var_acc = 0.0
            for c in range(3):
                vals = [float(rgb0[off + x * 3 + c]) for x in range(target_w)]
                mean_v = sum(vals) / float(target_w)
                var_acc += sum((v - mean_v) ** 2 for v in vals) / float(target_w)
            variances.append(var_acc / 3.0)
    else:
        inv_k = 1.0 / float(k_cnt)
        inv_3w = 1.0 / float(row_stride)
        for y in range(target_h):
            off = y * row_stride
            row_var_sum = 0.0
            for byte_x in range(row_stride):
                idx = off + byte_x
                s1 = 0.0
                s2 = 0.0
                for buf in aligned:
                    val = float(buf.rgb[idx])
                    s1 += val
                    s2 += val * val
                mean_v = s1 * inv_k
                row_var_sum += max(0.0, (s2 * inv_k) - (mean_v * mean_v))
            variances.append(row_var_sum * inv_3w)

    if band_height_fraction is not None:
        win_h = max(1, min(target_h, int(round(target_h * band_height_fraction))))
        best_y = 0
        cur_sum = sum(variances[:win_h])
        best_sum = cur_sum
        for y in range(1, target_h - win_h + 1):
            cur_sum += variances[y + win_h - 1] - variances[y - 1]
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_y = y
        return best_y, best_y + win_h, variances

    v_max = max(variances)
    v_min = min(variances)
    if (v_max - v_min) <= 1e-6:
        y0_def = int(0.35 * target_h)
        y1_def = max(y0_def + 1, int(0.65 * target_h))
        return y0_def, min(target_h, y1_def), variances

    sorted_v = sorted(variances)
    v_base = sorted_v[int(0.05 * (len(sorted_v) - 1))]
    thresh = v_base + 0.03 * (v_max - v_base)
    bridge_thresh = v_base + 0.01 * (v_max - v_base)

    active = [v >= thresh for v in variances]
    for y in range(1, target_h - 1):
        if not active[y] and active[y - 1] and active[y + 1] and variances[y] >= bridge_thresh:
            active[y] = True

    best_start = 0
    best_end = max(1, target_h)
    best_excess = -1.0
    y_idx = 0
    while y_idx < target_h:
        if not active[y_idx]:
            y_idx += 1
            continue
        run_start = y_idx
        excess = 0.0
        while y_idx < target_h and active[y_idx]:
            excess += max(0.0, variances[y_idx] - thresh)
            y_idx += 1
        run_end = y_idx
        if excess > best_excess:
            best_excess = excess
            best_start = run_start
            best_end = run_end

    return best_start, max(best_start + 1, best_end), variances


def _apply_stage2b_local_contrast(buf: ImageBuffer, amount: float = 0.65) -> ImageBuffer:
    """Apply Stage-2B 3x3 neighborhood local contrast stretching (strictly on cloned Stage-2B copies)."""
    w = buf.width
    h = buf.height
    src = buf.rgb
    out = bytearray(len(src))
    for y in range(h):
        ym1 = max(0, y - 1)
        yp1 = min(h - 1, y + 1)
        for x in range(w):
            xm1 = max(0, x - 1)
            xp1 = min(w - 1, x + 1)
            off_c = (y * w + x) * 3
            for c in range(3):
                neigh_sum = (
                    src[(ym1 * w + xm1) * 3 + c]
                    + src[(ym1 * w + x) * 3 + c]
                    + src[(ym1 * w + xp1) * 3 + c]
                    + src[(y * w + xm1) * 3 + c]
                    + src[off_c + c]
                    + src[(y * w + xp1) * 3 + c]
                    + src[(yp1 * w + xm1) * 3 + c]
                    + src[(yp1 * w + x) * 3 + c]
                    + src[(yp1 * w + xp1) * 3 + c]
                )
                mean_v = neigh_sum / 9.0
                center_v = float(src[off_c + c])
                boosted = center_v + amount * (center_v - mean_v)
                out[off_c + c] = min(255, max(0, int(round(boosted))))
    return ImageBuffer(w, h, bytes(out))


def _cylindrical_horizontal_unwarp(buf: ImageBuffer, target_width: int) -> ImageBuffer:
    """Unroll curved cylindrical packaging across `target_width` (~1.15x) via inverse-sine mapping."""
    sw = buf.width
    sh = buf.height
    out = bytearray(target_width * sh * 3)
    src = buf.rgb
    theta_max = math.pi / 3.0
    sin_max = math.sin(theta_max)

    x_coords: list[float] = []
    for nx in range(target_width):
        frac = float(nx) / float(max(1, target_width - 1)) if target_width > 1 else 0.5
        theta = -theta_max + (2.0 * theta_max * frac)
        x_src = 0.5 * float(sw - 1) * ((math.sin(theta) / sin_max) + 1.0)
        x_coords.append(min(float(sw - 1), max(0.0, x_src)))

    for y in range(sh):
        s_row = y * sw * 3
        d_row = y * target_width * 3
        for nx, fx in enumerate(x_coords):
            x0 = int(fx)
            x1 = min(sw - 1, x0 + 1)
            wx = fx - x0
            i0 = s_row + x0 * 3
            i1 = s_row + x1 * 3
            d_off = d_row + nx * 3
            for c in range(3):
                val = (1.0 - wx) * src[i0 + c] + wx * src[i1 + c]
                out[d_off + c] = min(255, max(0, int(round(val))))
    return ImageBuffer(target_width, sh, bytes(out))


def _render_label_header(width: int, height: int, label: str) -> ImageBuffer:
    canvas = bytearray(width * height * 3)
    for off in range(0, len(canvas), 3):
        canvas[off] = 20
        canvas[off + 1] = 30
        canvas[off + 2] = 52

    digits = [ch for ch in label if ch in _FONT_5X7]
    if not digits:
        return ImageBuffer(width, height, bytes(canvas))

    glyph_w = len(digits) * 6 - 1
    start_x = max(1, (width - glyph_w) // 2)
    start_y = max(1, (height - 7) // 2)

    for d_idx, ch in enumerate(digits):
        rows = _FONT_5X7[ch]
        gx = start_x + d_idx * 6
        for ry, mask in enumerate(rows):
            py = start_y + ry
            if py >= height:
                continue
            for rx in range(5):
                px = gx + rx
                if px >= width:
                    continue
                if (mask >> (4 - rx)) & 1:
                    off = (py * width + px) * 3
                    canvas[off] = 255
                    canvas[off + 1] = 255
                    canvas[off + 2] = 255
    return ImageBuffer(width, height, bytes(canvas))


def _stack_vertical(strips: Sequence[ImageBuffer], divider_px: int = 1) -> ImageBuffer:
    width = strips[0].width
    total_h = sum(s.height for s in strips) + divider_px * max(0, len(strips) - 1)
    out = bytearray(width * total_h * 3)
    row_bytes = width * 3
    cur_y = 0
    for idx, strip in enumerate(strips):
        n_bytes = strip.height * row_bytes
        dst_off = cur_y * row_bytes
        out[dst_off : dst_off + n_bytes] = strip.rgb
        cur_y += strip.height
        if idx + 1 < len(strips) and divider_px > 0:
            div_off = cur_y * row_bytes
            div_len = divider_px * row_bytes
            out[div_off : div_off + div_len] = bytes([64] * div_len)
            cur_y += divider_px
    return ImageBuffer(width, total_h, bytes(out))


def _join_horizontal(columns: Sequence[ImageBuffer], divider_px: int = 2) -> ImageBuffer:
    height = columns[0].height
    total_w = sum(c.width for c in columns) + divider_px * max(0, len(columns) - 1)
    out = bytearray(total_w * height * 3)
    dst_row_bytes = total_w * 3
    for y in range(height):
        dst_x = 0
        for idx, col in enumerate(columns):
            cw_bytes = col.width * 3
            s_off = y * cw_bytes
            d_off = y * dst_row_bytes + dst_x * 3
            out[d_off : d_off + cw_bytes] = col.rgb[s_off : s_off + cw_bytes]
            dst_x += col.width
            if idx + 1 < len(columns) and divider_px > 0:
                div_off = y * dst_row_bytes + dst_x * 3
                out[div_off : div_off + divider_px * 3] = bytes([44, 52, 68] * divider_px)
                dst_x += divider_px
    return ImageBuffer(total_w, height, bytes(out))


def _build_metadata_diff_hints(
    candidate_metadata: Sequence[Mapping[str, str]] | None,
    y_start: int,
    y_end: int,
) -> list[str]:
    hints: list[str] = [
        f"Saliency band rows {y_start}..{y_end} (2x zoom) and bottom-30% net-weight strip (2x zoom)."
    ]
    if not candidate_metadata or len(candidate_metadata) < 2:
        return hints

    panel_tokens: list[list[str]] = []
    for meta in candidate_metadata:
        joined = " ".join(f"{k} {v}" for k, v in sorted(meta.items()))
        panel_tokens.append(split_identifier_tokens(joined))

    for i in range(len(candidate_metadata) - 1):
        set_i = set(panel_tokens[i])
        set_j = set(panel_tokens[i + 1])
        only_i = [t for t in panel_tokens[i] if t not in set_j][:3]
        only_j = [t for t in panel_tokens[i + 1] if t not in set_i][:3]
        if only_i or only_j:
            lhs = ", ".join(only_i) if only_i else "shared base"
            rhs = ", ".join(only_j) if only_j else "shared base"
            hints.append(f"Panel {i + 1} has [{lhs}] vs Panel {i + 2} has [{rhs}]")
    return hints


def compile_dual_buffer_contact_sheet(
    query: ImageBuffer,
    candidates: Sequence[ImageBuffer],
    *,
    candidate_metadata: Sequence[Mapping[str, str]] | None = None,
    curved_container: bool = False,
    band_height_fraction: float | None = None,
) -> CompiledContactSheet:
    """Compile a Stage-2B high-contrast multi-panel contact sheet while preserving unsharpened Stage-2A color."""
    if query.width < 4 or query.height < 4:
        raise ValueError(f"query ImageBuffer must be at least 4x4, got {query.width}x{query.height}")
    if not candidates or len(candidates) > 25:
        raise ValueError(f"candidates must contain 1..25 ImageBuffers, got {len(candidates)}")
    if candidate_metadata is not None and len(candidate_metadata) != len(candidates):
        raise ValueError("candidate_metadata length must match candidates length")

    # Stage 2A buffer: Minkowski p=6 white balance ONLY (strictly unsharpened for embedding & CIE Lab).
    stage2a_query = shades_of_gray_white_balance(query, p=6.0)

    pw = max(16, min(96, query.width))
    ph = max(16, min(96, query.height))
    col_w = int(round(pw * 1.15)) if curved_container else pw

    wb_candidates = [
        shades_of_gray_white_balance(cand, p=6.0).resize_bilinear(pw, ph)
        for cand in candidates
    ]
    y_start, y_end, variance_profile = extract_inter_candidate_saliency_band(
        wb_candidates,
        band_height_fraction=band_height_fraction,
    )

    all_base_panels: list[ImageBuffer] = [stage2a_query.resize_bilinear(pw, ph)] + wb_candidates
    panel_labels = [str(idx) for idx in range(len(all_base_panels))]

    sal_h = max(1, y_end - y_start)
    bot_y0 = min(ph - 1, int(0.70 * ph))
    bot_h = max(1, ph - bot_y0)
    mid_y0 = int(0.25 * ph)
    mid_y1 = max(mid_y0 + 1, int(0.75 * ph))

    columns: list[ImageBuffer] = []
    for label, base_panel in zip(panel_labels, all_base_panels):
        # Stage 2B local contrast enhancement runs ONLY on cloned Stage-2B panel copies.
        s2b_panel = _apply_stage2b_local_contrast(base_panel, amount=0.65)
        header = _render_label_header(col_w, 12, label)
        main_view = s2b_panel.resize_bilinear(col_w, ph) if col_w != pw else s2b_panel
        sal_zoom = s2b_panel.crop(0, y_start, pw, y_end).resize_bilinear(col_w, sal_h * 2)
        weight_zoom = s2b_panel.crop(0, bot_y0, pw, ph).resize_bilinear(col_w, bot_h * 2)

        strips: list[ImageBuffer] = [header, main_view]
        if curved_container:
            unwarped = _cylindrical_horizontal_unwarp(s2b_panel.crop(0, mid_y0, pw, mid_y1), col_w)
            strips.append(unwarped)
        strips.append(sal_zoom)
        strips.append(weight_zoom)
        columns.append(_stack_vertical(strips, divider_px=1))

    composite = _join_horizontal(columns, divider_px=2)
    png_bytes = encode_png(composite)
    # Map saliency_band back to candidate input row coordinates if candidate[0] had height != ph
    cand0_h = candidates[0].height
    if cand0_h != ph:
        orig_y0, orig_y1, orig_var = extract_inter_candidate_saliency_band(
            candidates,
            band_height_fraction=band_height_fraction,
        )
    else:
        orig_y0, orig_y1, orig_var = y_start, y_end, variance_profile

    diff_hints = _build_metadata_diff_hints(candidate_metadata, orig_y0, orig_y1)
    return CompiledContactSheet(
        png_bytes=png_bytes,
        stage2a_query_buffer=stage2a_query,
        saliency_band=(orig_y0, orig_y1),
        variance_profile=orig_var,
        diff_hints=diff_hints,
        panel_labels=panel_labels,
    )
