#!/usr/bin/env python3
"""
Abstract Image Creator  (GUI version)
Pick shapes, textures, a color palette, and effects (bloom, lens
flares, light leaks, chromatic aberration, color grading...) or let the script use random values,
then generate unique abstract PNGs. Images are saved in the same folder as this script.
Saved color palettes are stored beside it in custom_palettes.json

Install dependencies:  pip install pillow numpy
           (tkinter ships with Python; on Linux: sudo apt install python3-tk)

To run:       python abstract_art_gui.py
"""

# ============================================================
# >>> DEFAULT RESOLUTION (width, height) <<<
# The window starts with this value; you can also change it in the GUI.
RESOLUTION = (1920, 1080)
# ============================================================

import colorsys
import json
import math
import os
import queue
import random
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import tkinter as tk
    from tkinter import colorchooser, messagebox, simpledialog, ttk
    from PIL import ImageTk
except ImportError:  # GUI libraries missing; handled in main()
    tk = None

W, H = RESOLUTION
SCALE = max(W, H)
_RS = getattr(Image, "Resampling", Image)
LANCZOS, BICUBIC, BOX = _RS.LANCZOS, _RS.BICUBIC, _RS.BOX
TAU = math.tau


# ====================================================================
#  COLOR + MATH HELPERS
# ====================================================================
def hex_to_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def rgb_to_hex(c):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in c)


PRESETS = {
    "Sunset":     ["#ff512f", "#f09819", "#dd2476", "#5b2a86", "#ffd194"],
    "Ocean":      ["#0f2027", "#2c5364", "#00c6ff", "#0072ff", "#7fdbda"],
    "Neon":       ["#ff00c8", "#00f0ff", "#7a00ff", "#39ff14", "#ffee00"],
    "Pastel":     ["#ffd1dc", "#c1e1ff", "#d4f0c0", "#fff3b0", "#e0c3fc"],
    "Earth":      ["#5e3023", "#895737", "#c08552", "#dab49d", "#3f4a3c"],
    "Forest":     ["#0b3d2e", "#1b7f5c", "#6bbf59", "#d6e685", "#2d4a22"],
    "Candy":      ["#ff6ec7", "#ffb3de", "#6ee7ff", "#b5ff6e", "#ffe66e"],
    "Fire & Ice": ["#ff3b00", "#ffb400", "#00c2ff", "#0066ff", "#ffffff"],
    "Cyberpunk":  ["#0d0221", "#261447", "#ff2a6d", "#05d9e8", "#d1f7ff"],
    "Vaporwave":  ["#ff71ce", "#01cdfe", "#05ffa1", "#b967ff", "#fffb96"],
    "Monochrome": ["#111111", "#444444", "#888888", "#cccccc", "#ffffff"],
}


PALETTE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_palettes.json")
STAR = "\u2605 "   # prefix that marks saved palettes in the dropdown


def _valid_hex(c):
    try:
        return isinstance(c, str) and len(c) == 7 and c[0] == "#" and int(c[1:], 16) >= 0
    except ValueError:
        return False


def load_custom_palettes():
    """Read saved palettes from disk ({name: [hex, ...]}); ignores a missing/corrupt file."""
    try:
        with open(PALETTE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): [c.lower() for c in v] for k, v in data.items()
                if isinstance(v, list) and 2 <= len(v) <= 10 and all(_valid_hex(c) for c in v)}
    except (OSError, ValueError, AttributeError):
        return {}


def write_custom_palettes(palettes):
    with open(PALETTE_FILE, "w", encoding="utf-8") as f:
        json.dump(palettes, f, indent=2)


def random_palette(n=None):
    n = n or random.randint(3, 6)
    base = random.random()
    scheme = random.choice(["analogous", "complementary", "triad", "random"])
    out = []
    for _ in range(n):
        if scheme == "analogous":
            h = base + random.uniform(-0.1, 0.1)
        elif scheme == "complementary":
            h = base + random.choice([0, 0.5]) + random.uniform(-0.04, 0.04)
        elif scheme == "triad":
            h = base + random.choice([0, 1 / 3, 2 / 3]) + random.uniform(-0.03, 0.03)
        else:
            h = random.random()
        r, g, b = colorsys.hsv_to_rgb(h % 1, random.uniform(0.45, 1), random.uniform(0.5, 1))
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def jitter(c, hj=0.02, sj=0.12, vj=0.12):
    """Slightly vary a color so repeated palette colors don't look identical."""
    h, s, v = colorsys.rgb_to_hsv(*(x / 255 for x in c))
    h = (h + random.uniform(-hj, hj)) % 1
    s = min(1, max(0, s + random.uniform(-sj, sj)))
    v = min(1, max(0, v + random.uniform(-vj, vj)))
    return tuple(x * 255 for x in colorsys.hsv_to_rgb(h, s, v))


def colormap(t, colors):
    """Map an array of 0..1 values onto a list of colors."""
    arr = np.array(colors, np.float32)
    stops = np.linspace(0, 1, len(arr))
    return np.stack([np.interp(t, stops, arr[:, c]) for c in range(3)], -1).astype(np.float32)


def gradient_array(w, h, colors, mode=None):
    mode = mode or random.choice(["linear", "radial"])
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    if mode == "radial":
        cx, cy = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
        t = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    else:
        a = random.uniform(0, TAU)
        t = (x - w / 2) * math.cos(a) + (y - h / 2) * math.sin(a)
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    return colormap(t, colors)


def fbm(T, octaves=4, base=4):
    """Cheap fractal noise (0..1) made from upscaled random grids."""
    out = np.zeros((T, T), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        g = base * 2 ** o
        a = (np.random.rand(g, g) * 255).astype(np.uint8)
        im = Image.fromarray(a).resize((T, T), BICUBIC)
        out += amp * np.asarray(im, np.float32) / 255
        tot += amp
        amp *= 0.5
    return out / tot


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _norm3(x, y, z):
    m = np.sqrt(x * x + y * y + z * z) + 1e-9
    return x / m, y / m, z / m


# ====================================================================
#  SHAPES   (each draws white onto a black mask; returns a "bevel" hint)
#  Registration order = order shown in the GUI.
# ====================================================================
SHAPES = {}   # key -> (label, group, drawer)


def shape(key, label, group):
    def deco(fn):
        SHAPES[key] = (label, group, fn)
        return fn
    return deco


def thick_path(d, pts, w):
    """Draw a round-capped stroke by stamping overlapping circles (no joint artifacts)."""
    w = max(1.0, float(w))
    r, step = w / 2, max(1.0, w / 4)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for i in range(n + 1):
            t = i / n
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)


def ribbon_polygon(d, pts, w):
    """Draw a flat-ended stroke as one polygon offset from the centerline."""
    left, right = [], []
    for i, (x, y) in enumerate(pts):
        a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        m = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / m * w / 2, dx / m * w / 2
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
    d.polygon(left + right[::-1], fill=255)


def fit_points(pts, c, R):
    """Center and scale a point list to fit inside radius R around c."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    rad = max(math.hypot(x - cx, y - cy) for x, y in pts)
    s = R / rad
    return [(c + (x - cx) * s, c + (y - cy) * s) for x, y in pts]


# ---------- basic ----------
@shape("circle", "Circle", "Basic")
def _circle(d, N):
    c, R = N / 2, N * 0.40
    d.ellipse([c - R, c - R, c + R, c + R], fill=255)


@shape("ellipse", "Ellipse", "Basic")
def _ellipse(d, N):
    c, R = N / 2, N * 0.40
    ry = R * random.uniform(0.35, 0.7)
    d.ellipse([c - R, c - ry, c + R, c + ry], fill=255)


@shape("rect", "Rounded rectangle", "Basic")
def _rect(d, N):
    c, R = N / 2, N * 0.40
    hw, hh = R * random.uniform(0.5, 0.85), R * random.uniform(0.3, 0.75)
    d.rounded_rectangle([c - hw, c - hh, c + hw, c + hh],
                        radius=min(hw, hh) * random.uniform(0, 0.6), fill=255)


@shape("capsule", "Capsule", "Basic")
def _capsule(d, N):
    c, R = N / 2, N * 0.40
    hw, hh = R * random.uniform(0.6, 0.95), R * random.uniform(0.2, 0.4)
    d.rounded_rectangle([c - hw, c - hh, c + hw, c + hh], radius=hh, fill=255)


@shape("triangle", "Triangle", "Basic")
def _triangle(d, N):
    c, R = N / 2, N * 0.40
    a0 = random.uniform(0, TAU)
    d.polygon([(c + R * random.uniform(0.8, 1) * math.cos(a0 + i * TAU / 3),
                c + R * random.uniform(0.8, 1) * math.sin(a0 + i * TAU / 3)) for i in range(3)], fill=255)


@shape("diamond", "Diamond", "Basic")
def _diamond(d, N):
    c, R = N / 2, N * 0.40
    v, s = R * random.uniform(0.8, 1), R * random.uniform(0.4, 0.8)
    d.polygon([(c, c - v), (c + s, c), (c, c + v), (c - s, c)], fill=255)


@shape("polygon", "Irregular polygon", "Basic")
def _polygon(d, N):
    c, R = N / 2, N * 0.40
    n = random.randint(5, 9)
    d.polygon([(c + R * random.uniform(0.55, 1) * math.cos(TAU * (i + random.uniform(-0.3, 0.3)) / n),
                c + R * random.uniform(0.55, 1) * math.sin(TAU * (i + random.uniform(-0.3, 0.3)) / n))
               for i in range(n)], fill=255)


@shape("ngon", "Hexagon / Octagon", "Basic")
def _ngon(d, N):
    c, R = N / 2, N * 0.40
    n, a0 = random.choice([5, 6, 7, 8, 10, 12]), random.uniform(0, TAU)
    d.polygon([(c + R * math.cos(a0 + i * TAU / n), c + R * math.sin(a0 + i * TAU / n)) for i in range(n)], fill=255)


@shape("ring", "Ring", "Basic")
def _ring(d, N):
    c, R = N / 2, N * 0.40
    w = R * random.uniform(0.12, 0.35)
    d.ellipse([c - R, c - R, c + R, c + R], fill=255)
    d.ellipse([c - R + w, c - R + w, c + R - w, c + R - w], fill=0)
    return w * 0.5


@shape("arc", "Arc", "Basic")
def _arc(d, N):
    c, R = N / 2, N * 0.40
    ra, w = R * random.uniform(0.55, 0.85), R * random.uniform(0.15, 0.35)
    a0, span = random.uniform(0, TAU), math.radians(random.uniform(70, 300))
    pts = [(c + ra * math.cos(a0 + span * i / 60), c + ra * math.sin(a0 + span * i / 60)) for i in range(61)]
    thick_path(d, pts, w)
    return w * 0.5


@shape("blob", "Organic blob", "Basic")
def _blob(d, N):
    c, R = N / 2, N * 0.40
    harm = [(k, random.uniform(0, 0.2) / math.sqrt(k - 1), random.uniform(0, TAU)) for k in range(2, 6)]
    pts = []
    for i in range(140):
        t = TAU * i / 140
        r = min(1.0, 0.62 + sum(a * math.cos(k * t + p) for k, a, p in harm))
        pts.append((c + R * r * math.cos(t), c + R * r * math.sin(t)))
    d.polygon(pts, fill=255)


@shape("line", "Line / stroke", "Basic")
def _line(d, N):
    c, R = N / 2, N * 0.40
    w = R * random.uniform(0.08, 0.2)
    thick_path(d, [(c - R * 0.95, c), (c + R * 0.95, c)], w)
    return w * 0.5


@shape("pie", "Pie slice", "Basic")
def _pie(d, N):
    c, R = N / 2, N * 0.40
    s = random.uniform(0, 360)
    d.pieslice([c - R, c - R, c + R, c + R], s, s + random.uniform(70, 260), fill=255)


# ---------- stars (4 - 8 points) ----------
def _make_star(points):
    def fn(d, N):
        c, R = N / 2, N * 0.40
        inner = R * (random.uniform(0.28, 0.42) if points == 4 else random.uniform(0.38, 0.55))
        a0 = random.uniform(0, TAU)
        pts = []
        for i in range(points * 2):
            r = R if i % 2 == 0 else inner
            a = a0 + i * math.pi / points
            pts.append((c + r * math.cos(a), c + r * math.sin(a)))
        d.polygon(pts, fill=255)
        return N * 0.06
    return fn


for _p in range(4, 9):
    SHAPES[f"star{_p}"] = (f"{_p}-point star", "Stars", _make_star(_p))


# ---------- unique ----------
@shape("elbow", "Elbow pipe (bent cylinder)", "Unique")
def _elbow(d, N):
    c, R = N / 2, N * 0.40
    w = R * random.uniform(0.24, 0.36)
    br = R * random.uniform(0.36, 0.55)
    L1, L2 = R * random.uniform(0.55, 0.85), R * random.uniform(0.55, 0.85)
    left, yh = c - (L1 + br) / 2, c - (br + L2) / 2
    xb = left + L1
    pts = [(left, yh)]
    for i in range(41):  # quarter-circle bend
        t = -math.pi / 2 + (math.pi / 2) * i / 40
        pts.append((xb + br * math.cos(t), yh + br + br * math.sin(t)))
    pts.append((xb + br, yh + br + L2))
    ribbon_polygon(d, pts, w)
    # pipe collars at both ends
    wc, lc = w * 1.28, w * 0.38
    d.rectangle([left, yh - wc / 2, left + lc, yh + wc / 2], fill=255)
    ey = yh + br + L2
    d.rectangle([xb + br - wc / 2, ey - lc, xb + br + wc / 2, ey], fill=255)
    return w * 0.4


@shape("crescent", "Crescent moon", "Unique")
def _crescent(d, N):
    c, R = N / 2, N * 0.40
    d.ellipse([c - R, c - R, c + R, c + R], fill=255)
    r2, off = R * random.uniform(0.75, 0.95), R * random.uniform(0.3, 0.55)
    d.ellipse([c + off - r2, c - r2, c + off + r2, c + r2], fill=0)
    return N * 0.05


@shape("spiral", "Spiral", "Unique")
def _spiral(d, N):
    c, R = N / 2, N * 0.40
    w, turns = R * random.uniform(0.09, 0.17), random.uniform(2, 3.5)
    pts = []
    for i in range(400):
        t = i / 399
        r = R * (0.08 + 0.92 * t) * 0.97
        pts.append((c + r * math.cos(turns * TAU * t), c + r * math.sin(turns * TAU * t)))
    thick_path(d, pts, w)
    return w * 0.5


@shape("flower", "Flower / rosette", "Unique")
def _flower(d, N):
    c, R = N / 2, N * 0.40
    k, a = random.randint(5, 9), random.uniform(0.15, 0.4)
    pts = []
    for i in range(240):
        t = TAU * i / 240
        r = R * (a + (1 - a) * abs(math.cos(k * t / 2)))
        pts.append((c + r * math.cos(t), c + r * math.sin(t)))
    d.polygon(pts, fill=255)
    if random.random() < 0.5:
        r0 = R * random.uniform(0.1, 0.22)
        d.ellipse([c - r0, c - r0, c + r0, c + r0], fill=0)


@shape("gear", "Gear", "Unique")
def _gear(d, N):
    c, R = N / 2, N * 0.40
    n, rin = random.randint(8, 14), R * random.uniform(0.74, 0.86)
    h, pts = math.pi / n * 0.45, []
    for i in range(n):
        a = TAU * i / n
        for da, r in ((-h * 1.4, rin), (-h * 0.9, R), (h * 0.9, R), (h * 1.4, rin)):
            pts.append((c + r * math.cos(a + da), c + r * math.sin(a + da)))
    d.polygon(pts, fill=255)
    rh = R * random.uniform(0.25, 0.4)
    d.ellipse([c - rh, c - rh, c + rh, c + rh], fill=0)


@shape("cross", "Cross / plus", "Unique")
def _cross(d, N):
    c, R = N / 2, N * 0.40
    t = R * random.uniform(0.28, 0.45)
    rad = t * random.uniform(0, 0.5)
    d.rounded_rectangle([c - R, c - t / 2, c + R, c + t / 2], radius=rad, fill=255)
    d.rounded_rectangle([c - t / 2, c - R, c + t / 2, c + R], radius=rad, fill=255)


@shape("teardrop", "Teardrop", "Unique")
def _teardrop(d, N):
    c, R = N / 2, N * 0.40
    m = random.uniform(1.0, 2.2)
    pts = [(math.sin(t) * math.sin(t / 2) ** m, -math.cos(t)) for t in np.linspace(0, TAU, 160)]
    d.polygon(fit_points(pts, c, R), fill=255)


@shape("wave", "Wavy ribbon", "Unique")
def _wave(d, N):
    c, R = N / 2, N * 0.40
    amp, per, ph = R * random.uniform(0.2, 0.5), random.uniform(1, 2.5), random.uniform(0, TAU)
    w = R * random.uniform(0.12, 0.25)
    pts = [(c + R * (2 * i / 120 - 1) * 0.95, c + amp * math.sin(TAU * per * i / 120 + ph)) for i in range(121)]
    thick_path(d, pts, w)
    return w * 0.5


@shape("bolt", "Lightning bolt", "Unique")
def _bolt(d, N):
    c, R = N / 2, N * 0.40
    base = [(0.55, 0), (0.1, 0.56), (0.42, 0.56), (0.25, 1.0), (0.9, 0.38), (0.57, 0.38), (0.78, 0)]
    d.polygon(fit_points(base, c, R), fill=255)


@shape("spiro", "Spirograph (wire)", "Unique")
def _spiro(d, N):
    c, R = N / 2, N * 0.40
    Rr, r = random.randint(5, 9), random.randint(2, 4)
    dd = r * random.uniform(0.6, 1.4)
    ts = np.linspace(0, TAU * r / math.gcd(Rr, r), 900)
    xs = (Rr - r) * np.cos(ts) + dd * np.cos((Rr - r) / r * ts)
    ys = (Rr - r) * np.sin(ts) - dd * np.sin((Rr - r) / r * ts)
    pts = fit_points(list(zip(xs.tolist(), ys.tolist())), c, R * 0.97)
    w = max(2.0, N * 0.007)
    d.line(pts, fill=255, width=int(w), joint="curve")
    return w * 0.7


@shape("burst", "Spiky burst", "Unique")
def _burst(d, N):
    c, R = N / 2, N * 0.40
    n, pts = random.randint(10, 22), []
    for i in range(n * 2):
        r = R * (random.uniform(0.75, 1) if i % 2 == 0 else random.uniform(0.4, 0.6))
        a = i * math.pi / n
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    d.polygon(pts, fill=255)
    return N * 0.06


@shape("shard", "Crystal shard", "Unique")
def _shard(d, N):
    c, R = N / 2, N * 0.40
    d.polygon([(c - R, c), (c - R * 0.1, c - R * random.uniform(0.12, 0.3)),
               (c + R, c + R * random.uniform(-0.1, 0.1)), (c + R * 0.2, c + R * random.uniform(0.12, 0.35))], fill=255)
    return N * 0.04


# ====================================================================
#  TEXTURES / MATERIALS
# ====================================================================
TEXTURES = {
    "flat":     "Flat color",
    "gradient": "Smooth gradient",
    "matte":    "Dull / matte",
    "glossy":   "Shiny / glossy",
    "metallic": "Metallic",
    "rough":    "Rough / gritty",
    "glass":    "Glass",
    "neon":     "Neon glow",
    "striped":  "Striped",
    "dotted":   "Halftone dots",
    "marble":   "Marble",
    "pearl":    "Pearlescent",
}


def surface(mask_img, bevel, bump=None):
    """Fake a 3D height map from the shape mask and return its surface normals."""
    sig = max(1.0, float(bevel))

    def bl(s):
        return np.asarray(mask_img.filter(ImageFilter.GaussianBlur(s)), np.float32) / 255.0

    h = 0.5 * bl(sig) + 0.3 * bl(sig * 2.2) + 0.2 * bl(sig * 4.5)
    gy, gx = np.gradient(h)
    k = sig * 2.4
    nx, ny = -gx * k, -gy * k
    if bump is not None:
        nx, ny = nx + bump[0], ny + bump[1]
    return _norm3(nx, ny, np.ones_like(nx))


def diffuse(n, light):
    return np.clip(n[0] * light[0] + n[1] * light[1] + n[2] * light[2], 0, 1)


def specular(n, light, shin):
    hx, hy, hz = _norm3(light[0], light[1], light[2] + 1.0)
    return np.clip(n[0] * hx + n[1] * hy + n[2] * hz, 0, 1) ** shin


def make_material(mat, mask_img, bevel, palette, light):
    """Return (rgb 0..255 float array, alpha 0..1 array) for a shape tile."""
    T = mask_img.size[0]
    m = np.asarray(mask_img, np.float32) / 255.0
    c1 = np.array(jitter(random.choice(palette)), np.float32)
    c2 = np.array(jitter(random.choice(palette)), np.float32)
    alpha = m.copy()
    y, x = np.mgrid[0:T, 0:T].astype(np.float32)

    if mat == "flat":
        rgb = np.broadcast_to(c1, (T, T, 3)).copy()

    elif mat == "gradient":
        cols = [tuple(c1), tuple(c2)]
        if random.random() < 0.5:
            cols.append(jitter(random.choice(palette)))
        rgb = gradient_array(T, T, cols)
        rgb *= (0.9 + 0.2 * diffuse(surface(mask_img, bevel), light))[..., None]

    elif mat == "matte":
        n = surface(mask_img, bevel)
        rgb = c1 * (0.30 + 0.80 * diffuse(n, light))[..., None]
        rgb = 0.85 * rgb + 0.15 * rgb.mean(axis=2, keepdims=True)  # slightly chalky

    elif mat == "glossy":
        n = surface(mask_img, bevel)
        rgb = c1 * (0.22 + 0.78 * diffuse(n, light))[..., None]
        rgb += 255 * (0.95 * specular(n, light, 90) + 0.18 * specular(n, light, 14))[..., None]
        rgb += ((1 - n[2]) ** 2)[..., None] * 0.35 * (c2 * 0.6 + 80)

    elif mat == "metallic":
        n = surface(mask_img, bevel)
        phi, ph = random.uniform(0, TAU), random.uniform(0, TAU)
        e = 0.5 + 0.5 * np.sin(7 * (n[0] * math.cos(phi) + n[1] * math.sin(phi)) + 2.2 * (1 - n[2]) + ph)
        rgb = c1 * (0.2 + 0.95 * e)[..., None]
        rgb *= (0.94 + 0.12 * np.random.rand(T, 1, 1))              # brushed streaks
        rgb += 200 * specular(n, light, 40)[..., None]

    elif mat == "rough":
        nz = fbm(T, 4, 6)
        gy, gx = np.gradient(nz)
        s = 0.5 / (max(gx.std(), gy.std()) + 1e-6)
        n = surface(mask_img, bevel, bump=(-gx * s, -gy * s))
        grit = 0.78 + 0.44 * np.random.rand(T, T)
        rgb = c1 * (0.25 + 0.85 * diffuse(n, light))[..., None] * grit[..., None]
        rgb += 255 * 0.12 * specular(n, light, 6)[..., None]

    elif mat == "glass":
        n = surface(mask_img, bevel)
        sp, fres = specular(n, light, 70), (1 - n[2]) ** 1.3
        alpha = m * np.clip(0.16 + 0.75 * fres + 1.1 * sp, 0, 1)
        rgb = c1 * (0.75 + 0.35 * diffuse(n, light))[..., None]
        rgb += 255 * (0.9 * sp + 0.25 * fres)[..., None]

    elif mat == "neon":
        blur = lambda s: np.asarray(mask_img.filter(ImageFilter.GaussianBlur(s)), np.float32) / 255
        bright = np.clip(c1 * 1.2 + 40, 0, 255)
        core = np.clip(blur(max(1.5, bevel)), 0, 1) ** 3
        glow = blur(max(2.0, T * 0.03))
        rgb = bright * (1 - core)[..., None] + 255 * 0.95 * core[..., None]
        alpha = np.clip(np.maximum(m, glow * 0.9), 0, 1)

    elif mat == "striped":
        n = surface(mask_img, bevel)
        a, per = random.uniform(0, math.pi), T * random.uniform(0.035, 0.11)
        t = (x * math.cos(a) + y * math.sin(a)) / per
        s = np.abs((t % 1.0) * 2 - 1)
        wdt = random.uniform(0.35, 0.65)
        k = smoothstep(wdt - 0.05, wdt + 0.05, s)[..., None]
        rgb = (c1 * (1 - k) + c2 * k) * (0.55 + 0.55 * diffuse(n, light))[..., None]

    elif mat == "dotted":
        n = surface(mask_img, bevel)
        a, cell = random.choice([0, math.pi / 4]), T * random.uniform(0.035, 0.08)
        u = (x * math.cos(a) + y * math.sin(a)) / cell
        v = (-x * math.sin(a) + y * math.cos(a)) / cell
        dist = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2)
        rad = 0.18 + 0.38 * diffuse(n, light)
        k = (1 - smoothstep(rad - 0.06, rad + 0.02, dist))[..., None]
        rgb = (c2 * 0.35 * (1 - k) + c1 * k) * (0.8 + 0.3 * diffuse(n, light))[..., None]

    elif mat == "marble":
        n = surface(mask_img, bevel)
        noise = fbm(T, 5, 3)
        t = x / T * random.uniform(2, 5) + y / T * random.uniform(0, 2) + 3.2 * noise
        vein = (1 - np.abs(np.sin(t * math.pi)) ** random.uniform(0.15, 0.4))[..., None]
        base = c1 * 0.9 + (c2 - c1) * 0.3 * noise[..., None]
        light_vein = np.clip(0.5 * c2 + 127, 0, 255)
        rgb = (base * (1 - vein) + light_vein * vein) * (0.45 + 0.65 * diffuse(n, light))[..., None]
        rgb += 255 * 0.5 * specular(n, light, 50)[..., None]

    else:  # "pearl" - iridescent colors from the palette that shift with the surface angle
        n = surface(mask_img, bevel)
        phi = random.uniform(0, TAU)
        t = (0.9 * (n[0] * math.cos(phi) + n[1] * math.sin(phi)) + 1.1 * (1 - n[2])
             + 0.7 * fbm(T, 3, 3) + random.random()) % 1.0
        cols = [tuple(p) for p in palette] + [tuple(palette[0])]
        rgb = colormap(t, cols) * (0.65 + 0.5 * diffuse(n, light))[..., None]
        rgb += 255 * 0.8 * specular(n, light, 45)[..., None]

    return rgb, alpha


# ====================================================================
#  SPARKLES
# ====================================================================
def sparkle_tile(size, kind):
    """A glowing glint: bright core, soft halo, 4 (or 8) thin rays."""
    r = np.linspace(-1, 1, size, dtype=np.float32)
    u, v = np.meshgrid(r, r)
    r2 = u * u + v * v
    glow = np.exp(-r2 * 16)
    core = np.exp(-r2 * 110)
    if kind == "dot":
        return glow * 0.55 + core * 1.2, glow
    au, av = np.abs(u), np.abs(v)
    rays = np.exp(-av * 55) * np.exp(-au * 4.2) + np.exp(-au * 55) * np.exp(-av * 4.2)
    if kind == "star8":
        p, q = np.abs(u + v) * 0.7071, np.abs(u - v) * 0.7071
        rays += 0.4 * (np.exp(-q * 70) * np.exp(-p * 7) + np.exp(-p * 70) * np.exp(-q * 7))
    return rays + core * 1.2 + glow * 0.35, glow


def add_tile(arr, tile, x, y):
    h, w = tile.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 < x1 and y0 < y1:
        arr[y0:y1, x0:x1] += tile[y0 - y:y1 - y, x0 - x:x1 - x]


def add_sparkles(arr, specs, amount, palette):
    def place(size):
        if specs and random.random() < 0.6:           # glints near shapes
            s = random.choice(specs)
            a, rr = random.uniform(0, TAU), s["size"] * 0.5 * random.uniform(0.2, 1.0)
            cx, cy = s["cx"] + rr * math.cos(a), s["cy"] + rr * math.sin(a)
        else:
            cx, cy = random.uniform(0, W), random.uniform(0, H)
        return int(cx - size // 2), int(cy - size // 2)

    for i in range(int(amount * 3.5)):                # tiny dust motes
        size = max(7, int(SCALE * random.uniform(0.004, 0.014))) | 1
        main, glow = sparkle_tile(size, "dot")
        pc = np.array(random.choice(palette), np.float32)
        tile = main[..., None] * (0.7 * 255 + 0.3 * pc) * random.uniform(0.5, 1.0)
        add_tile(arr, tile, *place(size))
    for i in range(int(amount)):                      # big glints
        size = max(21, int(SCALE * random.uniform(0.025, 0.11))) | 1
        main, glow = sparkle_tile(size, random.choice(["star4", "star4", "star8"]))
        pc = np.array(random.choice(palette), np.float32)
        tile = (main[..., None] * (0.75 * 255 + 0.25 * pc) + glow[..., None] * pc * 0.5) * random.uniform(0.7, 1.0)
        add_tile(arr, tile, *place(size))


# ====================================================================
#  CINEMATIC POST-PROCESSING
#  Every effect works on a float RGB array (0..255, may overshoot) and
#  takes a strength s in 0..1.  Strength 0 = effect skipped.
# ====================================================================
CINE_FX = [   # (key, label) -> one slider each in the GUI
    ("bloom", "Bloom"),
    ("halation", "Halation (red glow)"),
    ("streaks", "Anamorphic streaks"),
    ("flare", "Lens flare"),
    ("leaks", "Light leaks"),
    ("chroma", "Chromatic aberration"),
    ("mist", "Soft focus / mist"),
    ("edgeblur", "Edge blur (lens)"),
    ("contrast", "Filmic contrast"),
    ("dust", "Dust & scratches"),
]

GRADES = {   # shadow tint, highlight tint, contrast, saturation, black lift
    "None": None,
    "Teal & Orange": dict(sh=(-0.07, 0.02, 0.08), hi=(0.10, 0.03, -0.09), con=1.10, sat=1.12, lift=0.0),
    "Warm Film":     dict(sh=(0.03, 0.0, -0.02), hi=(0.07, 0.025, -0.06), con=1.05, sat=1.05, lift=0.02),
    "Cold Moody":    dict(sh=(-0.03, 0.01, 0.08), hi=(-0.03, 0.0, 0.05), con=1.15, sat=0.80, lift=0.01),
    "Bleach Bypass": dict(sh=(0, 0, 0), hi=(0, 0, 0), con=1.30, sat=0.50, lift=0.0),
    "Cross Process": dict(sh=(0.0, 0.05, 0.09), hi=(0.11, 0.06, -0.11), con=1.18, sat=1.25, lift=0.0),
    "Faded Matte":   dict(sh=(0.02, 0.01, 0.03), hi=(0.03, 0.01, -0.01), con=0.92, sat=0.85, lift=0.09),
    "Vivid Pop":     dict(sh=(0, 0, 0), hi=(0, 0, 0), con=1.15, sat=1.45, lift=0.0),
    "Noir":          dict(sh=(0, 0, 0), hi=(0, 0, 0), con=1.35, sat=0.0, lift=0.0),
}

LETTERBOX = {"None": None, "2.39:1 (Scope)": 2.39, "2.00:1": 2.0, "1.85:1 (Flat)": 1.85}

CINE_LOOKS = {   # one-click starting points (any key not listed = 0 / None)
    "Off": {},
    "Subtle Film": dict(bloom=25, halation=20, chroma=15, mist=15, contrast=30, dust=15,
                        grade="Warm Film", grade_strength=50),
    "Blockbuster": dict(bloom=40, halation=30, streaks=35, flare=45, chroma=25, contrast=50, edgeblur=30,
                        grade="Teal & Orange", grade_strength=80, letterbox="2.39:1 (Scope)"),
    "Sci-Fi Anamorphic": dict(bloom=45, streaks=70, flare=60, chroma=35, edgeblur=35, contrast=35,
                              grade="Cold Moody", grade_strength=75, letterbox="2.39:1 (Scope)"),
    "Dreamy": dict(bloom=60, halation=40, leaks=50, mist=70, edgeblur=20,
                   grade="Faded Matte", grade_strength=70),
    "Vintage Film": dict(halation=50, leaks=60, chroma=20, mist=20, contrast=40, dust=60,
                         grade="Cross Process", grade_strength=55),
    "Noir": dict(bloom=20, contrast=70, edgeblur=30, dust=40,
                 grade="Noir", grade_strength=100, letterbox="1.85:1 (Flat)"),
}

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def random_cinema():
    cin = {k: 0 for k, _ in CINE_FX}
    for k, _ in random.sample(CINE_FX, random.randint(3, 6)):
        cin[k] = random.randint(20, 80)
    cin["grade"] = random.choice(list(GRADES)[1:]) if random.random() < 0.65 else "None"
    cin["grade_strength"] = random.randint(40, 100)
    cin["letterbox"] = random.choice(["2.39:1 (Scope)", "1.85:1 (Flat)"]) if random.random() < 0.2 else "None"
    return cin


def soft_blur(arr, radius):
    """Gaussian blur of a float RGB array. Big radii are blurred on a shrunken copy (fast)."""
    h, w = arr.shape[:2]
    radius = max(0.6, float(radius))
    f = int(max(1, min(16, radius // 3)))
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if f > 1:
        im = im.resize((max(1, w // f), max(1, h // f)), BOX)
    im = im.filter(ImageFilter.GaussianBlur(radius / f))
    if f > 1:
        im = im.resize((w, h), BICUBIC)
    return np.asarray(im, np.float32)


def radial_r(h, w):
    """Distance from the image center: 0 in the middle, ~1 in the corners."""
    x = ((np.arange(w, dtype=np.float32) - w / 2) / (w / 2))[None, :]
    y = ((np.arange(h, dtype=np.float32) - h / 2) / (h / 2))[:, None]
    return np.sqrt(x * x + y * y) / 1.41421


def bright_pass(arr):
    """Keep only the brightest parts of the image (adaptive, so dark palettes still glow)."""
    lum = arr @ LUMA
    thr = max(float(np.percentile(lum[::4, ::4], 90)), 80.0)
    k = smoothstep(thr * 0.75, min(250.0, thr + 70.0), lum)
    return arr * k[..., None]


def fx_bloom(arr, bp, s):
    glow = np.zeros_like(arr)
    for r, wgt in ((0.003, 0.55), (0.012, 0.40), (0.035, 0.32), (0.09, 0.28)):
        glow += soft_blur(bp, SCALE * r) * wgt
    arr += glow * (0.7 * s)


def fx_halation(arr, bp, s):
    """Film halation: highlights bleed a warm red/orange halo into their surroundings."""
    g = soft_blur(bp, SCALE * 0.012) * 0.6 + soft_blur(bp, SCALE * 0.03) * 0.5
    arr += (g @ LUMA)[..., None] * np.array([1.0, 0.30, 0.10], np.float32) * (1.4 * s)


def hbox(a, L):
    """Horizontal box blur (running sum) of an HxWx3 array."""
    L = int(max(3, L)) | 1
    pad = L // 2
    c = np.cumsum(np.pad(a, ((0, 0), (pad + 1, pad), (0, 0))), axis=1, dtype=np.float64)
    return ((c[:, L:] - c[:, :-L]) / L).astype(np.float32)


def fx_streaks(arr, bp, s):
    """Anamorphic-lens look: bright areas smear into long, cool-blue horizontal streaks."""
    h, w = arr.shape[:2]
    sw, sh = max(8, w // 2), max(8, h // 2)
    small = np.asarray(Image.fromarray(np.clip(bp, 0, 255).astype(np.uint8)).resize((sw, sh), BOX), np.float32)
    L = sw * random.uniform(0.35, 0.6)
    st = hbox(hbox(small, L), L * 0.55)
    st /= (np.percentile(st[::3, ::3], 99.7) + 1e-3)
    st = np.clip(st, 0, 1.2)
    tint = np.array([0.45, 0.75, 1.3], np.float32)
    col = st * 0.35 + st.mean(-1, keepdims=True) * tint * 0.8
    full = np.asarray(Image.fromarray(np.clip(col * 190, 0, 255).astype(np.uint8)).resize((w, h), BICUBIC), np.float32)
    arr += full * (0.8 * s)


def _rainbow(t):
    return 0.5 + 0.5 * np.cos(TAU * (np.asarray(t, np.float32)[..., None] + np.array([0.0, 1 / 3, 2 / 3], np.float32)))


def _tile_grid(size):
    r = np.linspace(-1, 1, size, dtype=np.float32)
    return np.meshgrid(r, r)


def fx_flare(arr, s, palette):
    """Lens flare: glow + rainbow halo + hexagonal/round ghosts + anamorphic streak + star glint.
    The flare is anchored to the brightest spot of the picture (usually)."""
    h, w = arr.shape[:2]
    tiny = np.asarray(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(
        (max(4, w // 24), max(4, h // 24)), BOX), np.float32) @ LUMA
    if random.random() < 0.75:
        iy, ix = np.unravel_index(int(np.argmax(tiny)), tiny.shape)
        sx, sy = (ix + 0.5) * w / tiny.shape[1], (iy + 0.5) * h / tiny.shape[0]
    else:
        sx, sy = random.uniform(0.1, 0.9) * w, random.uniform(0.05, 0.6) * h
    cx, cy = w / 2, h / 2
    amp = 255.0 * s * 1.6

    # 1) hot core + soft halo
    size = int(SCALE * 0.4) | 1
    u, v = _tile_grid(size)
    r2 = u * u + v * v
    g = (np.exp(-r2 * 10) * 0.55 + np.exp(-r2 * 160) * 1.3)[..., None] * np.array([1.0, 0.88, 0.7], np.float32)
    add_tile(arr, g * amp, int(sx - size // 2), int(sy - size // 2))

    # 2) rainbow halo ring
    size = int(SCALE * random.uniform(0.28, 0.5)) | 1
    u, v = _tile_grid(size)
    r = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u) / TAU + 0.5
    ring = np.exp(-(((r - 0.62) / 0.07) ** 2))
    add_tile(arr, ring[..., None] * (0.35 + 0.65 * _rainbow(ang)) * 0.4 * amp, int(sx - size // 2), int(sy - size // 2))

    # 3) ghosts along the line source -> image center -> beyond
    for k in random.sample([0.25, 0.5, 0.8, 1.15, 1.5, 1.9, 2.4], random.randint(4, 6)):
        gx, gy = sx + (cx - sx) * k, sy + (cy - sy) * k
        size = max(9, int(SCALE * random.uniform(0.03, 0.16))) | 1
        u, v = _tile_grid(size)
        if random.random() < 0.5:   # hexagonal aperture
            d = np.maximum(np.abs(v), np.abs(u) * 0.866 + np.abs(v) * 0.5)
        else:
            d = np.sqrt(u * u + v * v)
        fill = smoothstep(1.0, 0.88, d) * (0.3 + 0.7 * smoothstep(0.35, 1.0, d))
        if random.random() < 0.5:
            col = np.array(random.choice(palette), np.float32) / 255 * 0.6 + 0.4
        else:
            col = _rainbow(np.array([random.random()]))[0]
        add_tile(arr, fill[..., None] * col * random.uniform(0.12, 0.38) * amp, int(gx - size // 2), int(gy - size // 2))

    # 4) anamorphic horizontal streak through the source
    th = max(9, int(SCALE * 0.035)) | 1
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.linspace(-1, 1, th, dtype=np.float32)[:, None]
    streak = np.exp(-np.abs(xs - sx) / (w * 0.16)) * np.exp(-(ys * ys) * 40)
    add_tile(arr, streak[..., None] * np.array([0.45, 0.75, 1.0], np.float32) * 0.9 * amp, 0, int(sy - th // 2))

    # 5) star glint on the source
    if random.random() < 0.8:
        size = int(SCALE * 0.22) | 1
        main, _ = sparkle_tile(size, "star8")
        add_tile(arr, main[..., None] * np.array([1.0, 0.95, 0.85], np.float32) * 0.7 * amp,
                 int(sx - size // 2), int(sy - size // 2))


def fx_leaks(arr, s, palette):
    """Light leaks: big, soft, warm glows entering from the frame edges (screen blended)."""
    h, w = arr.shape[:2]
    lw, lh = max(32, w // 4), max(32, h // 4)
    xx = (np.arange(lw, dtype=np.float32) / lw)[None, :]
    yy = (np.arange(lh, dtype=np.float32) / lh)[:, None]
    warm = [(255, 60, 20), (255, 130, 25), (255, 190, 60), (240, 35, 90), (255, 80, 170), (255, 215, 120)]
    cool = [(40, 150, 255), (130, 70, 255)]
    out = np.zeros((lh, lw, 3), np.float32)
    for _ in range(random.randint(2, 4)):
        col = random.choice(cool if random.random() < 0.15 else warm)
        if random.random() < 0.2:
            col = random.choice(palette)
        side = random.randint(0, 3)
        if side == 0:
            cx, cy = random.uniform(-0.12, 0.1), random.uniform(0, 1)
        elif side == 1:
            cx, cy = random.uniform(0.9, 1.12), random.uniform(0, 1)
        elif side == 2:
            cx, cy = random.uniform(0, 1), random.uniform(-0.12, 0.1)
        else:
            cx, cy = random.uniform(0, 1), random.uniform(0.9, 1.12)
        a = random.uniform(0, math.pi)
        rx, ry = random.uniform(0.1, 0.3), random.uniform(0.3, 0.9)
        dx, dy = xx - cx, yy - cy
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        t = np.exp(-((u / rx) ** 2 + (v / ry) ** 2) * 1.3)
        out += colormap(t, [(0, 0, 0), col, (255, 240, 210)]) * random.uniform(0.6, 1.0)
    n = Image.fromarray((fbm(256, 4, 3) * 255).astype(np.uint8)).resize((lw, lh), BICUBIC)
    out *= (0.5 + np.asarray(n, np.float32) / 255)[..., None]
    leak = np.asarray(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((w, h), BICUBIC), np.float32) / 255
    np.clip(arr, 0, 255, out=arr)
    k = np.clip(leak * s * 1.3, 0, 1)
    arr[:] = 255 - (255 - arr) * (1 - k)


def fx_chromatic(arr, s):
    """Lateral chromatic aberration: red and blue are scaled differently from the center,
    so colored fringes grow toward the edges of the frame."""
    h, w = arr.shape[:2]
    k = 0.014 * s
    cx, cy = w / 2, h / 2
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    r, _, b = im.split()
    affine = getattr(Image, "Transform", Image).AFFINE

    def scaled(ch, f):
        a = 1.0 / f
        return np.asarray(ch.transform((w, h), affine, (a, 0, cx * (1 - a), 0, a, cy * (1 - a)), resample=BICUBIC), np.float32)

    arr[..., 0] = scaled(r, 1 + k)
    arr[..., 2] = scaled(b, 1 - k)


def fx_mist(arr, s):
    """Pro-Mist / soft-focus filter: gentle overall softness, glowing highlights, lifted blacks."""
    blurred = soft_blur(arr, SCALE * 0.012)
    a = 0.55 * s
    arr[:] = arr * (1 - a) + blurred * a
    arr += np.clip(soft_blur(arr, SCALE * 0.035) - 110, 0, None) * 0.22 * s
    arr[:] = arr * (1 - 0.05 * s) + 12 * s


def fx_edge_blur(arr, s):
    """Lens falloff: the picture softens toward the corners."""
    h, w = arr.shape[:2]
    blurred = soft_blur(arr, SCALE * 0.012 * (0.5 + s))
    mask = smoothstep(0.45, 1.05, radial_r(h, w)) * min(1.0, s)
    arr += (blurred - arr) * mask[..., None]


def fx_grade(arr, name, amount):
    p = GRADES.get(name)
    if not p or amount <= 0:
        return
    x = np.clip(arr, 0, 255) / 255
    lum = (x @ LUMA)[..., None]
    sh = 1 - smoothstep(0.0, 0.65, lum)
    hi = smoothstep(0.35, 1.0, lum)
    y = x + sh * np.array(p["sh"], np.float32) + hi * np.array(p["hi"], np.float32)
    gray = (y @ LUMA)[..., None]
    y = gray + (y - gray) * p["sat"]
    y = (y - 0.5) * p["con"] + 0.5
    y = y * (1 - p["lift"]) + p["lift"]
    arr[:] = (x + (np.clip(y, 0, 1) - x) * min(1.0, amount)) * 255


def fx_contrast(arr, s):
    """Filmic S-curve: deeper shadows, brighter highlights, smooth roll-off."""
    x = np.clip(arr, 0, 255) / 255
    arr[:] = (x + (x * x * (3 - 2 * x) - x) * min(1.0, s * 1.2)) * 255


def fx_dust(arr, s):
    """Film dust, hairs and vertical scratches (bright and dark)."""
    h, w = arr.shape[:2]
    area = (w * h) / (1920 * 1080)
    sc = SCALE / 1920
    bright, dark = Image.new("L", (w, h), 0), Image.new("L", (w, h), 0)
    db, dd = ImageDraw.Draw(bright), ImageDraw.Draw(dark)
    for d_, n in ((db, int((70 + 330 * s) * area)), (dd, int((30 + 120 * s) * area))):
        for _ in range(n):
            x, y = random.uniform(0, w), random.uniform(0, h)
            r = max(0.7, random.uniform(0.6, 2.4) * sc * (3.0 if random.random() < 0.04 else 1.0))
            d_.ellipse([x - r, y - r, x + r, y + r], fill=random.randint(110, 255))
    for d_ in (db, dd):
        for _ in range(int((4 + 18 * s) * area)):
            x, y, ang = random.uniform(0, w), random.uniform(0, h), random.uniform(0, TAU)
            pts = [(x, y)]
            for _ in range(random.randint(4, 8)):
                ang += random.uniform(-0.8, 0.8)
                step = random.uniform(4, 14) * sc
                x += math.cos(ang) * step
                y += math.sin(ang) * step
                pts.append((x, y))
            d_.line(pts, fill=random.randint(90, 220), width=max(1, int(sc)))
    for _ in range(int((1 + 5 * s) * max(1.0, w / 1920))):
        x, y0, ln = random.uniform(0, w), random.uniform(0, h), random.uniform(0.1, 0.6) * h
        db.line([(x, y0), (x + random.uniform(-3, 3) * sc, y0 + ln)], fill=random.randint(40, 120), width=1)
    blur = ImageFilter.GaussianBlur(0.5 + 0.4 * sc)
    b = np.asarray(bright.filter(blur), np.float32)[..., None]
    d = np.asarray(dark.filter(blur), np.float32)[..., None]
    arr += b * (0.5 + 0.6 * s)
    arr *= 1 - d / 255 * (0.30 + 0.4 * s)


def fx_letterbox(arr, ratio):
    h, w = arr.shape[:2]
    bar = int((h - w / ratio) / 2)
    if bar > 0:
        arr[:bar] = 0
        arr[h - bar:] = 0


def cinema_pre(arr, cin, palette):
    """Effects applied before the vignette and film grain."""
    lv = lambda k: float(cin.get(k, 0) or 0) / 100.0
    if lv("bloom") > 0 or lv("halation") > 0 or lv("streaks") > 0:
        bp = bright_pass(arr)
        if lv("bloom") > 0:
            fx_bloom(arr, bp, lv("bloom"))
        if lv("halation") > 0:
            fx_halation(arr, bp, lv("halation"))
        if lv("streaks") > 0:
            fx_streaks(arr, bp, lv("streaks"))
        del bp
    if lv("flare") > 0:
        fx_flare(arr, lv("flare"), palette)
    if lv("leaks") > 0:
        fx_leaks(arr, lv("leaks"), palette)
    if lv("mist") > 0:
        fx_mist(arr, lv("mist"))
    if lv("edgeblur") > 0:
        fx_edge_blur(arr, lv("edgeblur"))
    if lv("chroma") > 0:
        fx_chromatic(arr, lv("chroma"))
    fx_grade(arr, cin.get("grade", "None"), lv("grade_strength") if "grade_strength" in cin else 1.0)
    if lv("contrast") > 0:
        fx_contrast(arr, lv("contrast"))


def cinema_post(arr, cin):
    """Effects applied after the vignette and film grain."""
    if float(cin.get("dust", 0) or 0) > 0:
        fx_dust(arr, float(cin["dust"]) / 100.0)
    ratio = LETTERBOX.get(cin.get("letterbox", "None"))
    if ratio:
        fx_letterbox(arr, ratio)


# ====================================================================
#  COMPOSITION
# ====================================================================
def make_light():
    phi = random.uniform(math.pi * 0.9, math.pi * 1.6)    # light from the upper-left-ish
    return _norm3(math.cos(phi) * 0.65, math.sin(phi) * 0.65, 0.75)


def make_background(palette):
    cols = random.sample(palette, k=min(len(palette), random.randint(2, 3)))
    f = random.uniform(0.18, 0.5) if random.random() < 0.6 else random.uniform(0.75, 1.0)
    img = Image.fromarray(np.clip(gradient_array(W, H, [tuple(c * f for c in col) for col in cols]), 0, 255).astype(np.uint8))
    # soft colored clouds for atmosphere
    sw, sh = max(8, W // 8), max(8, H // 8)
    glow = img.resize((sw, sh), BICUBIC)
    gd = ImageDraw.Draw(glow)
    for _ in range(random.randint(3, 7)):
        k = min(1.0, f * random.uniform(1.0, 2.0))
        col = tuple(int(c * k) for c in random.choice(palette))
        r = random.uniform(0.08, 0.3) * max(sw, sh)
        cx, cy = random.uniform(0, sw), random.uniform(0, sh)
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    glow = glow.filter(ImageFilter.GaussianBlur(max(sw, sh) * 0.07)).resize((W, H), BICUBIC)
    return Image.blend(img, glow, random.uniform(0.35, 0.7))


def render_shape_tile(key, T):
    ss = 2 if T <= 1200 else 1
    N = T * ss
    im = Image.new("L", (N, N), 0)
    bevel = SHAPES[key][2](ImageDraw.Draw(im), N)
    if bevel is None:
        bevel = N * 0.09
    if ss > 1:
        im = im.resize((T, T), LANCZOS)
    im = im.rotate(random.uniform(0, 360), resample=BICUBIC)
    return im, bevel / ss


def draw_shape(canvas, spec, textures, palette, light, shadows):
    T = spec["T"]
    mask_img, bevel = render_shape_tile(spec["key"], T)
    mat = random.choice(textures)
    rgb, alpha = make_material(mat, mask_img, bevel, palette, light)
    opacity = 1.0 if mat == "neon" else random.uniform(0.72, 1.0)
    alpha = alpha * opacity
    x, y = int(spec["cx"] - T / 2), int(spec["cy"] - T / 2)

    if shadows and mat not in ("neon", "glass"):
        sh = mask_img.filter(ImageFilter.GaussianBlur(T * 0.02 + 1)).point(lambda p: int(p * 0.38 * opacity))
        dx, dy = -light[0] / (math.hypot(light[0], light[1]) + 1e-9) * T * 0.05, \
                 -light[1] / (math.hypot(light[0], light[1]) + 1e-9) * T * 0.05
        canvas.paste(Image.new("RGB", (T, T), (0, 0, 0)), (int(x + dx), int(y + dy)), sh)

    tile = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    a = Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8), "L")
    canvas.paste(tile, (x, y), a)


def generate(settings, progress=None):
    global W, H, SCALE
    W, H = settings.get("resolution", RESOLUTION)
    SCALE = max(W, H)
    palette = settings["palette"]
    shapes = settings["shapes"] or list(SHAPES)
    textures = settings["textures"] or ["flat"]
    count = settings["count"]
    light = make_light()
    canvas = make_background(palette)

    specs = []
    for _ in range(count):
        diameter = SCALE * settings["size"] * (0.05 + 0.32 * random.random() ** 1.7)
        T = int(min(SCALE * 0.9, max(32, diameter / 0.8)))
        specs.append(dict(key=random.choice(shapes), T=T, size=T * 0.8,
                          cx=random.uniform(-0.03, 1.03) * W, cy=random.uniform(-0.03, 1.03) * H))
    specs.sort(key=lambda s: -s["size"] * random.uniform(0.7, 1.3))   # big ones behind small ones

    for i, spec in enumerate(specs):
        draw_shape(canvas, spec, textures, palette, light, settings["shadows"])
        if progress:
            progress(i + 1, count)

    arr = np.asarray(canvas, np.float32).copy()
    if settings["sparkles"] and settings["sparkle_amount"] > 0:
        add_sparkles(arr, specs, settings["sparkle_amount"], palette)
    cin = settings.get("cinema") or {}
    if cin:
        if progress:
            progress("adding cinematic effects...", 0)
        cinema_pre(arr, cin, palette)
    if settings["vignette"]:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41
        arr *= (1 - 0.5 * r ** 2)[..., None]
    if settings["grain"]:
        arr += np.random.normal(0, random.uniform(3, 8), (H, W, 1))
    if cin:
        cinema_post(arr, cin)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def random_settings():
    shapes = random.sample(list(SHAPES), random.randint(6, len(SHAPES)))
    textures = random.sample(list(TEXTURES), random.randint(3, len(TEXTURES)))
    return dict(shapes=shapes, textures=textures, count=random.randint(25, 110),
                size=random.uniform(0.7, 1.4), palette=random_palette(),
                sparkles=random.random() < 0.7, sparkle_amount=random.randint(8, 60),
                shadows=random.random() < 0.7, grain=random.random() < 0.5,
                vignette=random.random() < 0.5, cinema=random_cinema())


def create_and_save(settings=None, progress=None, resolution=None, keep_palette=None, keep_cinema=None):
    """Generate one image (random settings if None) and save it next to this script."""
    seed = random.SystemRandom().randrange(10 ** 9)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    settings = dict(settings) if settings is not None else random_settings()
    if keep_palette:                      # random mode, but with the user's own colors
        settings["palette"] = list(keep_palette)
    if keep_cinema is not None:           # random mode, but with the user's own cinematic effects
        settings["cinema"] = dict(keep_cinema)
    settings["resolution"] = tuple(resolution or RESOLUTION)
    img = generate(settings, progress)
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, f"abstract_gui_{time.strftime('%Y%m%d_%H%M%S')}_{seed}.png")
    img.save(path, "PNG")
    return img, path


# ====================================================================
#  GUI
# ====================================================================
# ====================================================================
#  THEMES  (Dark / Light, switchable with the button in the bottom bar)
# ====================================================================
THEMES = {
    "Dark":  dict(bg="#1c1c1f", panel="#26262b", field="#2d2d33", fg="#e8e8ec", dim="#8a8a94",
                  border="#3d3d46", hover="#34343c", accent="#5b8def", accent_fg="#ffffff"),
    "Light": dict(bg="#f0f0f3", panel="#e2e2e8", field="#ffffff", fg="#1c1c1f", dim="#6e6e78",
                  border="#bdbdc6", hover="#d6d6de", accent="#3a74e0", accent_fg="#ffffff"),
}
COLORS = dict(THEMES["Dark"])   # the active theme (updated in place by apply_theme)
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abstract_gui_settings.json")


def load_theme():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            name = json.load(f).get("theme")
        return name if name in THEMES else "Dark"
    except (OSError, ValueError, AttributeError):
        return "Dark"


def save_theme(name):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({"theme": name}, f)
    except OSError:
        pass


def apply_theme(root, name):
    COLORS.clear()
    COLORS.update(THEMES[name])
    c = COLORS
    style = ttk.Style(root)
    try:
        style.theme_use("clam")      # the only built-in theme that colors everything properly
    except tk.TclError:
        pass
    root.configure(bg=c["bg"])
    style.configure(".", background=c["bg"], foreground=c["fg"], fieldbackground=c["field"],
                    bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                    troughcolor=c["field"], focuscolor=c["bg"], insertcolor=c["fg"],
                    selectbackground=c["accent"], selectforeground=c["accent_fg"])
    # clam paints DISABLED widgets light gray (#d9d9d9) by default; override that for every widget
    # so the "ignore custom settings" mode greys things out instead of breaking the theme.
    style.map(".", background=[("disabled", c["bg"]), ("active", c["bg"])],
              foreground=[("disabled", c["dim"])],
              selectbackground=[("!focus", c["accent"])],
              selectforeground=[("!focus", c["accent_fg"])])
    style.configure("TFrame", background=c["bg"])
    style.configure("TLabel", background=c["bg"], foreground=c["fg"])
    style.configure("Hint.TLabel", background=c["bg"], foreground=c["dim"])
    style.map("Hint.TLabel", foreground=[("disabled", c["dim"])])
    style.configure("TLabelframe", background=c["bg"], bordercolor=c["border"])
    style.configure("TLabelframe.Label", background=c["bg"], foreground=c["fg"])
    style.map("TLabelframe.Label", foreground=[("disabled", c["dim"])])
    style.configure("TButton", background=c["panel"], foreground=c["fg"], bordercolor=c["border"], padding=4)
    style.map("TButton",
              background=[("disabled", c["bg"]), ("pressed", c["accent"]), ("active", c["hover"])],
              foreground=[("disabled", c["dim"]), ("pressed", c["accent_fg"])])
    style.configure("TCheckbutton", background=c["bg"], foreground=c["fg"],
                    indicatorbackground=c["field"], indicatorforeground=c["fg"],
                    indicatorcolor=c["field"], upperbordercolor=c["border"], lowerbordercolor=c["border"])
    style.map("TCheckbutton",
              background=[("disabled", c["bg"]), ("active", c["bg"])],
              foreground=[("disabled", c["dim"])],
              indicatorbackground=[("disabled", c["panel"]), ("selected", c["accent"])],
              indicatorforeground=[("disabled", c["dim"]), ("selected", c["accent_fg"])],
              indicatorcolor=[("disabled", c["panel"]), ("selected", c["accent"])])
    for name_ in ("TCombobox", "TSpinbox"):
        style.configure(name_, fieldbackground=c["field"], background=c["panel"], foreground=c["fg"],
                        arrowcolor=c["fg"], bordercolor=c["border"], selectbackground=c["field"],
                        selectforeground=c["fg"], insertcolor=c["fg"])
        style.map(name_,
                  fieldbackground=[("disabled", c["panel"]), ("readonly", c["field"])],
                  foreground=[("disabled", c["dim"])],
                  background=[("disabled", c["panel"]), ("active", c["hover"])],
                  arrowcolor=[("disabled", c["dim"])],
                  selectbackground=[("readonly", c["field"])],
                  selectforeground=[("readonly", c["fg"])])
    # drop-down list of the comboboxes (plain tk widget, created lazily)
    root.option_add("*TCombobox*Listbox.background", c["field"])
    root.option_add("*TCombobox*Listbox.foreground", c["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", c["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", c["accent_fg"])
    try:  # dark / light title bar on Windows 10/11
        import ctypes
        root.update()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        for attr in (20, 19):
            val = ctypes.c_int(1 if name == "Dark" else 0)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val)) == 0:
                break
    except Exception:
        pass


PREVIEW_W, PREVIEW_H = 560, 420
RES_MIN, RES_MAX = 64, 10000
RES_PRESETS = {
    "1280 x 720  (HD)": (1280, 720),
    "1920 x 1080  (Full HD)": (1920, 1080),
    "2560 x 1440  (QHD)": (2560, 1440),
    "3840 x 2160  (4K)": (3840, 2160),
    "1080 x 1080  (Square)": (1080, 1080),
    "1080 x 1920  (Phone / Portrait)": (1080, 1920),
    "3000 x 2000  (3:2 Print)": (3000, 2000),
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Abstract Image Creator")
        self.scales = []
        self.theme_name = load_theme()
        apply_theme(root, self.theme_name)
        self.q = queue.Queue()
        self.busy = False
        self.preview_photo = None

        self.shape_vars = {k: tk.BooleanVar(value=True) for k in SHAPES}
        self.tex_vars = {k: tk.BooleanVar(value=True) for k in TEXTURES}
        self.count_var = tk.IntVar(value=45)
        self.size_var = tk.IntVar(value=100)
        self.sparkle_var = tk.BooleanVar(value=True)
        self.sparkle_amt = tk.IntVar(value=25)
        self.shadow_var = tk.BooleanVar(value=True)
        self.grain_var = tk.BooleanVar(value=False)
        self.vignette_var = tk.BooleanVar(value=True)
        self.cin_vars = {k: tk.IntVar(value=0) for k, _ in CINE_FX}
        self.cin_vars["grade_strength"] = tk.IntVar(value=0)
        self.grade_var = tk.StringVar(value="None")
        self.letter_var = tk.StringVar(value="None")
        self.random_var = tk.BooleanVar(value=False)
        self.keep_palette_var = tk.BooleanVar(value=False)
        self.keep_cinema_var = tk.BooleanVar(value=False)
        self.batch_var = tk.IntVar(value=1)
        self.status = tk.StringVar(value="Ready.")
        self.palette = list(PRESETS["Sunset"])
        self.custom_palettes = load_custom_palettes()
        self.w_var = tk.StringVar(value=str(RESOLUTION[0]))
        self.h_var = tk.StringVar(value=str(RESOLUTION[1]))

        self.build()
        self.poll()

    # ---------- layout ----------
    def build(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.settings = ttk.Frame(main)
        self.settings.grid(row=0, column=0, sticky="nw")
        left = ttk.Frame(self.settings)
        left.grid(row=0, column=0, sticky="n", padx=(0, 8))
        mid = ttk.Frame(self.settings)
        mid.grid(row=0, column=1, sticky="n")

        # --- shapes ---
        bar = ttk.Frame(left)
        bar.pack(fill="x")
        ttk.Label(bar, text="Shapes", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Button(bar, text="None", width=6, command=lambda: self.set_all(self.shape_vars, False)).pack(side="right")
        ttk.Button(bar, text="All", width=6, command=lambda: self.set_all(self.shape_vars, True)).pack(side="right", padx=2)
        for group in ("Basic", "Stars", "Unique"):
            box = ttk.LabelFrame(left, text=group, padding=4)
            box.pack(fill="x", pady=3)
            items = [(k, v[0]) for k, v in SHAPES.items() if v[1] == group]
            cols = 2 if group != "Stars" else 3
            for i, (k, label) in enumerate(items):
                ttk.Checkbutton(box, text=label, variable=self.shape_vars[k]).grid(
                    row=i // cols, column=i % cols, sticky="w", padx=4)

        # --- output resolution (stays active even in random mode) ---
        obox = ttk.LabelFrame(left, text="Output resolution", padding=4)
        obox.pack(fill="x", pady=3)
        self.output_box = obox
        ttk.Label(obox, text="Preset:").grid(row=0, column=0, sticky="w")
        self.res_preset = ttk.Combobox(obox, values=list(RES_PRESETS) + ["Custom"], state="readonly", width=27)
        self.res_preset.grid(row=0, column=1, columnspan=3, sticky="w", padx=4, pady=2)
        self.res_preset.bind("<<ComboboxSelected>>", self.load_res_preset)
        ttk.Label(obox, text="Width:").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(obox, from_=RES_MIN, to=RES_MAX, increment=10, width=7,
                    textvariable=self.w_var).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(obox, text="Height:").grid(row=1, column=2, sticky="e")
        ttk.Spinbox(obox, from_=RES_MIN, to=RES_MAX, increment=10, width=7,
                    textvariable=self.h_var).grid(row=1, column=3, sticky="w", padx=4)
        ttk.Label(obox, text="Bigger sizes take longer and use more memory.",
                  style="Hint.TLabel").grid(row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self.w_var.trace_add("write", self.sync_res_preset)
        self.h_var.trace_add("write", self.sync_res_preset)
        self.sync_res_preset()

        # --- textures ---
        tbox = ttk.LabelFrame(mid, text="Textures", padding=4)
        tbox.pack(fill="x")
        for i, (k, label) in enumerate(TEXTURES.items()):
            ttk.Checkbutton(tbox, text=label, variable=self.tex_vars[k]).grid(
                row=i // 2, column=i % 2, sticky="w", padx=4)
        tb = ttk.Frame(tbox)
        tb.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(tb, text="All", width=6, command=lambda: self.set_all(self.tex_vars, True)).pack(side="left")
        ttk.Button(tb, text="None", width=6, command=lambda: self.set_all(self.tex_vars, False)).pack(side="left", padx=2)

        # --- sliders ---
        sbox = ttk.LabelFrame(mid, text="Amount", padding=4)
        sbox.pack(fill="x", pady=4)
        self.slider(sbox, "Total shapes", self.count_var, 1, 250)
        self.slider(sbox, "Shape size (%)", self.size_var, 20, 180)

        # --- palette ---
        pbox = self.pbox = ttk.LabelFrame(mid, text="Color palette  (click a swatch to change it)", padding=4)
        pbox.pack(fill="x")
        top = ttk.Frame(pbox)
        top.pack(fill="x")
        ttk.Label(top, text="Preset:").pack(side="left")
        self.preset_box = ttk.Combobox(top, values=self.palette_names(), state="readonly", width=20)
        self.preset_box.set("Sunset")
        self.preset_box.pack(side="left", padx=4)
        self.preset_box.bind("<<ComboboxSelected>>", self.load_preset)
        self.swatch_frame = tk.Frame(pbox, bg=COLORS["bg"])
        self.swatch_frame.pack(fill="x", pady=4)
        pb = ttk.Frame(pbox)
        pb.pack(fill="x")
        ttk.Button(pb, text="+ Color", width=8, command=self.add_color).pack(side="left")
        ttk.Button(pb, text="- Remove", width=9, command=self.remove_color).pack(side="left", padx=2)
        ttk.Button(pb, text="Shuffle", width=8, command=self.shuffle_palette).pack(side="left")
        pb2 = ttk.Frame(pbox)
        pb2.pack(fill="x", pady=(4, 0))
        ttk.Button(pb2, text="Save palette...", width=14, command=self.save_palette).pack(side="left")
        ttk.Button(pb2, text="Delete saved", width=12, command=self.delete_palette).pack(side="left", padx=2)
        self.refresh_swatches()

        # --- effects ---
        ebox = ttk.LabelFrame(mid, text="Effects", padding=4)
        ebox.pack(fill="x", pady=4)
        ttk.Checkbutton(ebox, text="Sparkles", variable=self.sparkle_var).grid(row=0, column=0, sticky="w", padx=4)
        ttk.Checkbutton(ebox, text="Drop shadows", variable=self.shadow_var).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Checkbutton(ebox, text="Film grain", variable=self.grain_var).grid(row=1, column=0, sticky="w", padx=4)
        ttk.Checkbutton(ebox, text="Vignette", variable=self.vignette_var).grid(row=1, column=1, sticky="w", padx=4)
        sf = ttk.Frame(ebox)
        sf.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.slider(sf, "Sparkle amount", self.sparkle_amt, 0, 120)

        # --- cinematic effects (third column) ---
        right = ttk.Frame(self.settings)
        right.grid(row=0, column=2, sticky="n", padx=(8, 0))
        self.cin_box = right
        cbox = ttk.LabelFrame(right, text="Cinematic effects  (0 = off)", padding=4)
        cbox.pack(fill="x")
        ctop = ttk.Frame(cbox)
        ctop.pack(fill="x", pady=(0, 4))
        ttk.Label(ctop, text="Look:").pack(side="left")
        self.look_box = ttk.Combobox(ctop, values=list(CINE_LOOKS), state="readonly", width=19)
        self.look_box.set("Off")
        self.look_box.pack(side="left", padx=4)
        self.look_box.bind("<<ComboboxSelected>>", self.load_look)
        for key, label in CINE_FX:
            self.slider(cbox, label, self.cin_vars[key], 0, 100, length=130, label_w=19)
        gbox = ttk.LabelFrame(right, text="Color grade & framing", padding=4)
        gbox.pack(fill="x", pady=4)
        grow = ttk.Frame(gbox)
        grow.pack(fill="x")
        ttk.Label(grow, text="Grade:").pack(side="left")
        ttk.Combobox(grow, values=list(GRADES), textvariable=self.grade_var, state="readonly",
                     width=16).pack(side="left", padx=4)
        self.slider(gbox, "Grade strength", self.cin_vars["grade_strength"], 0, 100, length=130, label_w=19)
        lrow = ttk.Frame(gbox)
        lrow.pack(fill="x", pady=(2, 0))
        ttk.Label(lrow, text="Letterbox:").pack(side="left")
        ttk.Combobox(lrow, values=list(LETTERBOX), textvariable=self.letter_var, state="readonly",
                     width=14).pack(side="left", padx=4)
        ttk.Label(right, text="Effects add a few seconds per image\n(more at 4K and above).",
                  style="Hint.TLabel").pack(anchor="w")

        # --- preview ---
        pv = self.pv = ttk.LabelFrame(main, text="Preview", padding=6)
        pv.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        pv.columnconfigure(0, weight=1)
        pv.rowconfigure(0, weight=1)
        self.preview = ttk.Label(pv, text="Your image will appear here.", anchor="center")
        self.preview.grid(sticky="nsew")

        # --- bottom bar ---
        bottom = ttk.Frame(main, padding=(0, 8, 0, 0))
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        bottom.columnconfigure(3, weight=1)
        ttk.Checkbutton(bottom, text="Ignore all custom settings and use random ones",
                        variable=self.random_var, command=self.toggle_random).grid(row=0, column=0, sticky="w")
        self.keep_chk = ttk.Checkbutton(bottom, text="Keep my color palette (random settings use it)",
                                        variable=self.keep_palette_var, command=self.toggle_random)
        self.keep_chk.grid(row=1, column=0, sticky="w", padx=(22, 0))
        self.keep_cin_chk = ttk.Checkbutton(bottom, text="Keep my cinematic effects (random settings use them)",
                                            variable=self.keep_cinema_var, command=self.toggle_random)
        self.keep_cin_chk.grid(row=2, column=0, sticky="w", padx=(22, 0))
        ttk.Label(bottom, text="Images:").grid(row=0, column=1, padx=(16, 2))
        ttk.Spinbox(bottom, from_=1, to=200, width=5, textvariable=self.batch_var).grid(row=0, column=2)
        ttk.Label(bottom, textvariable=self.status, anchor="w").grid(row=0, column=3, sticky="ew", padx=12)
        self.gen_btn = ttk.Button(bottom, text="Generate", command=self.on_generate)
        self.gen_btn.grid(row=0, column=4)
        self.theme_btn = ttk.Button(bottom, width=11, command=self.toggle_theme,
                                    text="Light mode" if self.theme_name == "Dark" else "Dark mode")
        self.theme_btn.grid(row=0, column=5, padx=(8, 0))
        self.toggle_random()

    def slider(self, parent, label, var, lo, hi, length=210, label_w=15):
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=label, width=label_w).pack(side="left")
        sc = tk.Scale(row, from_=lo, to=hi, orient="horizontal", variable=var, length=length,
                      bg=COLORS["bg"], fg=COLORS["fg"], troughcolor=COLORS["field"],
                      activebackground=COLORS["accent"], highlightthickness=0, bd=0, relief="flat")
        sc.pack(side="left", fill="x", expand=True)
        self.scales.append(sc)

    def toggle_theme(self):
        self.set_theme("Light" if self.theme_name == "Dark" else "Dark")

    def set_theme(self, name):
        self.theme_name = name
        apply_theme(self.root, name)
        c = COLORS
        for sc in self.scales:
            sc.configure(bg=c["bg"], fg=c["fg"], troughcolor=c["field"], activebackground=c["accent"])
        self.swatch_frame.configure(bg=c["bg"])
        self.restyle_popdowns(self.root)
        self.theme_btn.configure(text="Light mode" if name == "Dark" else "Dark mode")
        save_theme(name)

    def restyle_popdowns(self, widget):
        """Recolor combobox drop-down lists that already exist (new ones pick up the theme by themselves)."""
        if isinstance(widget, ttk.Combobox):
            try:
                pop = self.root.tk.call("ttk::combobox::PopdownWindow", str(widget))
                self.root.tk.call(f"{pop}.f.l", "configure", "-background", COLORS["field"],
                                  "-foreground", COLORS["fg"], "-selectbackground", COLORS["accent"],
                                  "-selectforeground", COLORS["accent_fg"])
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            self.restyle_popdowns(child)

    def load_look(self, _event=None):
        look = CINE_LOOKS.get(self.look_box.get())
        if look is None:
            return
        for k, v in self.cin_vars.items():
            v.set(look.get(k, 0))
        self.grade_var.set(look.get("grade", "None"))
        self.letter_var.set(look.get("letterbox", "None"))

    def get_cinema(self):
        cin = {k: v.get() for k, v in self.cin_vars.items()}
        cin["grade"] = self.grade_var.get()
        cin["letterbox"] = self.letter_var.get()
        return cin

    # ---------- palette ----------
    def refresh_swatches(self):
        for w in self.swatch_frame.winfo_children():
            w.destroy()
        for i, hx in enumerate(self.palette):
            lbl = tk.Label(self.swatch_frame, bg=hx, width=4, height=2, relief="raised", bd=2, cursor="hand2")
            lbl.grid(row=i // 5, column=i % 5, padx=2, pady=2)
            lbl.bind("<Button-1>", lambda e, i=i: self.pick_color(i))

    def pick_color(self, i):
        if self.random_var.get() and not self.keep_palette_var.get():
            return
        res = colorchooser.askcolor(color=self.palette[i], parent=self.root, title="Pick a color")
        if res and res[1]:
            self.palette[i] = res[1]
            self.refresh_swatches()

    def palette_names(self):
        return list(PRESETS) + [STAR + n for n in sorted(self.custom_palettes)]

    def load_preset(self, _event=None):
        name = self.preset_box.get()
        if name.startswith(STAR):
            self.palette = list(self.custom_palettes[name[len(STAR):]])
        else:
            self.palette = list(PRESETS[name])
        self.refresh_swatches()

    def save_palette(self):
        name = simpledialog.askstring("Save palette", "Name for this palette:", parent=self.root)
        if name is None:
            return
        name = name.strip().lstrip(STAR.strip()).strip()
        if not name:
            return
        if name in PRESETS:
            messagebox.showwarning("Name taken", f'"{name}" is a built-in palette. Please pick another name.')
            return
        if name in self.custom_palettes and not messagebox.askyesno(
                "Replace palette?", f'You already saved a palette called "{name}". Replace it?'):
            return
        self.custom_palettes[name] = list(self.palette)
        try:
            write_custom_palettes(self.custom_palettes)
        except OSError as e:
            messagebox.showerror("Could not save", f"Failed to write {PALETTE_FILE}:\n{e}")
            return
        self.preset_box.configure(values=self.palette_names())
        self.preset_box.set(STAR + name)
        self.status.set(f'Saved palette "{name}".')

    def delete_palette(self):
        name = self.preset_box.get()
        if not name.startswith(STAR):
            messagebox.showinfo("Delete saved palette", "Select one of your saved palettes (marked with a star) first.\n"
                                "Built-in presets can't be deleted.")
            return
        name = name[len(STAR):]
        if not messagebox.askyesno("Delete palette", f'Delete saved palette "{name}"?'):
            return
        self.custom_palettes.pop(name, None)
        try:
            write_custom_palettes(self.custom_palettes)
        except OSError as e:
            messagebox.showerror("Could not save", f"Failed to write {PALETTE_FILE}:\n{e}")
        self.preset_box.configure(values=self.palette_names())
        self.preset_box.set("")
        self.status.set(f'Deleted palette "{name}".')

    # ---------- resolution ----------
    def load_res_preset(self, _event=None):
        res = RES_PRESETS.get(self.res_preset.get())
        if res:
            self.w_var.set(str(res[0]))
            self.h_var.set(str(res[1]))

    def sync_res_preset(self, *_):
        try:
            cur = (int(self.w_var.get()), int(self.h_var.get()))
        except ValueError:
            cur = None
        name = next((k for k, v in RES_PRESETS.items() if v == cur), "Custom")
        self.res_preset.set(name)

    def get_resolution(self):
        try:
            w, h = int(self.w_var.get()), int(self.h_var.get())
            if not (RES_MIN <= w <= RES_MAX and RES_MIN <= h <= RES_MAX):
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid resolution",
                                   f"Width and height must be whole numbers between {RES_MIN} and {RES_MAX}.")
            return None
        return w, h

    def add_color(self):
        if len(self.palette) < 10:
            self.palette.append(rgb_to_hex(random_palette(1)[0]))
            self.refresh_swatches()

    def remove_color(self):
        if len(self.palette) > 2:
            self.palette.pop()
            self.refresh_swatches()

    def shuffle_palette(self):
        self.palette = [rgb_to_hex(c) for c in random_palette()]
        self.refresh_swatches()

    # ---------- helpers ----------
    @staticmethod
    def set_all(vars_, value):
        for v in vars_.values():
            v.set(value)

    def toggle_random(self):
        self.set_enabled(self.settings, not self.random_var.get())
        self.set_enabled(self.output_box, True)   # resolution still applies in random mode
        rnd = self.random_var.get()
        self.keep_chk.state(["!disabled"] if rnd else ["disabled"])
        self.keep_cin_chk.state(["!disabled"] if rnd else ["disabled"])
        if rnd and self.keep_palette_var.get():   # palette stays editable and is used
            self.set_enabled(self.pbox, True)
        if rnd and self.keep_cinema_var.get():    # cinematic effects stay editable and are used
            self.set_enabled(self.cin_box, True)

    def set_enabled(self, widget, enabled):
        for child in widget.winfo_children():
            self.set_enabled(child, enabled)
        try:
            if isinstance(widget, ttk.Widget):
                widget.state(["!disabled"] if enabled else ["disabled"])
            elif isinstance(widget, (tk.Scale, tk.Label)):
                widget.configure(state="normal" if enabled else "disabled")
        except tk.TclError:
            pass

    def collect_settings(self):
        shapes = [k for k, v in self.shape_vars.items() if v.get()]
        textures = [k for k, v in self.tex_vars.items() if v.get()]
        if not shapes:
            messagebox.showwarning("No shapes", "Tick at least one shape.")
            return None
        if not textures:
            messagebox.showwarning("No textures", "Tick at least one texture.")
            return None
        return dict(shapes=shapes, textures=textures, count=self.count_var.get(),
                    size=self.size_var.get() / 100.0, palette=[hex_to_rgb(h) for h in self.palette],
                    sparkles=self.sparkle_var.get(), sparkle_amount=self.sparkle_amt.get(),
                    shadows=self.shadow_var.get(), grain=self.grain_var.get(),
                    vignette=self.vignette_var.get(), cinema=self.get_cinema())

    # ---------- generation ----------
    def on_generate(self):
        if self.busy:
            return
        settings = None if self.random_var.get() else self.collect_settings()
        if settings is None and not self.random_var.get():
            return
        res = self.get_resolution()
        if res is None:
            return
        keep = None
        if self.random_var.get() and self.keep_palette_var.get():
            keep = [hex_to_rgb(h) for h in self.palette]
        keep_cin = self.get_cinema() if (self.random_var.get() and self.keep_cinema_var.get()) else None
        try:
            n = max(1, int(self.batch_var.get()))
        except (tk.TclError, ValueError):
            n = 1
        self.busy = True
        self.gen_btn.state(["disabled"])
        self.status.set("Starting...")
        threading.Thread(target=self.worker, args=(settings, n, res, keep, keep_cin), daemon=True).start()

    def worker(self, settings, n, res, keep=None, keep_cin=None):
        try:
            for i in range(1, n + 1):
                img, path = create_and_save(settings, lambda a, b, i=i: self.q.put(("progress", i, n, a, b)), res, keep, keep_cin)
                thumb = img.copy()
                thumb.thumbnail((PREVIEW_W, PREVIEW_H), LANCZOS)
                self.q.put(("saved", i, n, path, thumb, img.size))
        except Exception as e:  # report instead of dying silently
            import traceback
            traceback.print_exc()
            self.q.put(("error", str(e)))
        finally:
            self.q.put(("finished",))

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "progress":
                    _, i, n, a, b = msg
                    if isinstance(a, str):
                        self.status.set(f"Image {i}/{n}: {a}")
                    else:
                        self.status.set(f"Image {i}/{n}: drawing shape {a}/{b}...")
                elif msg[0] == "saved":
                    _, i, n, path, thumb, size = msg
                    self.pv.configure(text=f"Preview   (last image: {size[0]} x {size[1]} px)")
                    self.preview_photo = ImageTk.PhotoImage(thumb)
                    self.preview.configure(image=self.preview_photo, text="")
                    self.status.set(f"Saved {i}/{n}: {os.path.basename(path)}")
                elif msg[0] == "error":
                    messagebox.showerror("Error", msg[1])
                elif msg[0] == "finished":
                    self.busy = False
                    self.gen_btn.state(["!disabled"])
        except queue.Empty:
            pass
        self.root.after(100, self.poll)


def main():
    if tk is None:
        print("tkinter / Pillow ImageTk not found.\n"
              "  Windows/macOS: reinstall Python from python.org (tkinter is included)\n"
              "  Linux:         sudo apt install python3-tk")
        return
    try:  # crisp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
