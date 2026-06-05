"""
One-off helper: remove the white background from FootBot_Logo.png via edge
flood-fill (so interior whites stay solid), then export transparent assets:
  static/footbot-logo.png  - full wordmark (ball + "FootBot"), auto-cropped
  static/footbot-mark.png  - ball emblem only (square), for favicon / small icon
Run once locally; not part of the app runtime. Requires Pillow + numpy + scipy.
"""
import os
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

SRC = r"C:\Users\tomer\OneDrive\Desktop\לימודים\מערכות מידע שנה 3 - סמסטר ב\ML\FootBot_Logo.png"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(OUT_DIR, exist_ok=True)

im = Image.open(SRC).convert("RGBA")
arr = np.array(im).astype(np.int16)
rgb = arr[..., :3]
mx = rgb.max(axis=2)
mn = rgb.min(axis=2)

# Near-white, low-saturation pixels are background candidates.
near_white = (mn > 222) & ((mx - mn) < 20)

# Flood-fill from the border: only background connected to the edge is removed,
# so the white inside the ball / chat bubble is preserved.
lbl, _ = ndimage.label(near_white)
border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
border.discard(0)
bg = np.isin(lbl, list(border))

# Erode foreground by 1px to kill the white anti-aliasing ring (avoids a white
# halo on dark backgrounds), then a tiny blur for smooth edges.
alpha = np.where(bg, 0, 255).astype(np.uint8)
fg = alpha > 0
fg = ndimage.binary_erosion(fg, iterations=1)
alpha = np.where(fg, 255, 0).astype(np.uint8)
alpha_img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.6))
alpha = np.array(alpha_img)

out = arr.copy()
out[..., 3] = alpha
out = out.astype(np.uint8)
rgba = Image.fromarray(out)

# --- Full logo: crop to content bounding box ---
bbox = rgba.getbbox()
full = rgba.crop(bbox)
full.save(os.path.join(OUT_DIR, "footbot-logo.png"))

# --- Emblem: split off the ball by finding the transparent gap above the text ---
a = np.array(full)[..., 3]
rows_solid = (a > 16).sum(axis=1)
h = a.shape[0]
# Search the lower 35-75% band for the row with the fewest solid pixels (the gap).
band = range(int(h * 0.55), int(h * 0.80))
gap = min(band, key=lambda y: rows_solid[y])
emblem = full.crop((0, 0, full.size[0], gap))
# Re-crop emblem to its own bbox and pad to a square.
emblem = emblem.crop(emblem.getbbox())
ew, eh = emblem.size
side = max(ew, eh)
square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
square.paste(emblem, ((side - ew) // 2, (side - eh) // 2))
square.save(os.path.join(OUT_DIR, "footbot-mark.png"))

print("full logo :", full.size)
print("emblem    :", square.size, "(gap row", gap, "of", h, ")")
print("saved to  :", OUT_DIR)
