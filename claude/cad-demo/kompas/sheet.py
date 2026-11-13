"""Лист «со всех сторон»: python sheet.py out.png scan.stl model.stl [tol]
8 видов; в каждой паре слева скан, справа модель. Красным — где поверхность дальше tol от другой
(на скане: в модели этого нет; на модели: лишнее, чего нет на скане)."""
import sys
import numpy as np, trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from PIL import ImageFont
FN = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14) if __import__("os").path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") else None

def dist_to(a, b, n=400000):
    P, _ = trimesh.sample.sample_surface(b, n)
    d, _ = cKDTree(P).query(a.triangles_center)
    return d

def shade(mesh, view, bad, size=440, up=(0, 0, 1), c=None, ext=None):
    w = np.asarray(view, float); w /= np.linalg.norm(w)
    u = np.cross(up, w)
    if np.linalg.norm(u) < 1e-6: u = np.cross((0, 1, 0), w)
    u /= np.linalg.norm(u); v = np.cross(w, u)
    P = mesh.vertices - c
    xy = np.c_[P @ u, P @ v]; depth = P @ w
    px = (xy / ext * 0.5 + 0.5) * size; px[:, 1] = size - px[:, 1]
    F = mesh.faces; n = mesh.face_normals; front = n @ w > 0
    light = np.array([0.35, 0.45, 0.82]); light /= np.linalg.norm(light)
    lw = u * light[0] + v * light[1] + w * light[2]
    lam = np.clip(n @ lw, 0, 1)
    img = Image.new("RGB", (size, size), (24, 30, 38)); d = ImageDraw.Draw(img)
    for f in np.argsort(depth[F].mean(1)):
        if not front[f]: continue
        s = 0.2 + 0.75 * lam[f]
        base = (0.95, 0.35, 0.3) if bad[f] else (0.62, 0.72, 0.80)
        d.polygon([tuple(px[i]) for i in F[f]], fill=tuple(int(255 * min(1, k * s)) for k in base))
    return img

if __name__ == "__main__":
    out, a, b = sys.argv[1:4]; tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    A = trimesh.load(a, force="mesh"); B = trimesh.load(b, force="mesh")
    ba = dist_to(A, B) > tol; bb = dist_to(B, A) > tol
    c = A.bounds.mean(0); ext = np.linalg.norm(A.extents) * 0.55
    views = [((0, 0.001, 1), "сверху"), ((0, 0.001, -1), "снизу"), ((1, -1.2, 1), "спереди-справа"), ((-1, 1.2, 1), "сзади-слева"),
             ((1, 0, 0.15), "справа"), ((-1, 0, 0.15), "слева"), ((0, -1, 0.15), "спереди"), ((0, 1, 0.15), "сзади")]
    S = 440; G = Image.new("RGB", (4 * S + 30, 4 * (S + 20)), (255, 255, 255)); d = ImageDraw.Draw(G)
    for i, (vw, name) in enumerate(views):
        r, k = i // 2, i % 2
        x = k * (2 * S + 20); y = r * (S + 20)
        G.paste(shade(A, vw, ba, S, c=c, ext=ext), (x, y + 18)); G.paste(shade(B, vw, bb, S, c=c, ext=ext), (x + S + 5, y + 18))
        d.text((x + 4, y + 1), name + ": скан", fill=(0, 0, 0), font=FN); d.text((x + S + 9, y + 1), "модель", fill=(0, 0, 0), font=FN)
    G.save(out)
    print("красного на скане %.1f%%, на модели %.1f%%" % (100 * np.sum(A.area_faces[ba]) / A.area, 100 * np.sum(B.area_faces[bb]) / B.area))
