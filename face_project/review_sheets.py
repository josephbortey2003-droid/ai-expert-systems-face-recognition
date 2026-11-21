"""Draw every YOLO box onto thumbnails and tile them into numbered contact sheets for checking.
Usage: python review_sheets.py dataset review   -> review/sheet_000.jpg ... (48 thumbs each)"""
import sys
from pathlib import Path
import cv2
import numpy as np

ds, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
TW, TH, COLS, PER = 200, 260, 8, 48
items = sorted((ds / "images").rglob("*.jpg"))
for s in range(0, len(items), PER):
    sheet = np.zeros((TH * ((min(PER, len(items) - s) + COLS - 1) // COLS), TW * COLS, 3), np.uint8)
    for k, p in enumerate(items[s:s + PER]):
        img = cv2.imread(str(p)); H, W = img.shape[:2]
        lab = ds / "labels" / p.parent.name / (p.stem + ".txt")
        for line in lab.read_text().split("\n") if lab.exists() else []:
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
                x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), max(2, W // 200))
        sc = min(TW / W, (TH - 18) / H); t = cv2.resize(img, (int(W * sc), int(H * sc)))
        r, c = divmod(k, COLS)
        sheet[r * TH:r * TH + t.shape[0], c * TW:c * TW + t.shape[1]] = t
        cv2.putText(sheet, f"{s + k}", (c * TW + 3, r * TH + TH - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(out / f"sheet_{s // PER:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(len(items), "images ->", out)
