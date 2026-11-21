"""Build a YOLO face-recognition dataset for ONE person from photos + videos (low-RAM, two passes).

Pass 1  sample frames (more from longer videos), drop blurry / duplicate ones, detect every face (YuNet),
        compute a face "fingerprint" (SFace embedding) for each face; frames are written to disk at once.
Pass 2  identify the person: start from the reference selfies, add every face that matches them very
        confidently to a "gallery", then score every face against the gallery (robust to angle/lighting).
        Box ONLY that person's face (forehead to chin); frames with an uncertain face are skipped;
        frames with only other people / no face become a few unlabelled "background" images.
        Split train/val by SOURCE file so frames of one video never appear in both.

Usage:
  python prepare_dataset.py --src <folder> --refs ref1.jpg ref2.jpg --out dataset --name Joseph_Bortey --target 650
"""
import argparse, json, random, hashlib, shutil
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).parent
DET = HERE / "models" / "face_detection_yunet_2023mar.onnx"
REC = HERE / "models" / "face_recognition_sface_2021dec.onnx"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXT = {".mp4", ".mov", ".3gp", ".mkv", ".webm"}
SEED_MATCH = 0.50   # very confident match to the reference selfies -> goes into the gallery
MATCH = 0.42        # final: it's the person
UNSURE = 0.30       # between UNSURE and MATCH -> ambiguous, skip the frame (never guess)
MIN_FACE = 48       # px; smaller faces are too small to label well
MIN_SHARP = 15      # Laplacian variance; phone-video frames are softer than photos
MAX_SIDE = 1280


def resize(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def dhash(img, n=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n))
    return (g[:, 1:] > g[:, :-1]).flatten()


def unit(v):
    return v / np.linalg.norm(v)


class Faces:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(str(DET), "", (320, 320), 0.8, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(REC), "")

    def detect(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        return [] if faces is None else list(faces)

    def embed(self, img, face):
        return unit(self.rec.feature(self.rec.alignCrop(img, face)).flatten())


def tight_box(face, w, h):
    """YuNet box (brow-to-chin) -> forehead-to-chin, sides padded a little so cheeks are not cut."""
    x, y, bw, bh = face[:4]
    x0, x1 = max(0, x - 0.04 * bw), min(w, x + 1.04 * bw)
    y0, y1 = max(0, y - 0.12 * bh), min(h, y + 1.02 * bh)
    return float(x0), float(y0), float(x1), float(y1)


def video_frames(path):
    cap = cv2.VideoCapture(str(path))   # OpenCV applies the phone's rotation flag by default
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if n <= 0:
        return
    k = int(np.clip(n / fps * 1.5, 30, 360))        # ~1.5 frames per second, 30..360 per video
    for i in np.linspace(0, n - 1, min(n, k)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            yield int(i), f
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--name", default="Joseph_Bortey")
    ap.add_argument("--target", type=int, default=650)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--bg_share", type=float, default=0.08)
    a = ap.parse_args()
    random.seed(42)
    F = Faces()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    stage = out / "_stage"; stage.mkdir(parents=True)

    refs = []
    for r in a.refs:
        img = resize(cv2.imread(r)); fs = F.detect(img)
        assert fs, f"no face in reference {r}"
        refs.append(F.embed(img, max(fs, key=lambda f: f[2] * f[3])))
    refs = np.array(refs)

    files = sorted(p for s in a.src for p in Path(s).rglob("*") if p.suffix.lower() in IMG_EXT | VID_EXT)
    log = {"frames_read": 0, "blurry": 0, "duplicate": 0, "no_face": 0}
    recs = []            # one record per kept frame: path, size, source, tag, faces[(box, embedding)]
    seen = []

    def take(img, src, tag):
        log["frames_read"] += 1
        img = resize(img)
        if sharpness(img) < MIN_SHARP:
            log["blurry"] += 1; return
        h = dhash(img)
        if any((h != s).sum() <= 3 for s in seen[-80:]):
            log["duplicate"] += 1; return
        seen.append(h)
        H, W = img.shape[:2]
        faces = [(tight_box(f, W, H), F.embed(img, f)) for f in F.detect(img) if min(f[2], f[3]) >= MIN_FACE]
        if not faces:
            log["no_face"] += 1
            if random.random() > 0.15:            # keep only some face-free frames as background candidates
                return
        p = stage / f"{len(recs):06d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        recs.append({"path": p, "W": W, "H": H, "src": src, "tag": tag, "faces": faces})

    # ---- pass 1
    for p in files:
        if p.suffix.lower() in IMG_EXT:
            im = cv2.imread(str(p))
            if im is not None:
                take(im, p.name, "photo")
        else:
            for i, f in video_frames(p):
                take(f, p.name, f"f{i}")
        print(f"  pass1 {p.name}: kept frames so far {len(recs)}", flush=True)

    # ---- pass 2: gallery = refs + confident matches (spread over sources)
    all_emb = [(ri, fi, e) for ri, r in enumerate(recs) for fi, (_, e) in enumerate(r["faces"])]
    seed = [e for _, _, e in all_emb if float((refs @ e).max()) >= SEED_MATCH]
    gallery = np.vstack([refs] + ([np.array(seed)] if seed else []))
    print(f"gallery: {len(refs)} refs + {len(seed)} confident matches", flush=True)

    def score(e):
        s = gallery @ e
        top = np.sort(s)[-5:]                      # mean of the 5 best gallery matches: stable, not one lucky hit
        return float(max(top.mean(), (refs @ e).max()))

    pos, bg, amb, other = [], [], 0, 0
    for r in recs:
        sc = [(score(e), box) for box, e in r["faces"]]
        mine = [t for t in sc if t[0] >= MATCH]
        if any(UNSURE <= s < MATCH for s, _ in sc):
            amb += 1; continue
        if len(mine) > 1:                          # two faces both look like the person -> unsafe, skip
            amb += 1; continue
        if mine:
            pos.append((r, mine[0][1], round(mine[0][0], 3)))
        else:
            if r["faces"]:
                other += 1
            bg.append((r, None, 0))

    # balance: cap frames per source so one long video can't dominate, keep the most varied (spread in time)
    by_src = {}
    for t in pos:
        by_src.setdefault(t[0]["src"], []).append(t)
    cap = max(25, int(a.target / max(1, len(by_src)) * 2.5))
    chosen = []
    for v in by_src.values():
        idx = np.linspace(0, len(v) - 1, min(len(v), cap)).astype(int)
        chosen += [v[i] for i in sorted(set(idx))]
    bg = random.sample(bg, min(len(bg), int(len(chosen) * a.bg_share)))

    srcs = sorted(by_src)
    random.shuffle(srcs)
    # put ~20% of IMAGES (not sources) into val, whole sources at a time
    val_src, nval = set(), 0
    for s in srcs:
        if nval >= a.val * len(chosen):
            break
        val_src.add(s); nval += min(len(by_src[s]), cap)

    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    rows = []
    for r, box, s in chosen + bg:
        sp = "val" if r["src"] in val_src else "train"
        stem = f"{Path(r['src']).stem[:24]}_{r['tag']}_{hashlib.md5(str(r['path']).encode()).hexdigest()[:6]}"
        r["path"].replace(out / "images" / sp / f"{stem}.jpg")
        with open(out / "labels" / sp / f"{stem}.txt", "w") as fh:
            if box:
                x0, y0, x1, y1 = box; W, H = r["W"], r["H"]
                fh.write(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
        rows.append({"file": f"{sp}/{stem}.jpg", "source": r["src"], "label": bool(box), "score": s})
    shutil.rmtree(stage)
    (out / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\nnames:\n  0: {a.name}\n")
    (out / "manifest.json").write_text(json.dumps(rows, indent=1))
    per_src = {k: min(len(v), cap) for k, v in sorted(by_src.items(), key=lambda kv: -len(kv[1]))}
    tr = sum(r["file"].startswith("train") for r in rows)
    print(json.dumps({"labelled_images": len(chosen), "background": len(bg), "train": tr, "val": len(rows) - tr,
                      "sources_with_matches": len(by_src), "val_sources": sorted(val_src), "ambiguous_skipped": amb,
                      "other_people_only": other, **log, "per_source": per_src}, indent=1))


if __name__ == "__main__":
    main()
