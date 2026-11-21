# AI & Expert Systems — Face Detection and Recognition (Ultralytics YOLO)

**Student:** Joseph Borketey Bortey · PUIT/24120005 · Level 400 IT, Pentecost University
**Course:** AI and Expert Systems · **Lecturer:** Mr. Harry Attieku-Boateng
**Assessment:** Individual Assessment (IA) — the Ultralytics face project

---

## 1. The project

A **custom object-detection model that finds my face in a photo or video and labels it with my name** (`Joseph_Bortey`).
It is trained with **Ultralytics YOLO** on the **Ultralytics Platform** and served through an **inference link** where any image can be uploaded and tested.

| Stage | What happens | Where |
|---|---|---|
| 1. Data collection | 500+ images of me: photos plus frames taken from my own videos, in many places, lighting conditions, angles and expressions | `face_project/prepare_dataset.py` |
| 2. Annotation | One **tight bounding box per image around my face, forehead to chin**, label `Joseph_Bortey`. Other people's faces are left unlabelled so the model learns they are *not* me | same script + manual review |
| 3. Negatives | A small share of "background" images (other people / no face) with empty label files, to reduce false detections | same script |
| 4. Split | 80 % train / 20 % validation, split **by source video/photo** so near-identical frames never appear in both sets | same script |
| 5. Training | YOLO object-detection model on the Ultralytics Platform | Ultralytics Platform |
| 6. Evaluation | Precision, recall, mAP50, mAP50-95, confusion matrix, PR curve | Ultralytics Platform |
| 7. Inference | Public inference link for testing new images | submitted on the course form |

### Annotation rules (from the lecturer's 1 Nov guidance)
- The box **closely encloses the whole face — forehead to chin, across its full width**.
- **No shoulders, doors, roofs or background** inside the box; not so small that it cuts the face.
- The **same label name** on every image.
- Photos taken against **different backgrounds**, so the model does not link my identity to one location.

### How the automatic annotation works
1. **Face detection** — OpenCV **YuNet** finds every face in each image.
2. **Face identification** — OpenCV **SFace** turns each face into a 128-number "face fingerprint" (embedding). It is compared with fingerprints from my reference selfies using **cosine similarity**. Faces above the threshold are me; others are ignored. Images with an *uncertain* face are skipped instead of guessed.
3. **Box tightening** — the detector box is adjusted to cover forehead-to-chin.
4. **Quality filters** — blurry frames (low Laplacian variance) and near-duplicate frames (perceptual hash) are dropped.
5. **Human review** — every box is checked on contact sheets before upload; wrong boxes are fixed or removed.

### Dataset format (YOLO)
```
dataset/
  data.yaml            # names: {0: Joseph_Bortey}
  images/train/*.jpg   images/val/*.jpg
  labels/train/*.txt   labels/val/*.txt   # "0 x_center y_center width height" (0-1, relative)
```

### Results
_To be filled in after training: dataset size, mAP50, precision, recall, inference link._

---

## 2. Practice notebooks (class work)

| Notebook | Topic |
|---|---|
| `notebooks/01_python_basics_day1.ipynb` | Python basics (Week 1 tutorial) |
| `scripts/generate_patient_dataset.py` | Synthetic patient dataset (5,000 records, sickle-cell focus) + 10 scatter plots |
| `notebooks/03_heart_disease_classification.ipynb` | Hospital heart-disease task — logistic regression, random forest, gradient boosting, confusion matrix, ROC/AUC, threshold tuning |
| `notebooks/04_iris_multiclass_classification.ipynb` | Iris task — multi-class confusion matrix, per-class precision/recall, one-vs-rest AUC |
| `notebooks/05_linear_regression_practice.ipynb` | Linear regression by hand and with scikit-learn, residuals, multiple regression |
| `notebooks/06_ollama_llama3.2_1b.ipynb` | Running Llama 3.2 1B locally with Ollama and calling it from Python |

## 3. Run it yourself
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
jupyter notebook
```
For notebook 06: install [Ollama](https://ollama.com), then `ollama pull llama3.2:1b`.

## 4. Tools
Python · scikit-learn · pandas · matplotlib/seaborn · OpenCV (YuNet, SFace) · Ultralytics YOLO & Ultralytics Platform · Ollama · VS Code · Google Colab · GitHub
