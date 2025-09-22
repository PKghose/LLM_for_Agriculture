# zsl_classify_generate.py
import os, base64, json, csv
from pathlib import Path
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ========= CONFIG =========
# DATA_DIR = Path(r"C:\Users\partho.ghose\Desktop\leaf_data")  # <- uncomment & edit if you want
DATA_DIR = Path("leaf_data")     # expects: leaf_data/healthy, leaf_data/diseased
CLASSES = ["healthy", "diseased"]

MODEL = "qwen3:4b"           # or minicpmv:8b,  "llava:7b", qwen3:4b, gemma3:4b, deepseek-r1:8b
TEMPERATURE = 0

OUTPUT_CSV = "predictions_zsl.csv"
CLASS_REPORT_TXT = "classification_report_zsl.txt"
METRICS_JSON = "metrics_zsl.json"
CM_CSV = "confusion_zsl.csv"
CM_NORM_CSV = "confusion_zsl_normalized.csv"
CM_PNG = "confusion_zsl.png"

# Respect custom host/port via OLLAMA_HOST (e.g., 127.0.0.1:11435)
BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if not BASE.startswith("http"):
    BASE = "http://" + BASE
OLLAMA_GENERATE_URL = f"{BASE}/api/generate"

SYSTEM_PROMPT = (
    "You are a classifier. Look carefully at the leaf image. "
    "Classify ONLY as one of these labels exactly (lowercase): healthy or diseased. "
    "Respond with just the single word: healthy OR diseased. No extra words."
)

# ========= HELPERS =========
def b64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_generate(prompt_text: str, images_b64):
    # /api/generate accepts a single prompt and a list of images
    payload = {
        "model": MODEL,
        "prompt": prompt_text,
        "images": images_b64,     # list of base64 images
        "stream": False,
        "options": {"temperature": TEMPERATURE}
    }
    r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=240)
    r.raise_for_status()
    j = r.json()
    return (j.get("response") or "").strip().lower()

def force_label(text: str) -> str:
    t = text.strip().lower()
    if "healthy" in t or t == "h": return "healthy"
    if "diseas" in t or "sick" in t or t == "d": return "diseased"
    return "diseased" if "d" in t else "healthy"

def gather_images(root: Path):
    pairs = []
    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing folder: {cls_dir}")
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                pairs.append((p, cls))
    if not pairs:
        raise RuntimeError("No images found in healthy/ or diseased/ folders.")
    return pairs

def plot_confusion(cm, title, out_png):
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ticks = np.arange(len(CLASSES))
    ax.set_xticks(ticks); ax.set_xticklabels(CLASSES)
    ax.set_yticks(ticks); ax.set_yticklabels(CLASSES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.0f}", ha="center", va="center")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ========= MAIN =========
def main():
    all_images = gather_images(DATA_DIR)
    rows = [("path", "true", "zero_shot")]
    y_true, y_pred = [], []

    for img_path, true_lbl in all_images:
        raw = call_generate(SYSTEM_PROMPT, [b64(img_path)])
        pred = force_label(raw)
        rows.append((str(img_path), true_lbl, pred))
        y_true.append(true_lbl); y_pred.append(pred)
        print(f"[ZSL] {img_path.name:40s} true={true_lbl:8s} pred={pred:8s}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved → {OUTPUT_CSV}")

    idx = {c:i for i,c in enumerate(CLASSES)}
    y_true_idx = np.array([idx[t] for t in y_true])
    y_pred_idx = np.array([idx[p] for p in y_pred])

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=[0,1])
    acc = (cm.trace()/cm.sum()) if cm.sum() else 0.0
    cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    np.savetxt(CM_CSV, cm.astype(int), fmt="%d", delimiter=",", header=",".join(CLASSES))
    np.savetxt(CM_NORM_CSV, cm_norm, fmt="%.6f", delimiter=",", header=",".join(CLASSES))
    print(f"Saved confusion matrices → {CM_CSV}, {CM_NORM_CSV}")

    report = classification_report(y_true_idx, y_pred_idx, target_names=CLASSES, digits=4, zero_division=0)
    with open(CLASS_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n=== ZSL classification report ===")
    print(report)
    print(f"Saved → {CLASS_REPORT_TXT}")

    plot_confusion(cm, "Confusion Matrix (ZSL)", CM_PNG)
    print(f"Saved plot → {CM_PNG}")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "classes": CLASSES}, f, indent=2)
    print(f"Saved metrics → {METRICS_JSON}")

if __name__ == "__main__":
    main()
