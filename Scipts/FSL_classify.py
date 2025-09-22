# python FSL_leaf.py
import os, base64, json, csv, random
from pathlib import Path
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ========= CONFIG =========
# Windows example (uncomment & edit):
# DATA_DIR = Path(r"C:\Users\partho.ghose\Desktop\leaf_data")
DATA_DIR = Path("leaf_data")      # eval pool (expects healthy/, diseased/)
SUPPORT_DIR = Path("support")     # optional: support/healthy, support/diseased
CLASSES = ["healthy", "diseased"]

MODEL = "qwen3:4b"            # or "llava:7b", "llama3.2-vision" qwen3:4b
TEMPERATURE = 0
SEED = 42
K_PER_CLASS = 4                   # balanced: 1 from healthy + 1 from diseased (total 2)
SHUFFLE_SHOTS = True              # shuffle example order each time

OUTPUT_CSV = "predictions_fsl.csv"
CLASS_REPORT_TXT = "classification_report_fsl.txt"
METRICS_JSON = "metrics_fsl.json"
CM_CSV = "confusion_fsl.csv"
CM_NORM_CSV = "confusion_fsl_normalized.csv"
CM_PNG = "confusion_fsl.png"

BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if not BASE.startswith("http"):
    BASE = "http://" + BASE
OLLAMA_URL = f"{BASE}/api/chat"

SYSTEM_PROMPT = (
    "You are a classifier. Look carefully at the leaf image(s). "
    "Classify ONLY as one of these labels exactly (lowercase): healthy or diseased. "
    "Respond with just the single word: healthy OR diseased. No extra words."
)

# ========= HELPERS =========
def b64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_ollama(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": TEMPERATURE}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=240)
    r.raise_for_status()
    j = r.json()
    return (j.get("message", {}) or {}).get("content", "").strip().lower()

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
        raise RuntimeError(f"No images found in {root}/healthy or {root}/diseased.")
    return pairs

def try_gather_support():
    if not SUPPORT_DIR.exists():
        return None
    try:
        pairs = gather_images(SUPPORT_DIR)
        return pairs
    except Exception:
        return None

def pick_balanced_shots(pairs, k_per_class=1, exclude_path=None):
    # pairs: list[(path, label)]
    hp = [x for x in pairs if x[1] == "healthy" and x[0] != exclude_path]
    dp = [x for x in pairs if x[1] == "diseased" and x[0] != exclude_path]
    shots = []
    if hp:
        shots += random.sample(hp, min(k_per_class, len(hp)))
    if dp:
        shots += random.sample(dp, min(k_per_class, len(dp)))
    if SHUFFLE_SHOTS:
        random.shuffle(shots)
    return shots

def build_few_shot_msg(img_path: Path, shots):
    # Single-turn labeled examples to avoid copying previous assistant answers
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_path, ex_label in shots:
        msgs.append({
            "role": "user",
            "content": f"Example label: {ex_label}.",
            "images": [b64(ex_path)]
        })
    msgs.append({
        "role": "user",
        "content": "Classify this leaf. Reply with only one word.",
        "images": [b64(img_path)]
    })
    return msgs

def plot_confusion(cm, title, out_png):
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
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
    random.seed(SEED)

    all_images = gather_images(DATA_DIR)
    support_pairs = try_gather_support()  # None if not present/invalid

    rows = [("path", "true", f"few_shot_k{2*K_PER_CLASS}")]
    y_true, y_pred = [], []

    # Fallback pools if no SUPPORT_DIR provided
    hp_all = [p for p in all_images if p[1] == "healthy"]
    dp_all = [p for p in all_images if p[1] == "diseased"]

    for img_path, true_lbl in all_images:
        if support_pairs:
            shots = pick_balanced_shots(support_pairs, k_per_class=K_PER_CLASS, exclude_path=None)
        else:
            # Balanced from eval pool (excluding the current image)
            shots = pick_balanced_shots(all_images, k_per_class=K_PER_CLASS, exclude_path=img_path)

        msgs = build_few_shot_msg(img_path, shots)
        raw = call_ollama(msgs)
        pred = force_label(raw)

        rows.append((str(img_path), true_lbl, pred))
        y_true.append(true_lbl)
        y_pred.append(pred)

        # (Optional) show which labels we used this time:
        shot_labels = ",".join([s[1] for s in shots])
        print(f"[FSL] {img_path.name:40s} true={true_lbl:8s} pred={pred:8s}  [shots={shot_labels}]")

    # Save predictions
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved → {OUTPUT_CSV}")

    # Metrics
    idx = {c:i for i,c in enumerate(CLASSES)}
    y_true_idx = np.array([idx[t] for t in y_true])
    y_pred_idx = np.array([idx[p] for p in y_pred])

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=[0,1])
    acc = (cm.trace()/cm.sum()) if cm.sum() else 0.0
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    np.savetxt(CM_CSV, cm.astype(int), fmt="%d", delimiter=",", header=",".join(CLASSES))
    np.savetxt(CM_NORM_CSV, cm_norm, fmt="%.6f", delimiter=",", header=",".join(CLASSES))
    print(f"Saved confusion matrices → {CM_CSV}, {CM_NORM_CSV}")

    report = classification_report(y_true_idx, y_pred_idx, target_names=CLASSES, digits=4, zero_division=0)
    with open(CLASS_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n=== FSL classification report ===")
    print(report)
    print(f"Saved → {CLASS_REPORT_TXT}")

    plot_confusion(cm, "Confusion Matrix (FSL)", CM_PNG)
    print(f"Saved plot → {CM_PNG}")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "classes": CLASSES, "k_per_class": K_PER_CLASS}, f, indent=2)
    print(f"Saved metrics → {METRICS_JSON}")

if __name__ == "__main__":
    main()
