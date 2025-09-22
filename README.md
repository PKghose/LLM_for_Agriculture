# LLM_for_Agriculture
Local, privacy-friendly leaf image classification into healthy / diseased using Ollama vision models.



# HTTPS 
git clone https://github.com/PKghose/LLM_for_Agriculture.git
cd LLM_for_Agriculture
# or SSH
git clone git@github.com:/PKghose/LLM_for_Agriculture.git
cd LLM_for_Agriculture


# Leaf LLM Classifier (Zero-Shot & Few-Shot, Local via Ollama)

This repo provides two Python scripts for **classifying leaf images** into **`healthy`** or **`diseased`** using **local** LLM/VLM models served by **Ollama**.  
It supports both **Zero-Shot Learning (ZSL)** and **Few-Shot Learning (FSL)** and automatically produces **confusion matrices**, **classification reports**, and **CSV** outputs.

- **ZSL** – `zsl_classify_generate.py`  
  One image at a time, with a strict prompt. No labeled examples are sent, so it’s fast and simple.

- **FSL** – `FSL_leaf.py`  
  Sends a **balanced** set of labeled example images (**support shots**) along with each query image to improve robustness on edge cases. The number of shots per class is configurable.

Everything runs **fully offline** on your machine via Ollama.

---

## Features

- Works with **vision-capable models** exposed by Ollama (e.g., `llava:7b`, `minicpm-v:8b`, `paligemma:3b`, `gemma3:4b`, etc.)
- **Balanced few-shot** support: pick the same number of examples from each class; optional, fixed `support/` set for stable evaluation.
- **Deterministic** outputs by default (`temperature = 0`).
- Automatic exports:
  - `predictions_*.csv`
  - `confusion_*.csv` (+ normalized)
  - `confusion_*.png`
  - `classification_report_*.txt`
  - `metrics_*.json`

---

## Data layout

Place your dataset as:leaf_data/
healthy/
img001.jpg
...
diseased/
img101.jpg
...


**Note:** The scripts never read your folder names during prediction; folder names are only used locally to assign ground-truth labels and pick few-shot examples for prompting.

---

## Requirements

- **Ollama** (recent version; multimodal support required for image models)
- **Python 3.10+**
- Python packages:
  - `requests`, `numpy`, `matplotlib`, `scikit-learn`

Install Python dependencies with your preferred method (e.g., a virtual environment plus `pip install -r requirements.txt`).

---

## Models

Use a **vision-capable** model. Examples that work well locally:

- `llava:7b` (popular general VLM)
- `minicpm-v:8b` (efficient)
- `paligemma:3b` (Gemma-family vision)
- `gemma3:4b` (multimodal)
- Other vision tags that Ollama exposes for your hardware


