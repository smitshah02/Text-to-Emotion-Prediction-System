# Text-to-Emotion Prediction System (Multi-Label, DistilBERT + TensorFlow)

Classifies **multiple emotions per sentence** using a DistilBERT encoder and a lightweight Keras head. Built for fast experimentation with clean training/eval flows, threshold tuning, and reproducible results.

> **Why it matters:** Emotion understanding helps content moderation, support chat routing, mental-health triage signals, and analytics for product feedback.

---

## 🔑 Highlights (What I built)

- **Problem framing:** Multi-label emotion classification (sigmoid outputs) with **Binary Cross-Entropy** loss and **micro-F1** tracking.
- **Modern NLP backbone:** **DistilBERT (base-uncased)** tokenizer + encoder for robust sentence representations.
- **Training loop:** TF/Keras pipeline with `ModelCheckpoint` (best val F1) and `ReduceLROnPlateau` for stable convergence.
- **Inference & tuning:** Batch prediction + **threshold sweep (0.30–0.61)** to maximize micro-F1 when ground truth is available.
- **Reproducible data loader:** Uses Hugging Face `datasets` for CSV ingestion and TF dataset conversion.

---

## 🧱 Model & Data

- **Backbone:** `distilbert-base-uncased` (Hugging Face)
- **Max sequence length:** 64 tokens
- **Head:** Dense(256, ReLU) → Dropout(0.3) → Dense(num_labels, **sigmoid**)
- **Loss & metric:** Binary Cross-Entropy + **micro F1** (threshold=0.5 during training)
- **Input format (CSV):**
  - Column **`text`** (string)
  - One column per **label** (0/1), e.g., `anger, joy, sadness, fear, love, surprise, neutral`
  - Train/val: `train.csv`, `dev.csv`
  - Test: any CSV with at least a `text` column (label columns optional)

> The script treats every column **after `text`** as a label and builds `num_labels` accordingly during training.
> During `predict`, `num_labels` is set to **7**; set this to match your dataset’s label count if different. :contentReference[oaicite:1]{index=1}

---

## 🗂️ Repository Structure

├─ nn.py # Training / inference script (this project’s core)

├─ requirements.txt # Python dependencies

├─ train.csv # (user-provided) training data, CSV

├─ dev.csv # (user-provided) validation data, CSV

└─ test-in.csv # (user-provided) test data for inference
