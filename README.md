# 🍱 Food-101 Image Classification — Transfer Learning with EfficientNet

Fine-tuning a pretrained **EfficientNetB0** to classify food images, taking a 10-class subset of **Food-101** from **86% → 91% test accuracy** through a controlled sequence of feature-extraction and fine-tuning experiments.

<p align="left">
  <img alt="Python"     src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Keras"      src="https://img.shields.io/badge/Keras-EfficientNet-D00000?logo=keras&logoColor=white">
  <img alt="Task"       src="https://img.shields.io/badge/Task-Image%20Classification-1B7A43">
  <img alt="Accuracy"   src="https://img.shields.io/badge/Test%20Accuracy-91%25-success">
</p>

---

## TL;DR

I treated this as an **experiment, not a tutorial**: instead of training one model and hoping, I ran five models that isolate one variable at a time — *how much data, augmentation on/off, frozen vs. fine-tuned* — and measured each on the **same held-out test set**. The result is a clear, evidence-backed answer to "what actually moves the needle in transfer learning."

**Headline result:** fine-tuning the full base model on all available data reached **91.0% test accuracy** — a **+5-point gain** over the frozen feature-extractor baseline.

---

## The experiment

Every model uses the same pretrained **EfficientNetB0** backbone (ImageNet weights) and is evaluated on the identical Food-101 test split, so the numbers are directly comparable.

| # | Model | Training data | Augmentation | Base model | **Test accuracy** |
|---|-------|---------------|--------------|------------|:-----------------:|
| 0 | Feature extraction (baseline) | 10% | ✗ | frozen | **85.96%** |
| 1 | Feature extraction | **1%** | ✓ | frozen | 43.12% |
| 2 | Feature extraction | 10% | ✓ | frozen | 83.40% |
| 3 | **Fine-tuning** | 10% | ✓ | unfrozen (top layers) | 86.24% |
| 4 | **Fine-tuning** | **100%** | ✓ | unfrozen (top layers) | **91.00%** |

*(Test set: 79 batches, 10 food classes. Numbers are final `model.evaluate()` results on the full test set — not peak training accuracy.)*

### What the numbers tell you

- **Data matters most.** Model 1 (1% of data) collapses to 43% — augmentation can't rescue a model that hasn't seen enough examples. Scaling to 100% of the data (Model 4) is the single biggest jump.
- **Fine-tuning beats feature extraction — once you have data.** Unfreezing the top of EfficientNet (Model 3 vs. Model 2) adds ~3 points at 10% data, and the gap widens at full scale.
- **Augmentation is a regularizer, not a magic bullet.** On 10% data it slightly *lowered* raw accuracy (Model 0 → Model 2) but the augmented model generalizes better when later fine-tuned.

This is the actual senior-engineer takeaway: **more data → fine-tune → then worry about augmentation**, in that order of impact.

---

## Techniques demonstrated

- **Transfer learning** with the Keras Functional API — EfficientNetB0 backbone, `GlobalAveragePooling2D` head.
- **Two-phase training:** frozen-base feature extraction, then selective **fine-tuning** (`base_model.trainable = True` with a lowered learning rate).
- **Data augmentation** as in-model preprocessing layers (`RandomFlip`, `RandomRotation`, `RandomZoom`) — runs on-GPU, ships with the model.
- **`ModelCheckpoint`** callbacks to save and restore best weights.
- **Efficient input pipelines** with `image_dataset_from_directory` and prefetching.
- **Experiment tracking** — loss/accuracy curves per run, TensorBoard logging, and an apples-to-apples comparison harness.

---

## Repo contents

```
transfer-learning-food101/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── transfer_learning_food101.ipynb   # the full, executed experiment
└── src/
    └── predict.py                        # load a saved model, classify one image
```

> The notebook is the primary artifact — it carries every model, the training curves, and the evaluation outputs inline. `src/predict.py` shows how the trained model is used for inference on a new image.

---

## How to run

```bash
pip install -r requirements.txt
# open the notebook
jupyter notebook notebook/transfer_learning_food101.ipynb

# or, once you have a saved model, classify an image:
python src/predict.py --model saved_model/ --image path/to/food.jpg
```

The notebook was developed in **Google Colab** (GPU runtime). It downloads the Food-101 10-class subset automatically.

---

## What I'd do next (production direction)

The experiment is complete; turning it into a service is the natural next step, and one I've built for other models in my portfolio:

- Export the best model (`SavedModel` / TFLite) and serve it behind a **FastAPI** `/predict` endpoint.
- Containerize with **Docker** and deploy to **AWS EC2** via a **GitHub Actions** pipeline (the same stack I use in my [Rag-fullstack-docker-AWS](https://github.com/Amith-Ganta/Rag-fullstack-docker-AWS) and [FastAPI-ML-Docker-AWS](https://github.com/Amith-Ganta/FastAPI-ML-Docker-AWS) projects).

---

## About

Built by **Amith Ganta** — Agentic AI / RAG Engineer with hands-on ML, NLP and computer-vision foundations.
📧 amigan@ktu.lt

<sub>Dataset: [Food-101](https://www.tensorflow.org/datasets/catalog/food101) (Bossard et al., 2014). EfficientNet: Tan &amp; Le, 2019.</sub>
