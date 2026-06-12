# 🍱 Food-101 Image Classification — Transfer Learning with EfficientNet

A journey from a **CNN that couldn't learn** to a fine-tuned **EfficientNetB0** that classifies food at **91% test accuracy** — told through the experiments that got it there, not just the final number.

<p align="left">
  <img alt="Python"     src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Keras"      src="https://img.shields.io/badge/Keras-EfficientNet-D00000?logo=keras&logoColor=white">
  <img alt="Task"       src="https://img.shields.io/badge/Task-Image%20Classification-1B7A43">
  <img alt="Accuracy"   src="https://img.shields.io/badge/Test%20Accuracy-91%25-success">
</p>

> This repo is the **final part** of a longer build-up. Before EfficientNet, I trained convolutional networks from scratch, watched them stall, and worked out *why*. This README walks that whole path — **from detecting the problem to solving it** — because the reasoning is the point, not the weights.

---

## TL;DR

Classifying food images is deceptively hard: dishes overlap (steak vs. prime rib), lighting and plating vary wildly, and the useful signal is buried in texture. A convolutional network **trained from scratch** can technically do it — but it overfits fast, needs mountains of data, and still lands at an accuracy you'd never ship.

The fix wasn't a bigger from-scratch model. It was **transfer learning**: stand on a backbone (EfficientNetB0) that already learned what edges, textures and shapes look like from millions of ImageNet images, then teach it *food*. From there I ran a controlled sequence of experiments — changing one variable at a time — until the evidence pointed at a clear winner.

**Headline result:** a fully fine-tuned EfficientNetB0 reached **91.0% test accuracy** — a **+5-point gain** over the frozen feature-extractor baseline, and a different universe from the from-scratch CNN.

---

## The journey

### Act I — Detecting the problem: a CNN from scratch hits a wall

I started where everyone should: a plain convolutional baseline. Get the data, become one with it, build a TinyVGG-style stack of `Conv2D` + `MaxPool2D` layers, train, evaluate, repeat. On binary food classification it works. Push it to **multi-class** food and the cracks show:

- **Overfitting is immediate** — training accuracy climbs while validation accuracy flattens and then drifts the wrong way. The model is *memorising*, not *generalising*.
- **Data augmentation helps but doesn't save it** — random flips/rotations/zooms slow the overfitting, yet accuracy stays far below anything you'd put in front of a user.
- **The diagnosis:** training a vision model from zero means learning *everything* — edges, textures, shapes, then food — from only the handful of images you have. That's the wrong fight to pick.

That's the problem, named precisely: **not enough data to learn good visual features from scratch.** Everything after this is about not having to.

### Act II — The breakthrough: don't learn features, borrow them

Enter **transfer learning**. Instead of learning visual features from nothing, load a backbone pretrained on ImageNet, **freeze it**, and bolt a small trainable classification head on top. The frozen base is now a *feature extractor* — it turns a food image into a rich vector; the head just learns to map that vector to 10 food classes.

I tested two backbones as feature extractors — **ResNetV2** and **EfficientNetB0** (via TensorFlow Hub / Keras). The result was the turning point of the whole project: with the base frozen and only **10% of the training data**, the feature-extraction model reached **~86% test accuracy** — instantly past anything the from-scratch CNN managed with far more effort. EfficientNetB0 won on accuracy-per-parameter, so it became the backbone for everything that follows.

The lesson lands hard: **most of the value was in the pretrained weights, not in my training loop.**

### Act III — Solving it properly: a series of controlled experiments *(this repo)*

A single good result isn't an answer — it's a starting point. So I ran five models that each isolate **one variable** — *how much data, augmentation on/off, frozen vs. fine-tuned* — and evaluated every one on the **identical held-out Food-101 test set** so the numbers are directly comparable.

| # | Model | Training data | Augmentation | Base model | **Test accuracy** |
|---|-------|---------------|--------------|------------|:-----------------:|
| 0 | Feature extraction (baseline) | 10% | ✗ | frozen | **85.96%** |
| 1 | Feature extraction | **1%** | ✓ | frozen | 43.12% |
| 2 | Feature extraction | 10% | ✓ | frozen | 83.40% |
| 3 | **Fine-tuning** | 10% | ✓ | unfrozen (top layers) | 86.24% |
| 4 | **Fine-tuning** | **100%** | ✓ | unfrozen (top layers) | **91.00%** |

*(Test set: 79 batches, 10 food classes. Numbers are final `model.evaluate()` results on the full test set — not peak training accuracy.)*

**Fine-tuning** is the second phase: once the head has stabilised on a frozen base, unfreeze the top layers of EfficientNet and continue training at a **low learning rate**, so the backbone gently adapts its high-level features to food without destroying what it already knew.

#### What the experiments prove

- **Data matters most.** Model 1 (1% of data) collapses to 43% — augmentation can't rescue a model that hasn't seen enough examples. Scaling to 100% of the data (Model 4) is the single biggest jump.
- **Fine-tuning beats feature extraction — once you have data.** Unfreezing the top of EfficientNet (Model 3 vs. Model 2) adds ~3 points at 10% data, and the gap widens at full scale.
- **Augmentation is a regularizer, not a magic bullet.** On 10% data it slightly *lowered* raw accuracy (Model 0 → Model 2), but the augmented model generalises better when later fine-tuned.

The senior-engineer takeaway, in order of impact: **more data → fine-tune → then worry about augmentation.**

### Act IV — Where it scales: Food Vision

The same recipe doesn't stop at 10 classes. The natural next step is **all 101 Food-101 classes** — a "big dog" fine-tuned EfficientNet, evaluated with a full classification report, per-class accuracy, and a hunt for the *most wrong* predictions to see where the model genuinely confuses one dish for another. This repo is the distilled, reproducible core of that progression; scaling to the full dataset is the production direction below.

---

## Techniques demonstrated

- **Transfer learning** with the Keras Functional API — EfficientNetB0 backbone, `GlobalAveragePooling2D` head.
- **Two-phase training:** frozen-base feature extraction, then selective **fine-tuning** (`base_model.trainable = True` with a lowered learning rate).
- **Data augmentation** as in-model preprocessing layers (`RandomFlip`, `RandomRotation`, `RandomZoom`) — runs on-GPU, ships with the model.
- **`ModelCheckpoint`** callbacks to save and restore best weights between phases.
- **Efficient input pipelines** with `image_dataset_from_directory` and prefetching.
- **Experiment tracking** — loss/accuracy curves per run, TensorBoard logging, and an apples-to-apples comparison harness across all five models.

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

- Scale training to the **full 101-class Food Vision** model and export the best checkpoint (`SavedModel` / TFLite).
- Serve it behind a **FastAPI** `/predict` endpoint, containerize with **Docker**, and deploy to **AWS EC2** via a **GitHub Actions** pipeline — the same stack I use in my [Rag-fullstack-docker-AWS](https://github.com/Amith-Ganta/Rag-fullstack-docker-AWS) and [FastAPI-ML-Docker-AWS](https://github.com/Amith-Ganta/FastAPI-ML-Docker-AWS) projects.

---

## About

Built by **Amith Ganta** — Agentic AI / RAG Engineer with hands-on ML, NLP and computer-vision foundations.
📧 amigan@ktu.lt

<sub>Dataset: [Food-101](https://www.tensorflow.org/datasets/catalog/food101) (Bossard et al., 2014). EfficientNet: Tan &amp; Le, 2019.</sub>
