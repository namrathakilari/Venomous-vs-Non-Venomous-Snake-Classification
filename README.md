#  Venomous vs Non-Venomous Snake Classification using ResNet50

> A safety-critical image classifier that identifies whether a snake is venomous or non-venomous using Transfer Learning with fine-tuned ResNet50 — designed to assist rural communities, forest rangers, and wildlife rescue teams.

---

##  Project Overview

Given a photo of a snake, the model predicts its danger level with a confidence score and a **Grad-CAM heatmap** showing exactly which part of the snake influenced the decision.

```
Input Image → ResNet50 (fine-tuned) → Prediction + Confidence + Grad-CAM

Example Output:
Prediction  : VENOMOUS
Confidence  : 94.3%
Assessment  : 🔴 VENOMOUS — Exercise Caution!

non_venomous   ████░░░░░░░░░░░░░░░░░░░░░░░░░░  5.7%
venomous       ██████████████████████████░░░░  94.3%
```

---

##  Classes & Species Covered

| Class | Species |
|---|---|
| 🔴 **Venomous** | Indian Cobra, Russell's Viper, Common Krait, King Cobra, Saw-scaled Viper |
| 🟢 **Non-Venomous** | Indian Rat Snake, Indian Rock Python, Checkered Keelback, Trinket Snake, Buff-striped Keelback |

---

##  Model Architecture

```
ResNet50 (pretrained on ImageNet)
  → Freeze layers 1–2
  → Fine-tune layer3 + layer4
  → Replace FC head:
       Linear(2048 → 512) → BatchNorm → ReLU → Dropout(0.4)
       → Linear(512 → 128) → ReLU → Dropout(0.3)
       → Linear(128 → 2)
```

- **Optimizer:** Adam with differential learning rates
  - `layer3` → `5e-5` | `layer4` → `1e-4` | Custom head → `1e-3`
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Loss:** CrossEntropyLoss | **Epochs:** 25 | **Batch Size:** 32

---

##  Project Steps

| Step | Description |
|---|---|
| 1️⃣ Install & Import | PyTorch, torchvision, icrawler, scikit-learn, seaborn |
| 2️⃣ Dataset Scraping | Automated image collection via `icrawler` + Bing with 5 species keywords per class |
| 3️⃣ Data Cleaning | Remove corrupt files, images < 80×80px, non-RGB images + 80/20 train/val split |
| 4️⃣ EDA & Visualisation | Class distribution bar chart + sample image grid |
| 5️⃣ Data Augmentation | RandomCrop, Flip, Rotation, ColorJitter, Perspective, RandomErasing (occlusion) |
| 6️⃣ Model Building | Fine-tuned ResNet50 with 3-layer custom classification head |
| 7️⃣ Training | Differential LRs across layer3 / layer4 / head with ReduceLROnPlateau |
| 8️⃣ Evaluation | Confusion Matrix + ROC-AUC Curve + per-class classification report |
| 9️⃣ Grad-CAM | Heatmap visualisation verifying model focuses on snake features not background |
| 🔟 Prediction | `predict_snake()` — label, confidence, probability bars, Grad-CAM overlay |

---

##  Visualisations

- **Figure 1:** Class distribution bar chart (train vs val)
- **Figure 2:** Sample images grid (5 per class)
- **Figure 3:** Augmentation examples (same image, 9 transforms)
- **Figure 4:** Training Loss & Validation Accuracy curves
- **Figure 5:** Confusion Matrix + ROC Curve
- **Figure 6:** Grad-CAM heatmaps (original vs highlighted regions)

---

##  Results

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Non-Venomous | 0.89 | 0.96 | 0.92 | 51 |
| Venomous | 0.95 | 0.88 | 0.91 | 48 |
| **Overall Accuracy** | | | **0.92** | 99 |
| **ROC-AUC Score** | | | **0.9575** | |

> **Venomous Precision = 0.95** — when the model flags a snake as dangerous, it is correct 95% of the time, minimising false alarms.
> **Note:** Venomous Recall = 0.88 means ~12% of venomous snakes are missed. Future work targets lowering the classification threshold for venomous predictions to prioritise recall in safety-critical deployments.

---

##  Grad-CAM: Model Interpretability

Unlike standard classifiers, this project includes **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualise *where* the model looks when making a prediction. This verifies the model is responding to **scale patterns, head shape, and body markings** — not background elements like grass or rocks.

This is especially critical for a safety application where trust in the model's reasoning matters.

---

##  Requirements

```bash
pip install torch torchvision icrawler pillow matplotlib seaborn scikit-learn
```

> 💡 GPU runtime strongly recommended. In Colab: `Runtime → Change runtime type → T4 GPU`

---

##  Usage

```python
# Predict on any snake image
predict_snake('your_snake_image.jpg')

# Output:
# Prediction  : VENOMOUS
# Confidence  : 94.3%
# Assessment  : 🔴 VENOMOUS — Exercise Caution!
```

---

##  Comparison: Cattle/Buffalo vs This Project

| Aspect | Cattle vs Buffalo | Snake Classification |
|---|---|---|
| Inter-class similarity | Low | **High** (scales, colouring similar) |
| Fine-tuned layers | layer4 only | **layer3 + layer4** |
| Custom head depth | 2 layers | **3 layers + BatchNorm** |
| Additional evaluation | Confusion Matrix | **+ ROC-AUC Curve** |
| Interpretability | None | **Grad-CAM heatmaps** |
| Application | Agricultural | **Safety-critical / wildlife** |

---

##  Future Improvements

1. **Multi-class species ID** — extend to species-level classification (cobra vs krait vs viper)
2. **Lower venomous threshold** — tune decision boundary to maximise recall for dangerous snakes
3. **EfficientNetV2 benchmark** — compare accuracy vs inference speed against ResNet50
4. **Mobile deployment** — convert to TFLite/ONNX for field use on ranger smartphones
5. **Uncertainty estimation** — Monte Carlo Dropout to flag low-confidence predictions for human review

---

