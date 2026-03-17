# 🎮 League of Legends — Win Prediction
A machine learning project to predict match outcomes in **League of Legends** using player performance statistics and **binary classification with PyTorch**.

---

## 📌 Overview
Winning in League of Legends depends on a complex mix of combat performance, economy, and vision control. This project builds an **end-to-end binary classification pipeline** that predicts:
- **Win (1)**
- **Loss (0)**

The project is designed for **educational purposes**, focusing on correct ML practices — data preprocessing, model building, regularization, and iterative improvement.

---

## 🎯 Objectives
- Preprocess and normalize match statistics for model training
- Build a **logistic regression baseline** using PyTorch
- Understand **underfitting vs overfitting** through experimentation
- Apply regularization techniques (L2 via weight decay)
- Iteratively improve the model with deeper architectures, BatchNorm, and Dropout
- Evaluate using accuracy and other classification metrics

---

## 📁 Project Structure
```
lol-win-prediction/
│
├── data/                  # Dataset (excluded from git)
├── notebooks/             # Training, evaluation, experimentation
├── models/                # Saved model checkpoints (excluded)
├── reports/               # Metrics, plots, confusion matrices
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset
Match-level player statistics from League of Legends games.

### Features
| Feature | Description |
|---|---|
| `kills` | Number of kills |
| `deaths` | Number of deaths |
| `assists` | Number of assists |
| `gold_earned` | Total gold earned |
| `cs` | Creep score (minions killed) |
| `wards_placed` | Vision wards placed |
| `wards_killed` | Enemy wards destroyed |
| `damage_dealt` | Total damage dealt to champions |

### Target
| Label | Value |
|---|---|
| Win | 1 |
| Loss | 0 |

⚠️ Dataset is **not included** in this repository due to size constraints.

---

## 🛠️ Tech Stack
- Python
- PyTorch
- Scikit-learn
- NumPy, Pandas
- Matplotlib

---

## 🧠 Model Architecture

### Phase 1 — Baseline (Logistic Regression)
```
Input(8) → Linear(8→1) → Sigmoid → Output
```
- **Loss:** BCELoss
- **Optimizer:** SGD with L2 regularization (`weight_decay=0.01`)
- **Normalization:** StandardScaler (fit on train only)

### Phase 2 — Planned (Deep Model)
- Hidden layers with ReLU activation
- BatchNormalization and/or Dropout
- Adam optimizer
- DataLoader with mini-batch training
- Multiple architectures for comparison

---

## 📈 Results

### Phase 1 — Baseline
| | Train Accuracy | Test Accuracy |
|---|---|---|
| Without L2 | 54.25% | 49.50% |
| With L2 | 54.37% | 51.00% |

**Diagnosis: Underfitting** — model is too simple to learn meaningful patterns from this data.

---

## 🚧 Project Status

**Phase 1 — Baseline Pipeline: Complete ✅**
- Data loading, splitting, normalization
- Logistic regression model
- L2 regularization

**Phase 2 — Improvement: In Progress 🚧**

Planned work:
- Fix stuck/fixed loss issue
- Add `DataLoader` and mini-batch training
- Check and handle class imbalance
- Build deeper model with hidden layers
- Add `BatchNorm` and/or `Dropout`
- Compare multiple architectures

---

## 📜 Disclaimer
This project is intended **strictly for educational and research purposes**.
