# League of Legends Win Predictor

A binary classification model that predicts match outcomes using player performance statistics, implemented with PyTorch.

## Dataset

`league_of_legends_data_large.csv` with the following features:

| Feature | Description |
|---|---|
| `kills` | Number of kills |
| `deaths` | Number of deaths |
| `assists` | Number of assists |
| `gold_earned` | Total gold earned |
| `cs` | Creep score (minions killed) |
| `wards_placed` | Vision wards placed |
| `wards_killed` | Enemy wards destroyed |
| `damage_dealt` | Total damage dealt |
| `win` | **Target** — 1 = win, 0 = loss |

## Model

Logistic regression built with `torch.nn.Module` — a single linear layer followed by a sigmoid activation, trained with Binary Cross-Entropy loss and SGD optimizer (lr=0.01).

## Pipeline

1. Load CSV → split 80/20 train/test
2. Standardize features with `StandardScaler`
3. Convert to `float32` tensors
4. Train logistic regression model
5. Evaluate on held-out test set

## Requirements

```
pandas, scikit-learn, torch
```

---

**Two bugs to fix before training:**
- `nn.sigmoid(...)` → should be `torch.sigmoid(...)`
- `criterion` function returns the loss *class* but never calls it — should be `nn.BCELoss()(yhat, y)`
