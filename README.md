League of Legends Win Prediction
A binary classification project that predicts match win/loss outcomes from individual in-game performance stats using logistic regression in PyTorch.
Dataset
CSV file with 1000 match records. Features used:
kills, deaths, assists, gold_earned, cs, wards_placed, wards_killed, damage_dealt
Target: win (0 or 1)
What's Done
Phase 1 — Baseline Pipeline

80/20 train-test split
Feature normalization via StandardScaler (fit on train only, transform on test)
Conversion to float32 tensors (PyTorch default)
Logistic regression model: Linear(8→1) + Sigmoid activation
BCELoss + SGD optimizer (lr=0.01)
1000-epoch training loop with loss logged every 100 epochs
Accuracy evaluation on both train and test sets

Result: ~54% train accuracy, ~49% test accuracy
Phase 2 — Regularization

Added L2 regularization via weight_decay=0.01 in the SGD optimizer
Re-trained for 1000 epochs

Result: ~54% train accuracy, ~51% test accuracy — marginal improvement
Key Takeaways (so far)

Features must be normalized before training
Scaler must be fit only on training data — applying it separately to test data prevents data leakage
PyTorch expects float32; StandardScaler outputs float64 by default
Predicting team outcomes from individual stats has a natural accuracy ceiling for linear models


Future Goals

 Implement mini-batch training using DataLoader
 Add a multi-layer architecture (hidden layers + ReLU) to capture non-linear patterns
 Implement early stopping properly (per-epoch convergence check)
 Experiment with Adam optimizer
 Add precision, recall, F1 score — accuracy alone is insufficient for imbalanced data
 Move training to GPU (CUDA already confirmed available)


Requirements

Python 3.10+
PyTorch
scikit-learn
pandas
