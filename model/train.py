"""
train.py
--------
Model training script for the Smart Ticket Classifier.

Trains two independent classifiers:
    - Priority classifier  (Low / Medium / High)
    - Area classifier      (Networks / Hardware / Access)

Each classifier is a Pipeline combining:
    - TextPreprocessor        (lowercasing, punctuation removal, stopword filtering)
    - TF-IDF vectorizer       (converts text to numerical features)
    - LinearSVC               (high-performance classifier for short text)
    - CalibratedClassifierCV  (wraps LinearSVC to enable confidence scores)

After training, generates visual evaluation reports saved to /reports:
    - Confusion matrix heatmaps
    - Per-class precision / recall / F1 bar charts
    - Cross-validation accuracy chart
    - Summary metrics CSV

Usage:
    python model/train.py

Output:
    model/priority_model.pkl
    model/area_model.pkl
    reports/confusion_matrix_priority.png
    reports/confusion_matrix_area.png
    reports/metrics_priority.png
    reports/metrics_area.png
    reports/cross_validation.png
    reports/summary.csv
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Allow import of preprocessor from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessor import TextPreprocessor

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

DATA_PATH    = os.path.join("data", "tickets.csv")
MODEL_DIR    = "model"
REPORTS_DIR  = "reports"
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

print("=" * 52)
print("  Smart Ticket Classifier - Training")
print("=" * 52)

print("\n  [1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"        {len(df)} tickets loaded.")
print(f"        Priority distribution:\n{df['priority'].value_counts().to_string()}")
print(f"        Area distribution:\n{df['area'].value_counts().to_string()}")

X          = df["text"]
y_priority = df["priority"]
y_area     = df["area"]

# ------------------------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------------------------

print("\n  [2/6] Splitting into train and test sets (80/20)...")

X_train, X_test, yp_train, yp_test, ya_train, ya_test = train_test_split(
    X, y_priority, y_area,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_priority
)

print(f"        Training samples : {len(X_train)}")
print(f"        Test samples     : {len(X_test)}")

# ------------------------------------------------------------------------------
# Build pipeline
# ------------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Full text classification pipeline:
        TextPreprocessor  ->  cleans raw ticket text
        TfidfVectorizer   ->  converts text to weighted term-frequency matrix
        LinearSVC         ->  finds optimal class boundaries
        CalibratedCV      ->  enables predict_proba() for confidence scores
    """
    return Pipeline([
        ("preprocessor", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            min_df=1
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE)
        ))
    ])

# ------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------

print("\n  [3/6] Training models...")

print("        Training priority classifier...")
priority_model = build_pipeline()
priority_model.fit(X_train, yp_train)

print("        Training area classifier...")
area_model = build_pipeline()
area_model.fit(X_train, ya_train)

# ------------------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------------------

print("\n  [4/6] Running 5-fold cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

priority_cv = cross_val_score(build_pipeline(), X, y_priority, cv=cv, scoring="accuracy")
area_cv     = cross_val_score(build_pipeline(), X, y_area,     cv=cv, scoring="accuracy")

print(f"        Priority CV accuracy: {priority_cv.mean():.2%} (+/- {priority_cv.std():.2%})")
print(f"        Area CV accuracy:     {area_cv.mean():.2%} (+/- {area_cv.std():.2%})")

# ------------------------------------------------------------------------------
# Evaluate on test set
# ------------------------------------------------------------------------------

print("\n  [5/6] Evaluating on test set...")

yp_pred = priority_model.predict(X_test)
ya_pred = area_model.predict(X_test)

print("\n  --- Priority Classifier ---")
print(classification_report(yp_test, yp_pred))

print("  --- Area Classifier ---")
print(classification_report(ya_test, ya_pred))

# ------------------------------------------------------------------------------
# Generate visual reports
# ------------------------------------------------------------------------------

print("  [6/6] Generating visual reports...")
os.makedirs(REPORTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Saves a styled confusion matrix heatmap as a PNG."""
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"        Saved: {path}")


def plot_class_metrics(y_true, y_pred, labels, title, filename):
    """Saves a grouped bar chart of precision, recall, and F1 per class."""
    report  = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    x       = np.arange(len(labels))
    width   = 0.25
    colors  = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        values = [report[label][metric] for label in labels]
        bars   = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"        Saved: {path}")


def plot_cross_validation(priority_scores, area_scores, filename):
    """Saves a bar chart comparing cross-validation scores across folds."""
    folds  = [f"Fold {i+1}" for i in range(len(priority_scores))]
    x      = np.arange(len(folds))
    width  = 0.35
    colors = ["#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, priority_scores, width, label="Priority", color=colors[0])
    bars2 = ax.bar(x + width/2, area_scores,     width, label="Area",     color=colors[1])

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2%}",
                ha="center", va="bottom", fontsize=8
            )

    ax.axhline(priority_scores.mean(), color=colors[0], linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(area_scores.mean(),     color=colors[1], linestyle="--", linewidth=1, alpha=0.6)

    ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"        Saved: {path}")


def save_summary_csv(priority_cv, area_cv, yp_test, yp_pred, ya_test, ya_pred, filename):
    """Saves a summary CSV with key metrics for both classifiers."""
    from sklearn.metrics import accuracy_score, f1_score

    rows = [
        {
            "classifier":       "Priority (High/Medium/Low)",
            "test_accuracy":    f"{accuracy_score(yp_test, yp_pred):.2%}",
            "cv_accuracy_mean": f"{priority_cv.mean():.2%}",
            "cv_accuracy_std":  f"{priority_cv.std():.2%}",
            "macro_f1":         f"{f1_score(yp_test, yp_pred, average='macro'):.2%}",
        },
        {
            "classifier":       "Area (Networks/Hardware/Access)",
            "test_accuracy":    f"{accuracy_score(ya_test, ya_pred):.2%}",
            "cv_accuracy_mean": f"{area_cv.mean():.2%}",
            "cv_accuracy_std":  f"{area_cv.std():.2%}",
            "macro_f1":         f"{f1_score(ya_test, ya_pred, average='macro'):.2%}",
        },
    ]
    path = os.path.join(REPORTS_DIR, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"        Saved: {path}")


# Generate all reports
priority_labels = ["High", "Medium", "Low"]
area_labels     = ["Networks", "Hardware", "Access"]

plot_confusion_matrix(yp_test, yp_pred, priority_labels,
                      "Confusion Matrix — Priority Classifier",
                      "confusion_matrix_priority.png")

plot_confusion_matrix(ya_test, ya_pred, area_labels,
                      "Confusion Matrix — Area Classifier",
                      "confusion_matrix_area.png")

plot_class_metrics(yp_test, yp_pred, priority_labels,
                   "Precision / Recall / F1 — Priority Classifier",
                   "metrics_priority.png")

plot_class_metrics(ya_test, ya_pred, area_labels,
                   "Precision / Recall / F1 — Area Classifier",
                   "metrics_area.png")

plot_cross_validation(priority_cv, area_cv, "cross_validation.png")

save_summary_csv(priority_cv, area_cv, yp_test, yp_pred, ya_test, ya_pred, "summary.csv")

# ------------------------------------------------------------------------------
# Save models
# ------------------------------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

priority_path = os.path.join(MODEL_DIR, "priority_model.pkl")
area_path     = os.path.join(MODEL_DIR, "area_model.pkl")

with open(priority_path, "wb") as f:
    pickle.dump(priority_model, f)

with open(area_path, "wb") as f:
    pickle.dump(area_model, f)

print(f"\n  [OK] Models saved to /{MODEL_DIR}")
print(f"       {priority_path}")
print(f"       {area_path}")
print("\n  Training complete.\n")
