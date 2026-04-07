from __future__ import annotations

import os
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

"""
    Train and evaluate a Random Forest classifier on the Iris dataset. Random Forest is usefull for
    tabular data, (like the iris dataset)

    Optionally save the fitted model under models/ with joblib, reload it, and run predictions on the
    same train/test split.
    
    Prints metrics and displays a confusion matrix

    Author :        Martijn Folmer
    Date created :  07-04-2026
"""


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier on training data.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    """
    Plot a confusion matrix (counts, not normalized) using imshow.
    """

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()


def load_iris_split(
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load Iris features/labels and return a stratified train/test split plus target names.
    """
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target,
    )
    class_names = [str(name) for name in data.target_names]
    return X_train, X_test, y_train, y_test, class_names


def save_random_forest_model(model: RandomForestClassifier, path: str) -> None:
    """
    Persist a fitted RandomForestClassifier to disk (joblib).
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)


def load_random_forest_model(path: str) -> RandomForestClassifier:
    """
    Load a RandomForestClassifier previously saved with save_random_forest_model.
    """
    return joblib.load(path)


if __name__ == "__main__":

    random_state = 42
    test_size = 0.25
    n_estimators = 100
    max_depth: int | None = None
    save_model_to_disk = True
    models_dir = "models"
    model_filename = "random_forest_iris.joblib"
    model_path = os.path.join(models_dir, model_filename)

    ####################
    # Load data and split
    ####################

    X_train, X_test, y_train, y_test, class_names = load_iris_split(
        test_size=test_size,
        random_state=random_state,
    )
    print(f"Train samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Test samples:  {X_test.shape[0]}")
    print(f"Classes: {class_names}")

    ####################
    # Train Random Forest
    ####################

    model = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    ####################
    # Evaluate
    ####################

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    ####################
    # Save model (optional)
    ####################

    if save_model_to_disk:
        save_random_forest_model(model, model_path)
        print(f"\nModel saved to: {model_path}")

    ####################
    # Load model and run on same train/test data
    ####################

    if save_model_to_disk:
        model_loaded = load_random_forest_model(model_path)
        y_pred_train_loaded = model_loaded.predict(X_train)
        y_pred_test_loaded = model_loaded.predict(X_test)
        train_acc_loaded = accuracy_score(y_train, y_pred_train_loaded)
        test_acc_loaded = accuracy_score(y_test, y_pred_test_loaded)
        matches_in_memory_test = np.array_equal(y_pred, y_pred_test_loaded)
        print("\nAfter load from disk (same X_train / X_test):")
        print(f"  Train accuracy: {train_acc_loaded:.4f}")
        print(f"  Test accuracy:  {test_acc_loaded:.4f}")
        print(f"  Test predictions match in-memory model: {matches_in_memory_test}")

    ####################
    # Confusion matrix (display only)
    ####################

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        title="Random Forest - Iris (test set)",
    )
    plt.show()
