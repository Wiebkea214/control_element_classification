import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import psutil
import re
import time

from setup_vector_database import *
from preprocessing_data import *
from pathlib import Path
from collections import Counter

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

###########################################

def data_plot(data, title):
    # divide header and data
    titles = data[0]
    data_entries = data[1:]

    n_rows = len(data_entries)
    n_cols = len(titles)

    # group after columns
    columns = list(zip(*data_entries))

    # setup bars
    x = np.arange(n_rows)
    bar_width = 0.25
    colors = plt.cm.get_cmap('plasma')(np.linspace(0, 1, n_cols))

    for i, col in enumerate(columns):
        plt.bar(x + i * bar_width, col, width=bar_width, label=titles[i], color=colors[i])

    # axis and title
    plt.xticks(x + bar_width, [f"Zeile {i + 1}" for i in range(n_rows)])
    plt.xlabel("entry")
    plt.ylabel("sim score")
    plt.title(title)
    plt.legend()
    plt.close()


def monitor_cpu(interval, cpu_usage, stop_event, timestamps, start_time):
    # Calibration
    psutil.cpu_percent(interval=None)

    # Actual measuring
    while not stop_event.is_set():
        cpu_usage.append(psutil.cpu_percent(interval=interval))
        timestamps.append(time.perf_counter() - start_time)

    return cpu_usage, timestamps


def analysis_cpu_usage(interval, train_start, train_end, pred_start, pred_end, cpu_start, cpu_usage, timestamps, path_dir):
    training_start = train_start - cpu_start
    training_end = train_end - cpu_start
    prediction_start = pred_start - cpu_start
    prediction_end = pred_end - cpu_start

    plt.ion()
    fig = plt.figure(figsize=(10,6))
    title_size = fig.get_figwidth() * 1
    plt.ylim(0, 100)
    plt.plot(timestamps, cpu_usage, color="blue", marker=".", label="CPU-Load")

    if training_start and training_end:
        mask_train = [(training_start <= t <= training_end) for t in timestamps]
        red_times = [t for t, m in zip(timestamps, mask_train) if m]
        red_usage = [u for u, m in zip(cpu_usage, mask_train) if m]
        plt.plot(red_times, red_usage, color="red", marker=".", label="Training")
        plt.axvspan(training_start, training_end, color="red", alpha=0.1)

    if prediction_start and prediction_end:
        mask_pred = [(prediction_start <= t <= prediction_end) for t in timestamps]
        red_times = [t for t, m in zip(timestamps, mask_pred) if m]
        red_usage = [u for u, m in zip(cpu_usage, mask_pred) if m]
        plt.plot(red_times, red_usage, color="green", marker=".", label="Prediction")
        plt.axvspan(prediction_start, prediction_end, color="green", alpha=0.1)

    plt.title("CPU-load during training", fontsize=title_size)
    plt.xlabel(f"measure points (all {interval*1000}ms)")
    plt.ylabel("CPU-load (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path_dir, "cpu_usage.png"), bbox_inches="tight")
    plt.close()


def analysis_performance(y_test, y_pred, encoder, path_dir):
    # Performance data
    class_names = encoder.classes_

    # Precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    x = np.arange(len(precision))

    # Plots
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    title_size = fig.get_figwidth() * 1
    plt.bar(x, precision, width=0.25, label="Precision")
    plt.bar([i + 0.25 for i in x], recall, width=0.25, label="Recall")
    plt.bar([i + 0.50 for i in x], f1, width=0.25, label="F1-Score")
    plt.ylim(0, 1)
    plt.xticks([i + 0.25 for i in x], class_names, rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Value")
    plt.title("Precision, Recall and F1 per class", fontsize=title_size)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(path_dir, "performance.png"), bbox_inches="tight")
    plt.close()


def analysis_conf_matrix(y_test, y_pred, encoder, path_dir, filename):
    if encoder:
        class_names = encoder.classes_
    else:
        class_names = np.array(list(set(y_pred)))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig = plt.figure(figsize=(8,6))
    title_size = fig.get_figwidth() * 0.5
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix", fontsize=title_size)
    plt.xlabel("Predicted classes")
    plt.ylabel("True classes")
    plt.savefig(os.path.join(path_dir, filename), bbox_inches="tight")
    plt.close()


def analysis_learning(train_sizes, train_scores, test_scores, path_dir):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.ion()
    fig = plt.figure(figsize=(8,6))
    title_size = fig.get_figwidth() * 1
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Validation Score")
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.1, color="green")
    plt.title("Learning Curves (SVM)", fontsize=title_size)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(path_dir, "learning_curve.png"), bbox_inches="tight")
    plt.close()


def analysis_kernels(x, y, path_dir):
    print("--- Starting kernel analysis ---")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    results = {}
    best_params = {}

    ### Linear kernel ###
    param_grid_linear = {"kernel": ["linear"],
                         "C": [0.01, 0.1, 1, 10, 100]}
    model_linear = GridSearchCV(SVC(), param_grid_linear, cv=5)
    model_linear.fit(x_train, y_train)
    results["Linear"] = model_linear.score(x_test, y_test)
    best_params["Linear"] = model_linear.best_params_

    ### RBF kernel ###
    param_grid_rbf = {"kernel": ["rbf"],
                          "C": [0.01, 0.1, 1, 10, 100], "gamma": ["scale", "auto"]}
    model_rbf = GridSearchCV(SVC(), param_grid_rbf, cv=5)
    model_rbf.fit(x_train, y_train)
    results["RBF"] = model_rbf.score(x_test, y_test)
    best_params["RBF"] = model_rbf.best_params_

    ### Sigmoid kernel ###
    param_grid_sig = {"kernel": ["sigmoid"],
                      "C": [0.01, 0.1, 1, 10, 100], "gamma": ["scale", "auto"]}
    model_sig = GridSearchCV(SVC(), param_grid_sig, cv=5)
    model_sig.fit(x_train, y_train)
    results["Sigmoid"] = model_sig.score(x_test, y_test)
    best_params["Sigmoid"] = model_sig.best_params_

    ### Poly kernel ###
    param_grid_poly = {"kernel": ["poly"],
                       "C": [0.01, 0.1, 1, 10, 100], "gamma": ["scale", "auto"],
                       "degree": [1, 2, 3, 4]}
    model_poly = GridSearchCV(SVC(), param_grid_poly, cv=5)
    model_poly.fit(x_train, y_train)
    results["Poly"] = model_poly.score(x_test, y_test)
    best_params["Poly"] = model_poly.best_params_

    ### All ###
    param_grid = [param_grid_linear, param_grid_rbf, param_grid_sig, param_grid_poly]
    grid = GridSearchCV(SVC(), param_grid, cv=5)
    grid.fit(x_train, y_train)

    print("--- Finished kernel analysis ---")

    # Plot results
    plt.ion()
    fig = plt.figure(figsize=(10,6))
    title_size = fig.get_figwidth() * 1
    bars = plt.bar(results.keys(), results.values())
    plt.ylabel("Accuracy")
    plt.title(f"SVM Kernel Vergleich", fontsize=title_size)
    plt.ylim(0, 1)

    # Scores over bars
    for bar, score in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,f"{score:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Parametertext
    param_text = ( f"Linear Kernel:\n{best_params['Linear']}"
                   f"\n\nRBF Kernel:\n{best_params['RBF']}"
                   f"\n\nSigmoid Kernel:\n{best_params['Sigmoid']}"
                   f"\n\nPoly Kernel:\n{best_params['Poly']}"
                   f"\n\n#1 Kernel:\n{grid.best_params_}")

    plt.text(1.05, 0.5, param_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment="center",
             bbox=dict(facecolor="white", alpha=0.8))

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, f"vergleich_allCabs_kernel_analysis.png"), bbox_inches="tight")
    plt.close()


def analysis_sts(path_dir, path_train, embedding, persistent_dir):

    # Load excel file
    if os.path.exists(path_train):
        reader = pd.read_excel(path_train, engine='openpyxl')
    else:
        print(f"!!! File {path_train} not found")
        return

    positions = []

    for i, line in reader.iterrows():
        text = str(line["Text"])
        correct_label = line["Label"]
        cab = line["Cab"]

        text = re.sub(r"[^a-zA-Z0-9]", " ", text.strip().lower())
        text = f"{text}. In {cab}"

        persistent_dir_cabx = select_cab(cab, persistent_dir)

        # Calculate STS score
        results = []

        if persistent_dir_cabx:
            results = calc_similarity(text, persistent_dir_cabx, embedding, 1000)
        else:
            print(f"!!! Matching vector-db not found with cab={cab} !!!")
            return -1


        retrieved_ids = []
        for doc in results:
            id = doc[0].metadata["id"]
            retrieved_ids.append(id)

        if correct_label in retrieved_ids:
            pos = retrieved_ids.index(correct_label) + 1
        else:
            pos = None
        positions.append(pos)

    # Plot data
    top_k = 15
    normalized = [p if p is not None else top_k + 1 for p in positions]

    counts = Counter(normalized)
    labels = list(range(1, top_k + 2))
    values = [counts.get(pos, 0) for pos in labels]
    x_labels = [str(i) for i in range(1, top_k + 1)] + ["Nicht gefunden"]
    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, values, color="steelblue")
    plt.title("Analyse Top-k STS-Ergebnissen")
    plt.xlabel("Trefferposition")
    plt.ylabel("Anzahl")
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, f"vergleich_allCabs__sts_analysis.png"), bbox_inches="tight")
    plt.close()