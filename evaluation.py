import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import psutil
import os
import time

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from scipy.stats import uniform

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

    # Diagramm anzeigen
    plt.show()


def monitor_cpu(interval, cpu_usage, stop_event, timestamps, start_time):
    # Calibration
    psutil.cpu_percent(interval=None)

    # Actual measuring
    while not stop_event.is_set():
        cpu_usage.append(psutil.cpu_percent(interval=interval))
        timestamps.append(time.perf_counter() - start_time)

    return cpu_usage, timestamps


def analysis_cpu_usage(interval, train_start, train_end, pred_start, pred_end, cpu_start, cpu_usage, timestamps, path_dir):
    n = len(cpu_usage)
    training_start = train_start - cpu_start
    training_end = train_end - cpu_start
    prediction_start = pred_start - cpu_start
    prediction_end = pred_end - cpu_start

    colors = ['r' if training_start <= t <= training_end else 'b' for t in timestamps]

    plt.ion()
    plt.figure(figsize=(10,6))
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

    plt.title("CPU-load during training")
    plt.xlabel(f"measure points (all {interval*1000}ms)")
    plt.ylabel("CPU-load (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path_dir, "cpu_usage.png"))


def analysis_performance(y_test, y_pred, encoder, path_dir):
    # Performance data
    class_names = encoder.classes_

    # Precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    x = np.arange(len(precision))

    # Plots
    plt.ion()
    plt.figure(figsize=(10, 6))
    plt.bar(x, precision, width=0.25, label="Precision")
    plt.bar([i + 0.25 for i in x], recall, width=0.25, label="Recall")
    plt.bar([i + 0.50 for i in x], f1, width=0.25, label="F1-Score")
    plt.xticks([i + 0.25 for i in x], class_names, rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Value")
    plt.title("Precision, Recall and F1 per class")
    plt.legend()
    plt.savefig(os.path.join(path_dir, "performance.png"))


def analysis_conf_matrix(y_test, y_pred, encoder, path_dir, filename):
    if encoder:
        class_names = encoder.classes_
    else:
        class_names = np.array(list(set(y_pred)))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted classes")
    plt.ylabel("True classes")
    plt.savefig(os.path.join(path_dir, filename))


def analysis_learning(train_sizes, train_scores, test_scores, path_dir):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.ion()
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Validation Score")
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.1, color="green")
    plt.title("Learning Curves (SVM)")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(path_dir, "learning_curve.png"))


def analysis_kernels(x, y, path_dir, method="grid"):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    results = {}
    best_params = {}

    ### Linear kernel ###
    if method == "grid":
        param_grid_linear = {"C": [0.1, 1, 10, 100]}
        model_linear = GridSearchCV(SVC(kernel="linear"), param_grid_linear, cv=5)
    else:
        param_dist_linear = {"C": uniform(0.1, 100)}
        model_linear = RandomizedSearchCV(SVC(kernel="linear"), param_dist_linear, n_iter=10, cv=5)

    model_linear.fit(x_train, y_train)
    acc_linear = model_linear.score(x_test, y_test)
    results["Linear"] = acc_linear
    best_params["Linear"] = model_linear.best_params_

    ### RBF kernel ###
    if method == "grid":
        param_grid_rbf = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]}
        model_rbf = GridSearchCV(SVC(kernel="rbf"), param_grid_rbf, cv=5)
    else:
        param_dist_rbf = {"C": uniform(0.1, 100), "gamma": uniform(0.0001, 1)}
        model_rbf = RandomizedSearchCV(SVC(kernel="rbf"), param_dist_rbf, n_iter=20, cv=5)

    model_rbf.fit(x_train, y_train)
    acc_rbf = model_rbf.score(x_test, y_test)
    results["RBF"] = acc_rbf
    best_params["RBF"] = model_rbf.best_params_

    # Plot results
    plt.ion()
    plt.figure(figsize=(10,6))
    bars = plt.bar(results.keys(), results.values(), color=["steelblue","orange"])
    plt.ylabel("Accuracy")
    plt.title(f"SVM Kernel Vergleich ({method.capitalize()} Search)")
    plt.ylim(0, 1)

    # Scores over bars
    for bar, score in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,f"{score:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Parametertext
    param_text = ( f"Linear Kernel:\n{best_params['Linear']}"
                   f"\n\nRBF Kernel:\n{best_params['RBF']}" )

    plt.text(1.05, 0.5, param_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment="center",
             bbox=dict(facecolor="white", alpha=0.8))

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, f"vergleich_allCabs_kernel_analysis.png"))