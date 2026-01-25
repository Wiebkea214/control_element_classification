import os
import re

from evaluation import *
from setup_vector_database import *
from feature_vector import *

import pandas as pd
import numpy as np
import joblib
import psutil
import threading
import time
import winsound

from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance

###########################################################################

def get_traindata(path_train, persistent_dir, embedding, k, feat):
    col_text = 'Text'
    col_label = 'Label'
    col_cab = 'Cab'

    x = []
    y = []
    y_sts = []

    sts_time = []
    sts_mem = []
    dim = 0

    # Load excel file
    print(f"\n--- Reading training data in {path_train} ---")
    if os.path.exists(path_train):
        reader = pd.read_excel(path_train, engine='openpyxl')
    else:
        print(f"!!! File {path_train} not found")
        return

    print(f"\n--- Organize training data and labels in x and y lists ---")
    for i, line in reader.iterrows():
        text = str(line[col_text])
        label = line[col_label]
        cab = line[col_cab]

        # Compute feature vector
        features, top1, dim, sts_time, sts_mem = build_feature_vector(embedding, persistent_dir, text, cab, k, feat)

        y_sts.append(top1.metadata["id"])
        x.append(features)
        y.append(label)

    x = np.array(x)
    y = np.array(y)
    y_sts = np.array(y_sts)

    print(f"\n--- Finished preparation of training data ---")

    return x, y, y_sts, sts_time, sts_mem, dim


def train_svm(x, y, kernel, eval_dir):
    print("\n--- Start training of SVM ---")
    encoder = LabelEncoder()
    scaler = StandardScaler()
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Split training and test data
    x_train, x_test, y_train_str, y_test_str = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    # Convert string classes to numeric classes
    encoder.fit(y_train_str)
    y_test_num = encoder.transform(y_test_str)
    y_train_num = encoder.transform(y_train_str)
    print("y_train_str: " + str(np.unique(y_train_str, return_counts=True)) + "\ny_train_num: " + str(np.unique(y_train_num, return_counts=True)))

    # Normalize vector
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # SVM setup (RBF-Kernel is standard)
    if kernel == "poly":
        svm = SVC(kernel="poly", C=90, degree=1, gamma="scale", probability=True, random_state=42)
    else:
        svm = SVC(kernel="linear", C=10, probability=True, random_state=42)

    # Training
    svm.fit(x_train_scaled, y_train_str)

    # Save model
    joblib.dump(svm, f"svm_model_allCabs_{time_now}.joblib")
    joblib.dump(encoder, f"encoder_allCabs_{time_now}.joblib")
    print("\n--- Finished training and saving of SVM ---")

    # Prediction with test data
    print("\n--- Start evaluation of model with test data ---")
    y_score = svm.decision_function(x_test_scaled)
    y_pred = np.argmax(y_score, axis=1)

    report = classification_report(y_test_num, y_pred, zero_division=0, target_names=encoder.classes_)
    print(report)


def evaluate_svm(x, y, eval_dir, kernel):
    print("\n--- Start training of SVM ---")
    encoder = LabelEncoder()
    scaler = StandardScaler()
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Split training and test data
    x_train, x_test, y_train_str, y_test_str = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    encoder.fit(y_train_str)
    y_test_num = encoder.transform(y_test_str)
    y_train_num = encoder.transform(y_train_str)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("y_train_str: " + str(np.unique(y_train_str, return_counts=True)) + "\ny_train_num: " + str(np.unique(y_train_num, return_counts=True)))

    # SVM setup (SVC for classification, RBF-Kernel is standard)
    if kernel == "poly":
        svm = SVC(kernel="poly", C=90, degree=1, gamma="scale", probability=True, random_state=42)
    else:
        svm = SVC(kernel="linear", C=10, probability=True, random_state=42)

    ############# Set preconditions for time and load analysis ##################
    process = psutil.Process()
    cpu_usage = []
    timestamps = []
    interval = 0.01
    delay = 0.5
    stop_event = threading.Event()
    cpu_start = time.perf_counter()
    ###
    monitor_thread = threading.Thread(target=monitor_cpu, args=(interval, cpu_usage, stop_event, timestamps, cpu_start))
    monitor_thread.start()
    ###
    time.sleep(delay)
    mem_train_before = process.memory_info().rss / (1024*1024)
    train_start = time.perf_counter()
    #############################################################################

    # Training
    svm.fit(x_train_scaled, y_train_str)

    ############# Get results for time and load analysis ########################
    train_end = time.perf_counter()
    mem_train_after = process.memory_info().rss / (1024 * 1024)
    #############################################################################

    print("\n--- Finished training of SVM ---")

    ############# Set preconditions for inference analysis ######################
    mem_pred_before = process.memory_info().rss / (1024 * 1024)
    pred_start = time.perf_counter()
    #############################################################################

    # Prediction with test data
    print("\n--- Start evaluation of model with test data ---")
    y_score = svm.decision_function(x_test_scaled)
    y_pred = np.argmax(y_score, axis=1)

    ############## Training evaluation ##########################################
    pred_end = time.perf_counter()
    mem_pred_after = process.memory_info().rss / (1024 * 1024)
    time.sleep(delay)
    stop_event.set()
    monitor_thread.join()
    acc = accuracy_score(y_test_num, y_pred)
    report = classification_report(y_test_num, y_pred, zero_division=0, target_names=encoder.classes_)
    train_sizes, train_scores, test_scores = learning_curve(SVC(kernel='linear'), x, y, cv=5,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_time = train_end - train_start
    pred_time = pred_end - pred_start
    mem_train = mem_train_after - mem_train_before
    mem_pred = mem_pred_after - mem_pred_before

    ### Preparation evaluation results
    if os.path.exists(eval_dir):
        pass
    else:
        os.makedirs(eval_dir)

    ### Perform evaluation SVM
    cross_val = np.mean(cross_val_score(svm, x_train_scaled, y_train_str, cv=5))
    r = permutation_importance(svm, x_train_scaled, y_train_str, n_repeats=30, random_state=0)
    analysis_cpu_usage(interval, train_start, train_end, pred_start, pred_end, cpu_start, cpu_usage, timestamps, eval_dir)
    analysis_performance(y_test_num, y_pred, encoder, eval_dir)
    analysis_conf_matrix(y_test_num, y_pred, encoder, eval_dir, "confusion_matrix_svm.png")
    analysis_learning(train_sizes, train_scores, test_scores, eval_dir)
    ###################################################################################

    return acc, report, train_time, pred_time, mem_train, mem_pred, cross_val, r
