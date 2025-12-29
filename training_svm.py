import os
import re

from evaluation import *
from similarity_by_db import *

import pandas as pd
import numpy as np
import joblib
import psutil
import threading
import time

from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


###########################################################################

def get_traindata(path_train, persistent_dir, embedding, cab_train):
    pers_dir_cab1 = persistent_dir[0]
    pers_dir_cab2 = persistent_dir[1]

    col_text = 'Text'
    col_label = 'Label'
    col_cab = 'Cab'

    sts_time = []
    sts_mem = []
    process = psutil.Process()

    x = []
    y = []
    y_sts = []

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
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.strip().lower())
        text_emb = embedding.embed_query(str(text))
        label = line[col_label]
        cab = line[col_cab]

        # Calculate STS score
        candidates_top5 = []
        found = True

        mem_sts_before = process.memory_info().rss / (1024 * 1024)
        sts_start = time.perf_counter()
        if (cab == 'cab1' or cab == 'no cab') and cab_train == 'cab1':
            candidates_top5 = calc_similarity(text, pers_dir_cab1, embedding)
        elif (cab == 'cab2' or cab == 'no cab') and cab_train == 'cab2':
            candidates_top5 = calc_similarity(text, pers_dir_cab2, embedding)
        else:
            found = False
        sts_end = time.perf_counter()
        mem_sts_after = process.memory_info().rss / (1024 * 1024)
        sts_time.append(sts_end - sts_start)
        sts_mem.append(mem_sts_before - mem_sts_after)

        features = []
        sts_all = []
        cosine_all = []
        last_sts = 0
        sts_diff = 0

        if found:
            for sts_candidate, sts_score in candidates_top5:
                sts_emb = embedding.embed_query(str(sts_candidate.page_content))
                cosine_score = cosine_similarity([text_emb], [sts_emb])[0][0]
                cosine_scaled = (cosine_score + 1)/2
                if last_sts:
                    sts_diff = sts_score - last_sts
                else:
                    last_sts = sts_score

                if sts_diff:
                    features.append(sts_diff)
                else:
                    pass

                features.extend([sts_score, cosine_scaled])
                sts_all.append(sts_score)
                cosine_all.append(cosine_scaled)

            mean_sts = np.mean(sts_all, axis=0)
            mean_cos = np.mean(cosine_all, axis=0)
            var_sts = np.var(sts_all, axis=0)
            var_cos = np.var(cosine_all, axis=0)
            features.extend([mean_sts, mean_cos, var_sts, var_cos])

            top1 = candidates_top5[0][0]
            y_sts.append(top1.metadata["id"])
            x.append(features)
            y.append(label)

    x = np.array(x)
    y = np.array(y)
    y_sts = np.array(y_sts)

    print(f"\n--- Finished preparation of training data ---")

    return x, y, y_sts, sts_time, sts_mem


def train_svm(x, y, cab, eval_dir):
    print("\n--- Start training of SVM ---")
    encoder = LabelEncoder()
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Split training and test data
    x_train, x_test, y_train_str, y_test_str = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    encoder.fit(y_train_str)
    y_test_num = encoder.transform(y_test_str)
    y_train_num = encoder.transform(y_train_str)

    print("y_train_str: " + str(np.unique(y_train_str, return_counts=True)) + "\ny_train_num: " + str(np.unique(y_train_num, return_counts=True)))

    # SVM setup (RBF-Kernel is standard)
    svm = SVC(kernel="rbf", probability=True)

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
    svm.fit(x_train, y_train_str)

    ############# Get results for time and load analysis ########################
    train_end = time.perf_counter()
    mem_train_after = process.memory_info().rss / (1024 * 1024)
    #############################################################################

    joblib.dump(svm, f"svm_model_{time_now}.joblib")
    print("\n--- Finished training of SVM ---")

    ############# Set preconditions for inference analysis ######################
    mem_pred_before = process.memory_info().rss / (1024 * 1024)
    pred_start = time.perf_counter()
    #############################################################################

    # Prediction with test data
    print("\n--- Start evaluation of model with test data ---")
    y_score = svm.decision_function(x_test)
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
    analysis_cpu_usage(interval, train_start, train_end, pred_start, pred_end, cpu_start, cpu_usage, timestamps, eval_dir)
    analysis_performance(y_test_num, y_pred, encoder, eval_dir)
    analysis_conf_matrix(y_test_num, y_pred, encoder, eval_dir, "confusion_matrix_svm.png")
    analysis_learning(train_sizes, train_scores, test_scores, eval_dir)
    ###################################################################################

    return acc, report, train_time, pred_time, mem_train, mem_pred
