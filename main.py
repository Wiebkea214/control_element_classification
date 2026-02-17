from preprocessing_data import *
from setup_vector_database import *
from write_data_in_excel import *
from training_svm import *
from training_data_augmentation import *
from gather_information import *
from feature_vector import *

from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
import matplotlib.pyplot as plt
import winsound

########################## Init #############################

#if __name__ == "__main__":
def main(cab, k, feat, kernel, c, path_train, dir_name, config, test_step):
    import multiprocessing
    multiprocessing.freeze_support()  # important for Windows

#################### Interface ##############################

    # FTS file for STS pre-analysis
    path_fts = "F:\\OneDrive\\Masterarbeit\\FTS Daten\\TRAXX_AC3_ISR__Control_front_and_tail_lights.xlsx"

    # BMV
    bmv_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "BMV"))
    path_bmv_cab1 = os.path.join(bmv_dir,"BMV_Labels_cab1_14class.xlsx")
    path_bmv_cab2 = os.path.join(bmv_dir,"BMV_Labels_cab2_14class.xlsx")

    # SVM
    svm_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SVM Models"))

    # Evaluation
    evaluation_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation"))
    gather_path = os.path.join(evaluation_dir, "Evaluation auto")
    top_k_path = os.path.join(evaluation_dir, "Evaluation Top-k")
    eval_dir = str(os.path.join(os.path.join(evaluation_dir, "Evaluation auto"), dir_name))

####################### Globals #############################

    current_dir = os.path.dirname(os.path.abspath(__file__))
    store_name1 = "chroma_db_cab1"
    store_name2 = "chroma_db_cab2"
    persistent_dir_cab1 = str(os.path.join(current_dir, store_name1))
    persistent_dir_cab2 = str(os.path.join(current_dir, store_name2))
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

######################## Main ###############################

    # Preprocess BMV and edit vektor-database
    if "edit_db" in config:
        docs_bmv_cab1 = get_BMV(path_bmv_cab1)
        docs_bmv_cab2 = get_BMV(path_bmv_cab2)

        # Create persistent storage/ embedding database from BMV
        edit_vector_db(docs_bmv_cab1, store_name1, persistent_dir_cab1, embedding_model)
        edit_vector_db(docs_bmv_cab2, store_name2, persistent_dir_cab2, embedding_model)

    # Preprocess FTS for STS-analysis (not part of normal operation)
    dict_fts = {}
    if "load_fts" in config:
        dict_fts = get_FTS(path_fts)

        # Get control element by similarity from db
        headers = ["ID",
                   "Test Step",
                   "Result 1",
                   "Result 2",
                   "Result 3",
                   "Score 1",
                   "Score 2",
                   "Score 3"]
        path = str(os.path.join(current_dir, "Output", f"Similarity_Results_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.xlsx"))
        init_excel(headers=headers, path=path)

        if "similarity" in config:
            for id, text in dict_fts.items():
                result = calc_similarity(text, persistent_dir_cab1, embedding_model)

                # Write result in excel file
                data = [id, text, result[0][0], result[1][0], result[2][0], result[0][1], result[1][1], result[2][1]]
                write_excel(headers=headers, data=data, path=path)

    # Train and evaluate model
    if "train_svm" in config:
        txt = []
        x, y, y_sts, sts_time, sts_mem, dim = get_traindata(path_train,[persistent_dir_cab1, persistent_dir_cab2], embedding_model, k, feat)

        if "train_svm_only" in config:
            train_svm(x, y, svm_dir)

        if "evaluate_model" in config:
            # SVM analysis

            data = {
                "cross_val" : [],
                "svm_val" : [],
                "train_time" : [],
                "mem_train" : [],
                "pred_time" : [],
                "mem_pred" : [],
                "dim" : []
            }
            report = ""

            for i in range(1):
                svm_acc, report, train_time, pred_time, mem_train, mem_pred, cross_val, r = evaluate_svm(x, y, eval_dir, kernel, c)
                data["cross_val"].append(cross_val)
                data["svm_val"].append(svm_acc)
                data["train_time"].append(train_time)
                data["pred_time"].append(pred_time)
                data["mem_train"].append(mem_train)
                data["mem_pred"].append(mem_pred)
                data["dim"].append(dim)

            means = {key: sum(values) / len(values) for key, values in data.items()}

            txt.append("SVM:" +
                    "\ncross validation score: " + str(round(means["cross_val"]*100, 2)) + " %" +
                    "\naccuracy SVM: " + str(round(means["svm_val"]*100, 2)) + " %" +
                    "\ntrain time: " + str(round(means["train_time"] * 1000, 2)) + " ms" +
                    "\ntrain RAM: " + str(round(means["mem_train"], 2)) + " MB" +
                    "\ninference time: " + str(round(means["pred_time"] * 1000, 2)) + " ms" +
                    "\ninference RAM: " + str(round(means["mem_pred"], 2)) + " MB" +
                    "\nvector dimensions: " + str(means["dim"]) + " Features" +
                    "\nreport:" + report)
            print(report)

            # STS Analysis
            analysis_conf_matrix(y, y_sts, encoder=0, path_dir=eval_dir, filename="confusion_matrix_sts.png")
            avg_time = sum(sts_time) / len(sts_time)
            avg_mem = sum(sts_mem) / len(sts_mem)
            sts_acc = accuracy_score(y, y_sts)
            txt.append("STS:" +
                  "\naccuracy STS: " + str(round(sts_acc, 3)) +
                  "\naverage time: " + str(round(avg_time*1000, 2)) + " ms" +
                  "\naverage RAM: " + str(round(avg_mem, 2)) + " MB")

            with open(os.path.join(eval_dir, "evaluation_log.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(txt))

            winsound.Beep(600, 500)

        if "evaluate_kernel" in config:
            analysis_kernels(x, y, gather_path)

        if "evaluate_sts" in config:
            analysis_sts(gather_path, path_train, embedding_model, [persistent_dir_cab1, persistent_dir_cab2])

    # Gather information for evaluation
    if "gather_information" in config:
        if "log" in config:
            keyword = "cnt150"
            addition = ""
            gather_pictures(keyword, addition, gather_path)
            gather_log(keyword, "cross validation score", addition, gather_path)
            gather_log(keyword, "accuracy SVM", addition, gather_path)
            gather_log(keyword, "train time", addition, gather_path)
            gather_log(keyword, "train RAM", addition, gather_path)
            gather_log(keyword, "inference time", addition, gather_path)
            gather_log(keyword, "inference RAM", addition, gather_path)
            gather_log(keyword, "vector dimensions", addition, gather_path)

        if "top-k" in config:
            gather_top_k(top_k_path)


    # Normal operation prediction
    prediction = ""
    if "predict" in config:
        # Check if element nr. is used in test step
        persistent_dir_cabx = select_cab(cab, [persistent_dir_cab1, persistent_dir_cab2])
        prediction = element_precheck(test_step, persistent_dir_cabx)

        # If not, perform classification
        if prediction:
            print(f"\nElement found in test step: {prediction}")
        else:
            svm_model = joblib.load("svm_model.joblib")
            features, top1, dim, sts_time, sts_mem = build_feature_vector(embedding_model, [persistent_dir_cab1, persistent_dir_cab2], test_step, cab, k, feat)

            prediction = svm_model.predict([features])[0]

            # Confidence
            scores = svm_model.decision_function([features])[0]
            softmax = np.exp(scores) / np.sum(np.exp(scores))
            confidence = softmax.max()

            # Human-in-the-Loop
            if (top1.metadata["id"] == prediction) and confidence >= 0.5:
                print(f"\nSVM prediction is valid: {prediction}")

            else:
                print(f"\nSVM prediction is unsure: prediction={prediction} vs. STS={top1.metadata['id']} and confidence = {confidence}")

    plt.close()
    print("--- Finished ---")

    return prediction

#######################################################################################################
#######################################################################################################

evaluation = 0

if evaluation:
    # Automatic evaluation execution
    top_xs = [3, 5, 7, 10, 10000]
    feats = [0, 9]
    classes = [4,8,12,16,20,24,28]
    kernels = ["linear", "poly"]

    # Available configs: load_fts, edit_db, similarity, train_svm, train_svm_only, evaluate
    config_x = "train_svm, evaluate_model"

    # Current values
    cab_x = ""
    class_x = 28
    top_x = 5
    feat_x = 9
    cnt = 250
    kernel_x = "linear"
    c_x = 10

    path_train_x = f"F:\\OneDrive\\Masterarbeit\\FTS Daten\\Training\\TRAXX_AC3_Training_allCabs_{class_x}class_cnt{cnt}.xlsx"
    dir_name_x = f"evaluation_allCabs_top{top_x}_{class_x}class_{feat_x}feat_{kernel_x}Kernel_cnt{cnt}"
    print(f"----- Start with param feat_x={feat_x}, top_xs={top_x}, class={class_x}, c={c_x}, cnt={cnt} -----")

    main(cab_x, top_x, feat_x, kernel_x, c_x, path_train_x, dir_name_x, config_x)

else:
    # Normal operation parameters
    project = "AC3"
    class_x = 28
    cnt = 250

    # Interface parameters
    cab_x = "cab1"
    config_x = "predict"
    test_step_x = "Switch on the battery"

    # Paths
    dir_train = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train data"))
    path_train = dir_train.joinpath(dir_train, f"TRAXX_{project}_Training_allCabs_{class_x}class_cnt{cnt}.xlsx")

    main(cab_x, k = 5, feat = 9, kernel = "linear", c = 10, path_train = path_train, dir_name = "", config = config_x, test_step = test_step_x)