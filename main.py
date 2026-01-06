from preprocessing_data import *
from setup_vector_database import *
from write_data_in_excel import *
from training_svm import *
from training_data_augmentation import *
from gather_information import *
from feature_vector import *

from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
import winsound

########################## Init #############################

#if __name__ == "__main__":
def main(cab, k, feat, path_train, dir_name, config):
    import multiprocessing
    multiprocessing.freeze_support()  # important for Windows

#################### Interface ##############################

    #cab = "cab2"
    #k = 5
    #feat = 9
    test_step = [""]
    path_fts = "F:\\OneDrive\\Masterarbeit\\FTS Daten\\TRAXX_AC3_ISR__Control_front_and_tail_lights.xlsx"
    path_bmv_cab1 = "F:\\OneDrive\\Masterarbeit\\FTS Daten\\Labels\\BMV_Labels_cab1_top14.xlsx"
    path_bmv_cab2 = "F:\\OneDrive\\Masterarbeit\\FTS Daten\\Labels\\BMV_Labels_cab2_top14.xlsx"
    #path_train = f"F:\\OneDrive\\Masterarbeit\\FTS Daten\\Training\\TRAXX_AC3_Training_{cab}_7class_cnt100.xlsx"
    #dir_name = f"evaluation_{cab}_top{k}_7class_{feat}feat_scaled_cnt100"
    gather_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation auto"))
    eval_dir = str(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation auto"), dir_name))

    # Available configs: load_fts, edit_db, similarity, train_svm, train_svm_only, evaluate
    #config = "train_svm, evaluate"

####################### Globals #############################

    current_dir = os.path.dirname(os.path.abspath(__file__))
    store_name1 = "chroma_db_cab1"
    store_name2 = "chroma_db_cab2"
    persistent_dir_cab1 = str(os.path.join(current_dir, store_name1))
    persistent_dir_cab2 = str(os.path.join(current_dir, store_name2))
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

######################## Main ###############################

    # Preprocess BMV
    if "edit_db" in config:
        docs_bmv_cab1 = get_BMV(path_bmv_cab1)
        docs_bmv_cab2 = get_BMV(path_bmv_cab2)

        # Create persistent storage/ embedding database from BMV
        edit_vector_db(docs_bmv_cab1, store_name1, persistent_dir_cab1, embedding_model)
        edit_vector_db(docs_bmv_cab2, store_name2, persistent_dir_cab2, embedding_model)

    # Preprocess FTS
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
            for id, test_step in dict_fts.items():
                result = calc_similarity(test_step, persistent_dir_cab1, embedding_model)

                # Write result in excel file
                data = [id, test_step, result[0][0], result[1][0], result[2][0], result[0][1], result[1][1], result[2][1]]
                write_excel(headers=headers, data=data, path=path)

    if "train_svm" in config:
        txt = []
        x, y, y_sts, sts_time, sts_mem, dim = get_traindata(path_train,[persistent_dir_cab1, persistent_dir_cab2], embedding_model, cab, k, feat)

        if "train_svm_only" in config:
            train_svm(x, y, cab, eval_dir)

        if "evaluate" in config:
            # SVM analysis
            svm_acc, report, train_time, pred_time, mem_train, mem_pred, cross_val, r = evaluate_svm(x, y, cab, eval_dir)
            txt.append("SVM:" +
                    "\ncross validation score: " + str(round(cross_val*100, 2)) + " %" +
                    "\naccuracy SVM: " + str(round(svm_acc*100, 2)) + " %" +
                    "\ntrain time: " + str(round(train_time * 1000, 2)) + " ms" +
                    "\ntrain RAM: " + str(round(mem_train, 2)) + " MB" +
                    "\ninference time: " + str(round(pred_time * 1000, 2)) + " ms" +
                    "\ninference RAM: " + str(round(mem_pred, 2)) + " MB" +
                    "\nvector dimensions: " + str(dim) + " Features" +
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

            # Gather information for evaluation
            gather_pictures("_scaled_", gather_path, cab)
            gather_log("_scaled_", "cross validation score", gather_path, cab)
            gather_log("_scaled_", "accuracy SVM", gather_path, cab)
            gather_log("_scaled_", "train time", gather_path, cab)
            gather_log("_scaled_", "train RAM", gather_path, cab)
            gather_log("_scaled_", "inference time", gather_path, cab)
            gather_log("_scaled_", "inference RAM", gather_path, cab)
            gather_log("_scaled_", "vector dimensions", gather_path, cab)

    if "predict" in config:
        svm_model = joblib.load("svm_model_{cab}_{time_now}.joblib")
        label_encoder = joblib.load("encoder_{cab}_{time_now}.joblib")
        features, top1, dim, sts_time, sts_mem = build_feature_vector(embedding_model, [persistent_dir_cab1, persistent_dir_cab2], test_step, cab, k, feat)

        y_pred_encoded = svm_model.predict(features)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        print(f"\nSVM prediction: {y_pred[0]}")

    winsound.Beep(600, 500)
    print("--- Finished ---")

#######################################################################################################

# Automatic evaluation execution
cabs = ["cab2"]
top_xs = [3, 5, 7]
feats = [2, 6, 8, 9]
config_x = "train_svm, evaluate"

for cab_x in cabs:
    for top_x in top_xs:
        for feat_x in feats:
            path_train_x = f"F:\\OneDrive\\Masterarbeit\\FTS Daten\\Training\\TRAXX_AC3_Training_{cab_x}_7class_cnt100.xlsx"
            dir_name_x = f"evaluation_{cab_x}_top{top_x}_7class_{feat_x}feat_scaled_cnt100"
            main(cab_x, top_x, feat_x, path_train_x, dir_name_x, config_x)