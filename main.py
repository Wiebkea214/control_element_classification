from preprocessing_data import *
from setup_vector_database import *
from similarity_by_db import *
from write_data_in_excel import *
from training_svm import *
from training_data_augmentation import *

from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

########################## Init #############################

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # important for Windows

#################### Interface (lvl1) #######################

    path_fts = "F:\\OneDrive\\Masterarbeit\\FTS Daten\\TRAXX_AC3_ISR__Control_front_and_tail_lights.xlsx"
    path_bmv_cab1 = "F:\\OneDrive\\Masterarbeit\\FTS Daten\\Labels\\BMV_Labels_cab1_top14.xlsx"
    path_bmv_cab2 = "F:\\OneDrive\\Masterarbeit\\FTS Daten\\Labels\\BMV_Labels_cab2_top14.xlsx"
    path_train = "F:\\OneDrive\\Masterarbeit\\FTS Daten\\Training\\TRAXX_AC3_Training_cab1_top7_cnt100.xlsx"
    dir_name = f"evaluation_cab1_top5_7class_18feat_cnt100"
    eval_dir = str(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation"), dir_name))
    cab = "cab1"

####################### Globals #############################

    current_dir = os.path.dirname(os.path.abspath(__file__))
    store_name1 = "chroma_db_cab1"
    store_name2 = "chroma_db_cab2"
    persistent_dir_cab1 = str(os.path.join(current_dir, store_name1))
    persistent_dir_cab2 = str(os.path.join(current_dir, store_name2))
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Available configs: load_fts, edit_db, similarity, train_svm
    config = "train_svm"

######################## Main ###############################

    # Preprocess training data
    if "train_data_augmentation" in config:
        training_data_augmentation(path_train, embedding_model, max_class_cnt=50)

    # Preprocess BMZ
    if "edit_db" in config:
        dict_bmv_cab1, docs_bmv_cab1 = get_BMV(path_bmv_cab1)
        dict_bmv_cab2, docs_bmv_cab2 = get_BMV(path_bmv_cab2)

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
        x, y, y_sts, sts_time, sts_mem = get_traindata(path_train,[persistent_dir_cab1, persistent_dir_cab2], embedding_model, cab)
        svm_acc, report, train_time, pred_time, mem_train, mem_pred = train_svm(x, y, cab, eval_dir)
        txt.append("SVM:\n"
                "acc: " + str(round(svm_acc, 3)) +
                "\ntrain time: " + str(round(train_time * 1000, 2)) + " ms"
                "\ntrain RAM: " + str(round(mem_train, 2)) + " MB"
                "\ninference time: " + str(round(pred_time * 1000, 2)) + " ms"
                "\ninference RAM: " + str(round(mem_pred, 2)) + " MB"
                "\nreport:" + report)

        # STS Analysis
        analysis_conf_matrix(y, y_sts, encoder=0, path_dir=eval_dir, filename="confusion_matrix_sts.png")
        avg_time = sum(sts_time) / len(sts_time)
        avg_mem = sum(sts_mem) / len(sts_mem)
        sts_acc = accuracy_score(y, y_sts)
        txt.append(f"STS:\n"
              "acc: " + str(round(sts_acc, 3)) +
              "\naverage time: " + str(round(avg_time*1000, 2)) + " ms"
              "\naverage RAM: " + str(round(avg_mem, 2)) + " MB")

        with open(os.path.join(eval_dir, "evaluation_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(txt))
        print(txt)

    print("--- Finished ---")
