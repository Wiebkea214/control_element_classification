import winsound

from project.functions__write_data_in_excel import *
from project.functions__training_svm import *
from project.functions__gather_information import *
from project.functions__feature_vector import *
from project.functions__human_in_the_loop import *
from project.functions__prediction import *

from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

########################## Main #############################

def main(cab, k, feat, kernel, c, path_train, dir_name, config, test_step):
    import multiprocessing
    multiprocessing.freeze_support()  # important for Windows

    #################### Globals ##############################

    base_dir = Path(__file__).resolve().parent

    # BMV
    bmv_dir = base_dir / 'BMV'
    path_bmv_cab1 = bmv_dir / "BMV_Labels_cab1_14class.xlsx"
    path_bmv_cab2 = bmv_dir / "BMV_Labels_cab2_14class.xlsx"

    # Evaluation
    evaluation_dir = base_dir / "Evaluation"
    gather_path = evaluation_dir / "Evaluation auto"
    top_k_path = evaluation_dir / "Evaluation Top-k"
    eval_dir = evaluation_dir / "Evaluation auto" / dir_name

    #Vector Database
    persistent_dir_cab1 = str(base_dir / "chroma_db_cab1")
    persistent_dir_cab2 = str(base_dir / "chroma_db_cab2")

    # Sentence Transformer
    model_path = str(base_dir / "Sentence Transformer" / "all-MiniLM-L6-v2")
    if not os.path.exists(model_path):
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        model.save(model_path)

    embedding_model = HuggingFaceEmbeddings(model_name=model_path)

    ######################## Main ##############################

    # Preprocess BMV and edit vektor-database
    if "edit_db" in config or not Path(persistent_dir_cab1).exists() or not Path(persistent_dir_cab2).exists():
        docs_bmv_cab1 = get_BMV(path_bmv_cab1)
        docs_bmv_cab2 = get_BMV(path_bmv_cab2)

        # Create persistent storage/ embedding database from BMV
        edit_vector_db(docs_bmv_cab1, "chroma_db_cab1", persistent_dir_cab1, embedding_model)
        edit_vector_db(docs_bmv_cab2, "chroma_db_cab2", persistent_dir_cab2, embedding_model)

    # Preprocess FTS for STS-analysis (not part of normal operation)
    path_fts = base_dir / "Train data" / ""
    if "load_fts" in config and path_fts.exists():

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
        path = str(base_dir / "Output" / f"Similarity_Results_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.xlsx")
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
            train_svm(x, y, base_dir)

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
                svm_acc, report, train_time, pred_time, mem_train, mem_pred, cross_val, r = evaluate_svm(x, y, eval_dir, base_dir, kernel, c)
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

            with open(eval_dir / "evaluation_log.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(txt))

            winsound.Beep(600, 500)

        if "evaluate_kernel" in config:
            analysis_kernels(x, y, gather_path)

        if "evaluate_sts" in config:
            analysis_sts(gather_path, path_train, embedding_model, [persistent_dir_cab1, persistent_dir_cab2])


    # Manual evaluation
    if "evaluate_manually" in config:
        reader = pd.read_excel(path_train, engine='openpyxl')
        reader = reader.fillna("")
        y_label = []
        y_svm = []
        conf = []

        for i, line in reader.iterrows():
            text = line["Text"]
            element_label = line["Label"]
            cabin = line["Cab"]
            element_predict, confidence = predict_element(cabin, text, k, feat, base_dir, path_train, persistent_dir_cab1,
                                              persistent_dir_cab2,
                                              embedding_model, ui=False)
            y_label.append(element_label)
            y_svm.append(element_predict)
            conf.append(confidence)

        analysis_conf_matrix(y_label, y_svm, encoder=0, path_dir=eval_dir, filename="confusion_matrix_manuelSVM.png")
        svm_report = classification_report(y_label, y_svm)
        print(f"SVM manual report:\n {svm_report}")


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
        prediction, confidence = predict_element(cab, test_step, k, feat, base_dir, path_train, persistent_dir_cab1, persistent_dir_cab2, embedding_model, ui=True)

    plt.close()
    print(prediction)

    return prediction