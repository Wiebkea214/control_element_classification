from project.functions__human_in_the_loop import *
from project.functions__training_svm import *


##########################################################################

def predict_element(cab, test_step, k, feat, base_dir, path_train, persistent_dir_cab1, persistent_dir_cab2, embedding_model, ui):

    # Check if element nr. is used in test step
    persistent_dir_cabx = select_cab(cab, [persistent_dir_cab1, persistent_dir_cab2])
    prediction = element_precheck(test_step, persistent_dir_cabx)
    confidence = 1
    conf_threshold = 0.6

    # If not, perform classification
    if prediction:
        print(f"\nElement found in test step: {prediction}")
    else:
        svm_model = joblib.load(base_dir / "svm_model.joblib")
        scaler = joblib.load(base_dir / "scaler.joblib")

        features, top1, dim, sts_time, sts_mem = build_feature_vector(embedding_model, [persistent_dir_cab1, persistent_dir_cab2], test_step, cab, k, feat)

        # Normalize vector
        features_scaled = scaler.transform([features])

        prediction = svm_model.predict(features_scaled)[0]

        # Confidence
        scores = svm_model.decision_function(features_scaled)[0]
        softmax = np.exp(scores) / np.sum(np.exp(scores))
        confidence = softmax.max()

        # Human-in-the-Loop
        if ui:
            if (top1.metadata["id"] == prediction) and confidence >= conf_threshold:
                print(f"\nSVM prediction is valid: {prediction}")

            else:
                print(f"\nSVM prediction is unsure: prediction={prediction} vs. STS={top1.metadata['id']} and confidence = {confidence}")
                feedback = hil_ui(prediction, confidence, persistent_dir_cabx, path_train, test_step)

                if feedback["status"] == "incorrect_excel" and feedback["correct_label"]:
                    # Save correct label in training excel
                    save_fedback_to_excel(train_path=path_train, text=test_step, cab=cab, correct_label=feedback["correct_label"])
                    prediction = feedback["correct_label"]
                    print(f"\nCorrected SVM prediction: {prediction} (Added to training db)")

                if feedback["status"] == "incorrect" and feedback["correct_label"]:
                    # Do not save label in training excel
                    prediction = feedback["correct_label"]
                    print(f"\nCorrected SVM prediction: {prediction} (not added to training db because of duplicate)")

                elif feedback["status"] == "retrain":
                    x, y, y_sts, sts_time, sts_mem, dim = get_traindata(path_train,[persistent_dir_cab1, persistent_dir_cab2], embedding_model, k, feat)
                    train_svm(x, y, base_dir)

                    # Clear "new" column
                    df = pd.read_excel(path_train, engine="openpyxl")
                    df["New"] = ""
                    df.to_excel(path_train, index=False)

                else:
                    # no action, canceled/correct prediction
                    pass

    return prediction, confidence