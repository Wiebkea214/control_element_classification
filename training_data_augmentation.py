import pandas as pd
import nlpaug.augmenter.word as word_aug
import re

from sentence_transformers import util
from evaluation import *

# Global:

###########################################################

def filter_variants(original, variants, threshold, embedding_model):
    orig_emb = embedding_model.encode(original, convert_to_tensor=True)
    filtered = []
    sim_val = []
    for i, v in enumerate(variants):
        sim_cos = util.cos_sim(orig_emb, embedding_model.encode(v, convert_to_tensor=True)).item()
        sim_val.append(sim_cos)
        if sim_cos >= threshold and sim_cos <= 0.98:
            filtered.append(v[i])
    return filtered, sim_val


def training_data_augmentation(path_train, embedding_model, max_class_cnt):

    # Read prepared training excel
    df = pd.read_excel(path_train)

    print(f"\n--- Start preprocess of augmentation ---")

    # Create augmentation with different methods (synonym, back translation, context)
    #syn_aug = word_aug.SynonymAug(aug_src='wordnet')
    bt_aug = word_aug.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de',
        to_model_name='Helsinki-NLP/opus-mt-de-en',
        device='cpu')
    #ctxt_aug = word_aug.ContextualWordEmbsAug(
    #    model_path='bert-base-uncased',
    #    action='substitute')

    # Perform balancing of class labels
    class_count = df["Label"].value_counts()

    print(f"\n--- Start augmentation ---")

    for label, count in class_count.items():
        needed = max_class_cnt - count
        augmented_rows = []
        similarities = [["back translation"]]

        texts = df[df["Label"] == label]["Text"].tolist()


        for text_t in texts:
            if needed > 0:
                cleaned1_text = text_t.lower().strip()
                cleaned2_text = re.sub(r"[^a-zA-Z0-9]", " ", cleaned1_text)

                variants = [bt_aug.augment(cleaned2_text)]
                filtered, sim_val = filter_variants(cleaned2_text, variants, 0.5, embedding_model)
                similarities.append(sim_val)

                for v in filtered:
                    augmented_rows.append({"Text": v, "Label": label})
                    needed -= 1
            else:
                break

        for text_a in augmented_rows:
            if needed > 0:
                cleaned1_text = text_a.lower().strip()
                cleaned2_text = re.sub(r"[^a-zA-Z0-9]", " ", cleaned1_text)

                variants = [bt_aug.augment(cleaned2_text)]
                filtered, sim_val = filter_variants(cleaned2_text, variants, 0.5, embedding_model)
                similarities.append(sim_val)

                for v in filtered:
                    augmented_rows.append({"Text": v, "Label": label})
                    needed -= 1

        if augmented_rows:

            # Analyze similarity distribution
            data_plot(similarities, label)

            # Write back to excel
            df_aug = pd.concat([df, pd.DataFrame(augmented_rows)])
            df_aug.to_excel(path_train, index=False)

    print(f"\n--- Finished augmentation ---")