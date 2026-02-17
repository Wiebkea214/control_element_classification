import re
import psutil
import time
import numpy as np

from setup_vector_database import *
from preprocessing_data import *
from sklearn.metrics.pairwise import cosine_similarity

#########################################################

def build_feature_vector(embedding, persistent_dir, text, cab, k, feat):

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.strip().lower())
    text = f"{text}. In {cab}"
    text_emb = embedding.embed_query(str(text))

    process = psutil.Process()
    sts_time = []
    sts_mem = []

    # Calculate STS score
    persistent_dir_cabx = select_cab(cab, persistent_dir)

    mem_sts_before = process.memory_info().rss / (1024 * 1024)
    sts_start = time.perf_counter()

    if persistent_dir_cabx:
        candidates_top_k = calc_similarity(text, persistent_dir_cabx, embedding, k)
    else:
        print(f"!!! Matching vector-db not found with cab={cab} !!!")
        return -1

    sts_end = time.perf_counter()
    mem_sts_after = process.memory_info().rss / (1024 * 1024)
    sts_time.append(sts_end - sts_start)
    sts_mem.append(mem_sts_before - mem_sts_after)

    features = []
    sts_all = []
    weighted_emb_all = []
    #cosine_all = []

    if feat != 0:
        for sts_candidate, sts_score in candidates_top_k:
            sts_emb = np.array(embedding.embed_query(str(sts_candidate.page_content)))
            #cosine_score = cosine_similarity([text_emb], [sts_emb])[0][0]
            #cosine_scaled = (cosine_score + 1) / 2

            '''
            if feat == 2 or feat == 6:
                features.append(sts_score)
                features.append(cosine_score)
            '''

            features.append(sts_score)
            sts_all.append(sts_score)
            weighted_emb_all.extend([sts_score * entry for entry in sts_emb])
            #cosine_all.append(cosine_scaled)

        #mean_cos = np.mean(cosine_all, axis=0)
        #var_cos = np.var(cosine_all, axis=0)
        mean_sts = np.mean(sts_all, axis=0)
        var_sts = np.var(sts_all, axis=0)
        min_sts = np.min(sts_all, axis=0)
        max_sts = np.max(sts_all, axis=0)
        range_sts = (max_sts - min_sts)
        rel_sts = (sts_all[0] + 0.001) / (sts_all[1] + 0.001)  # addition of 0.001 to prevent zero division
        weight_emb = np.sum(weighted_emb_all)

        '''
        if feat == 6:
            features.extend([mean_sts, var_sts, mean_cos, var_cos])
        '''

        if feat == 9:
            features.extend([weight_emb, mean_sts, var_sts, min_sts, max_sts, range_sts, rel_sts])
            features.extend(text_emb)

        '''
        if feat == 11:
            features.extend([weight_emb, mean_sts, var_sts, min_sts, max_sts, range_sts, rel_sts])
            features.extend(text_emb)
        elif feat == 12:
            features.extend([weight_emb, var_sts, min_sts, max_sts, range_sts, rel_sts])
            features.extend(text_emb)
        elif feat == 13:
            features.extend([weight_emb, mean_sts, min_sts, max_sts, range_sts, rel_sts])
            features.extend(text_emb)
        elif feat == 14:
            features.extend([weight_emb, mean_sts, var_sts, max_sts, range_sts, rel_sts])
            features.extend(text_emb)
        elif feat == 15:
            features.extend([weight_emb, mean_sts, var_sts, min_sts, range_sts, rel_sts])
            features.extend(text_emb)
        elif feat == 16:
            features.extend([weight_emb, mean_sts, var_sts, min_sts, max_sts, rel_sts])
            features.extend(text_emb)
        elif feat == 17:
            features.extend([weight_emb, mean_sts, var_sts, min_sts, max_sts, range_sts])
            features.extend(text_emb)
        elif feat == 18:
            features.extend([weight_emb, mean_sts, var_sts, min_sts, max_sts, range_sts, rel_sts])
        elif feat == 19:
            features.extend([mean_sts, var_sts, min_sts, max_sts, range_sts, rel_sts])
            features.extend(text_emb)
        '''

    else:
        features.extend(text_emb)

    top1 = candidates_top_k[0][0]
    dim = len(features)

    return features, top1, dim, sts_time, sts_mem


