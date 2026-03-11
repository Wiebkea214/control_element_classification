import re
import psutil
import time
import numpy as np

from project.functions__setup_vector_database import *
from project.functions__preprocessing_data import *

#########################################################

def build_feature_vector(embedding, persistent_dir, text, cab, k, feat):

    # Normalize text input
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.strip().lower())
    text = f"{text}. In {cab}"

    # Create text-embedding
    text_emb = embedding.embed_query(str(text))

    # Initialize supervision
    process = psutil.Process()
    sts_time = []
    sts_mem = []

    # Start STS supervision
    mem_sts_before = process.memory_info().rss / (1024 * 1024)
    sts_start = time.perf_counter()

    # Perform STS
    persistent_dir_cabx = select_cab(cab, persistent_dir)
    candidates_top_k = calc_similarity(text, persistent_dir_cabx, embedding, k)

    # End STS supervision
    sts_end = time.perf_counter()
    mem_sts_after = process.memory_info().rss / (1024 * 1024)
    sts_time.append(sts_end - sts_start)
    sts_mem.append(mem_sts_before - mem_sts_after)

    # Create feature vector
    sts_all = []
    weighted_emb_all = []

    if feat == 9:
        for sts_candidate, sts_score in candidates_top_k:
            sts_emb = np.array(embedding.embed_query(str(sts_candidate.page_content)))
            sts_all.append(sts_score)
            weighted_emb_all.extend(sts_score * sts_emb)

        mean_sts = np.mean(sts_all, axis=0)
        var_sts = np.var(sts_all, axis=0)
        min_sts = np.min(sts_all, axis=0)
        max_sts = np.max(sts_all, axis=0)
        range_sts = (max_sts - min_sts)
        rel_sts = (sts_all[0] + 0.001) / (sts_all[1] + 0.001)  # addition of 0.001 to prevent zero division
        weight_emb = np.sum(weighted_emb_all)

        features = [
            weight_emb,
            mean_sts,
            var_sts,
            min_sts,
            max_sts,
            range_sts,
            rel_sts,
            *weighted_emb_all,
            *text_emb]

    else:
        features = text_emb

    top1 = candidates_top_k[0][0]
    dim = len(features)

    return features, top1, dim, sts_time, sts_mem


