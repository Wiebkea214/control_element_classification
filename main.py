from preprocessing_data import *
from setup_vector_database import *
from similarity_by_db import *
from write_data_in_excel import *
from training_svm import *

from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

#################### Interface (lvl1) #######################

path_fts = "C:\\Users\\Wiebke\\OneDrive\\Masterarbeit\\FTS Daten\\TRAXX_AC3_ISR__Control_front_and_tail_lights.xlsx"
path_bmz = "C:\\Users\\Wiebke\\OneDrive\\Masterarbeit\\FTS Daten\\BMZ_3EGM216280L0001_Stand_E_manipulated_DescriptionAsLabels.xlsx"
path_train = "C:\\Users\\Wiebke\\OneDrive\\Masterarbeit\\FTS Daten\\Training_1.xlsx"
cab = 1

####################### Globals #############################

current_dir = os.path.dirname(os.path.abspath(__file__))
store_name = "chroma_db"
persistent_dir = str(os.path.join(current_dir, store_name))
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Available configs: load_fts, edit_db, similarity, train_svm
config = "load_fts, similarity"

######################## Main ###############################

# Preprocess BMZ
if "edit_db" in config:
    dict_bmz, docs_bmz = get_BMV(path_bmz)

    # Create persistent storage/ embedding database from BMZ
    edit_vector_db(docs_bmz, store_name, persistent_dir, embedding_model)

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
            result = calc_similarity(test_step, cab, persistent_dir, embedding_model)

            # Write result in excel file
            data = [id, test_step, result[0][0], result[1][0], result[2][0], result[0][1], result[1][1], result[2][1]]
            write_excel(headers=headers, data=data, path=path)

    if "train_svm" in config:
        get_traindata(path_train, cab, persistent_dir, embedding_model)

print("--- Finished ---")
