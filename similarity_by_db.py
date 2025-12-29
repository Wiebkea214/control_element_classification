import chromadb

from langchain_chroma import Chroma

################################

def calc_similarity(test_step, persistent_dir, embedding_model):

    # Load chroma db
    db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persistent_dir)

    # Combine search string
    input_txt = f"{test_step}."

    # Calculate top 3 similar results
    top5 = db.similarity_search_with_score(query=input_txt, k=5)

    return top5