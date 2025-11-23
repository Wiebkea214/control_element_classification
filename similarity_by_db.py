import chromadb

from langchain_chroma import Chroma

################################

def calc_similarity(test_step, cab, persistent_dir, embedding):

    # Load chroma db
    db = Chroma(
        embedding_function=embedding,
        persist_directory=persistent_dir)

    # Combine search string
    input_txt = f"{test_step}. In cab {cab}."

    # Calculate top 3 similar results
    top5 = db.similarity_search_with_relevance_scores(query=input_txt, k=3)

    return top5