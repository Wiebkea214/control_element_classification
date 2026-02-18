import os
import chromadb

from langchain_chroma import Chroma

######################################################

def calc_similarity(test_step, persistent_dir, embedding_model, k):

    # Load chroma db
    db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persistent_dir)

    # Combine search string
    input_txt = f"{test_step}."

    # Calculate top k similar results
    top_k = db.similarity_search_with_score(query=input_txt, k=k)

    return top_k


# Function to create and persist vector store
def edit_vector_db(docs, store_name, persistent_dir, embedding_model):

    if not os.path.exists(persistent_dir):
        print(f"\n--- Creating vector db {store_name} ---")

        Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persistent_dir)

        print(f"--- Finished creating vector db {store_name} ---")

    else:
        print(f"--- Vector db {store_name} already exists. Update in proceeding ---") # need to delete and re-add entry

        # Access via chroma client (for deleting)
        chroma_client = chromadb.PersistentClient(path=persistent_dir)
        collection = chroma_client.get_collection(name="langchain")
        # Access via LangChain (for adding)
        db = Chroma(
            embedding_function=embedding_model,
            persist_directory=persistent_dir)

        for doc in docs:
            collection.delete(where={"id": doc.metadata["id"] })
            db.add_documents([doc])

        print(f"--- Finished updating vector db {store_name} ---")


def get_all_classes(persistent_dir):

    chroma_client = chromadb.PersistentClient(path=persistent_dir)
    collection = chroma_client.get_collection(name="langchain")
    metadatas = collection.get(include=["metadatas"])
    meta_key = "id"

    classes = []
    for md in metadatas["metadatas"]:
        if meta_key in md:
            classes.append(md[meta_key])

    return classes