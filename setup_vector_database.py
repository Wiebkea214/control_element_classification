import os
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Global content


######################################################

# Function to create and persist vector store
def edit_vector_db(docs, store_name, persistent_dir, embedding):

    if not os.path.exists(persistent_dir):
        print(f"\n--- Creating vector db {store_name} ---")

        Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persistent_dir)

        print(f"--- Finished creating vector db {store_name} ---")

    else:
        print(f"--- Vector db {store_name} already exists. Update in proceeding ---") # need to delete and re-add entry

        # Access via chroma client (for deleting)
        chroma_client = chromadb.PersistentClient(path=persistent_dir)
        collection = chroma_client.get_collection(name="langchain")
        # Access via LangChain (for adding)
        db = Chroma(
            embedding_function=embedding,
            persist_directory=persistent_dir)

        for doc in docs:
            collection.delete(where={"id": doc.metadata["id"] })
            db.add_documents([doc])

        print(f"--- Finished updating vector db {store_name} ---")



'''
# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)

        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


# Define the user's question
query = "text"

# Query each vector store
query_vector_store("chroma_db_huggingface", query, embedding_model)

'''

