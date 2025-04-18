from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Split documents into smaller chunks
def split_document(document, chunk_size=1000, chunk_overlap=200):
    """Splits a single document into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_docs = splitter.split_documents(document)
    return splitted_docs


# Function to process documents in small batches
def batch_process(documents, batch_size=50):
    """Yields batches of document chunks for embedding."""
    # len(documents) = 6779
    for i in range(0, len(documents), batch_size):
        output = documents[i:i + batch_size]
        yield output


# Perform batch processing within the single document
def get_vectorstore(docs, embeddings_model, doc_batch_size):
    vectorstore = None
    for batch in batch_process(docs, batch_size=doc_batch_size):
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings_model)  # Initialize FAISS with first batch
        else:
            vectorstore.add_documents(batch)  # Add subsequent batches
    return vectorstore

# from concurrent.futures import ThreadPoolExecutor

# # Function to process documents in parallel batches
# def batch_process(documents, batch_size=50, max_workers=16):
#     """Yields batches of document chunks for embedding using parallel processing."""
#     def process_batch(start_idx):
#         return documents[start_idx:start_idx + batch_size]

#     print("Batch process start")
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_batch, i) for i in range(0, len(documents), batch_size)]
#         for future in futures:
#             yield future.result()
#     print("Batch process end")





# from concurrent.futures import ThreadPoolExecutor

# # Perform batch processing within the single document using parallel processing
# def get_vectorstore(docs, embeddings_model, args):
#     print("Vectorstore process start")
#     vectorstore = None

#     def process_batch(batch):
#         """Creates FAISS embeddings for a given batch."""
#         return FAISS.from_documents(batch, embeddings_model)

#     # Use ThreadPoolExecutor to parallelize embedding creation
#     with ThreadPoolExecutor(max_workers=16) as executor:
#         futures = [executor.submit(process_batch, batch) for batch in batch_process(docs, batch_size=args.doc_batch_size)]

#         for future in futures:
#             if vectorstore is None:
#                 vectorstore = future.result()  # Initialize FAISS with the first processed batch
#             else:
#                 vectorstore.merge_from(future.result())  # Merge FAISS indexes

#     print("Vectorstore process end")
#     return vectorstore


