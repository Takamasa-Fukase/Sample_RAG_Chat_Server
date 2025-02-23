import os
import dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from recursive_text_splitter import recursive_text_splitter

# .envを読み込む
dotenv.load_dotenv(dotenv.find_dotenv())

loader = DirectoryLoader('./txt/fukase_spain')
documents = loader.load()
# docs = recursive_text_splitter.split_documents(documents)

# for doc in docs:
#     print(f'docの中身: {doc}, len: {len(doc.page_content)}\n\n')

# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(docs, embeddings)

# index_path = './faiss_index/fukase_spain'
# db.save_local(index_path)
