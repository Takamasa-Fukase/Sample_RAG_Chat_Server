from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import dotenv

# .envを読み込む
dotenv.load_dotenv(dotenv.find_dotenv())

embeddings = OpenAIEmbeddings()
# spain_fukase_vector_store = FAISS.load_local("./faiss_index/fukase_spain/", embeddings)
vector_store_2019 = FAISS.load_local("./faiss_index/2019/", embeddings)
vector_store_2022 = FAISS.load_local("./faiss_index/2022/", embeddings)
vector_store_2025 = FAISS.load_local("./faiss_index/2025_2/", embeddings)