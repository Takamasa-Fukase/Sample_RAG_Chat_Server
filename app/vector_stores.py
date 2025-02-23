from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import dotenv

# .envを読み込む
dotenv.load_dotenv(dotenv.find_dotenv())

embeddings = OpenAIEmbeddings()
spain_fukase_vector_store = FAISS.load_local("./faiss_index/fukase_spain/", embeddings)