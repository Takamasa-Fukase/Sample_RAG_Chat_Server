import os
import app_const
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from typing import Any, Dict, List, Union
from langchain.docstore.document import Document

os.environ["OPENAI_API_KEY"] = app_const.OPEN_AI_API_KEY

loader = DirectoryLoader('./txt/genre4')
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=300, # 暫定で300で設定
    chunk_overlap=20, # 暫定で20で設定
    separator="\n" # なぜかデフォの\n\nではなく\nじゃないと正しくchunk_sizeが認識されない気がする
    # TODO: - （chunk_sizeをあまりにも超過したデータができちゃう原因と対策を調査）もしかすると、指定したchunk_sizeを超えた後にseparatorで設定した文字が来るまでは分割されないのかも？？
)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

index_path = './faiss_index/genre4'
db.save_local(index_path)
