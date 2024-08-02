from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import  HuggingFaceInstructEmbeddings
import os
from dotenv import load_dotenv, dotenv_values 

load_dotenv() 
 
HUGGING_FACE_API = os.getenv("HUGGING_FACE_API")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
print(HUGGING_FACE_API)


# khởi tạo vector database từ file pdf 
def create_db_from_file(data_path: str, vector_db_path: str): 
    # load file pdf 
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls = PyPDFLoader)
    documents = loader.load()

    # phân chia đoạn văn bản 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    model_name = "hkunlp/instructor-xl"
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name=model_name,
    )
    
    
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)

if __name__ == '__main__': 

    data_path = 'data'
    vector_db_path = 'vectorstores/db_faiss'

    create_db_from_file(data_path, vector_db_path)



