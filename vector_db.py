from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import  HuggingFaceInstructEmbeddings
from time import time 
import os
from dotenv import load_dotenv, dotenv_values 


load_dotenv() 
 


class Vector_DB: 

    def __init__(self, input_folder_path):
        self.folder_path = input_folder_path 

    # khởi tạo vector database từ file pdf 
    def create_db_from_file(self): 
        if not os.path.exists(self.folder_path): 
            raise FileNotFoundError(f" '{self.folder_path}' not exist.")
        # load file pdf 
        loader = DirectoryLoader(self.folder_path, glob='*.pdf', loader_cls = PyPDFLoader)
        documents = loader.load()

        # phân chia đoạn văn bản 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        
        model_name = "hkunlp/instructor-xl"
        embedding_model = HuggingFaceInstructEmbeddings(
            model_name=model_name,
        )
        
        
        self.db = Chroma.from_documents(documents=chunks, embedding=embedding_model)
        return self.db

if __name__ == '__main__': 


    def test_init(): 
        start = time()
        vector_db = Vector_DB(input_folder_path = './data').create_db_from_file()
        end = time()

        print(f'Running time: {end - start}')

        


