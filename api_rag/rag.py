from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os 
from time import time 
from dotenv import load_dotenv
import json

load_dotenv()

GEMINI_API = os.getenv("GEMINI_API_KEY")


class Rag_API: 
    
    def __init__(self, API_KEY, data_path, question_num = 10):
        self.folder_path = data_path
        self.question_num  = question_num
        self.api_key = API_KEY

    # load model via api 
    def load_model(self): 
        model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro-latest', 
                                       temperature=0.2, 
                                       api_key= self.api_key)
        return model 
    
    def vector_embbeding_data(self): 
        if not os.path.exists(self.folder_path): 
            raise FileNotFoundError(f" '{self.folder_path}' not exist.")
        
        loader = DirectoryLoader(self.folder_path, glob='*.pdf', loader_cls = PyPDFLoader)
        documents = loader.load()

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

       
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",  google_api_key = self.api_key)
        db = Chroma.from_documents(documents=chunks, embedding=embeddings).as_retriever(search_kwargs = {'k' : 3})
        
        return db
    
    def load_prompt(self):
        template  = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
                        {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        question =  """ Bạn hãy tạo ra %d câu hỏi trắc nghiệm có 4 đáp án trong đó có 1 đáp án đúng từ thông tin trên. Kết quả trả về một đối tượng JSON chứa toàn bộ nội dung câu hỏi và câu trả lời.
                        Bạn cũng cần sửa lại câu hỏi và câu trả lời nếu có lỗi chỉnh tả.
                        Mỗi mục nên được gán nhãn như sau:
                        - "question 1": {
                            "Câu hỏi": "<Câu hỏi trắc nghiệm yêu cầu hình ảnh kèm theo>"
                            "Đáp án 1": "<Đáp án đầu tiên của câu hỏi>"
                            "Đáp án 2": "<Đáp án thứ hai của câu hỏi>"
                            "Đáp án 3": "<Đáp án thứ ba của câu hỏi>"
                            "Đáp án 4": "<Đáp án thứ tư của câu hỏi>"
                            "Đáp án đúng": "<Đáp án đúng của câu hỏi>"
                        }
                        - "question 2": {
                            "Câu hỏi": "<Câu hỏi trắc nghiệm tiếp theo yêu cầu hình ảnh kèm theo>"
                            "Đáp án 1": "<Đáp án đầu tiên của câu hỏi>"
                            "Đáp án 2": "<Đáp án thứ hai của câu hỏi>"
                            "Đáp án 3": "<Đáp án thứ ba của câu hỏi>"
                            "Đáp án 4": "<Đáp án thứ tư của câu hỏi>"
                            "Đáp án đúng": "<Đáp án đúng của câu hỏi>"
                        }
                        ...
                        - "question %d": {
                            "Câu hỏi": "<Câu hỏi trắc nghiệm thứ n yêu cầu hình ảnh kèm theo>"
                            "Đáp án 1": "<Đáp án đầu tiên của câu hỏi>"
                            "Đáp án 2": "<Đáp án thứ hai của câu hỏi>"
                            "Đáp án 3": "<Đáp án thứ ba của câu hỏi>"
                            "Đáp án 4": "<Đáp án thứ tư của câu hỏi>"
                            "Đáp án đúng": "<Đáp án đúng của câu hỏi>"
                        }
                    """ % (self.question_num, self.question_num)

        return prompt, question  

    """ 
    Hàm output return file json tại output 
    
    """
    def generate_question(self, output_path): 
        prompt, question = self.load_prompt()
        model = self.load_model()
        docs = self.vector_embbeding_data().get_relevant_documents(question)

        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        result = response['output_text']
        json_str = (result).strip('```json\n').strip('```')
    
        response_json = json.loads(json_str)

        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(response_json, json_file, ensure_ascii=False, indent=4)


        return response['output_text']
    
    



if __name__ == '__main__': 

    def test(): 
        start = time()
        
        rag = Rag_API(GEMINI_API, './server/rag/input')

        print(rag.generate_question('./server/rag/output/output.json'))


        finish = time()

        print(f"Run time: {finish - start}")

    test()






