from local_rag.pretrain_model import Local_pretrain_model
from local_rag.vector_db import Vector_DB 
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.question_answering.chain import load_qa_chain
from langchain.chains.llm import LLMChain
from time import time 


class Question_Generate: 
    """
    Class chính dùng để chạy toàn thể với đầu vào : 
    1. model_name là id của model trên huggingface
    2. input_folder_path là file_path của dữ liệu đầu vào 
    3. input là số lượng câu hỏi người dùng yêu cầu (mặc định là 5)
    
    """
    def __init__(self, model_name, input_folder_path, input=5): 
        self.pretrain_model = Local_pretrain_model(model_name)
        self.vector_db = Vector_DB(input_folder_path)
        self.num_question = input 

    # load_pretrain model 
    def load_model(self): 
        model_pipeline = self.pretrain_model.load_model()
        return model_pipeline
    
    # load vector_db 
    def load_db(self): 
        db = self.vector_db.create_db_from_file()
        return db
    
    # load mẫu prompt 
    def load_prompt(self): 
        template  = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
                        {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        question =  """Hãy tạo ra {self.num_question} câu hỏi trắc nghiệm lựa chọn 4 đáp án và trong đó có 1 đáp án đúng từ tập tài liệu trên. Kết quả trả về phải là một file định dạng JSON có dạng:\n 
                    - "question1": {
                        "Question": "<Câu hỏi trắc nghiệm có 4 đáp án>",
                        "Answer1": "<Lựa chọn thứ 1 của câu hỏi trắc nghiệm đó>",
                        "Answer2": "<Lựa chọn thứ 2 của câu hỏi trắc nghiệm đó>",
                        "Answer3": "<Lựa chọn thứ 3 của câu hỏi trắc nghiệm đó>",
                        "Answer4": "<Lựa chọn thứ 4 của câu hỏi trắc nghiệm đó>"
                    }
                    ..... 
                    - "question{self.num_question}": {
                        "Question": "<Câu hỏi trắc nghiệm có 4 đáp án>",
                        "Answer1": "<Lựa chọn thứ 1 của câu hỏi trắc nghiệm đó>",
                        "Answer2": "<Lựa chọn thứ 2 của câu hỏi trắc nghiệm đó>",
                        "Answer3": "<Lựa chọn thứ 3 của câu hỏi trắc nghiệm đó>",
                        "Answer4": "<Lựa chọn thứ 4 của câu hỏi trắc nghiệm đó>"
                    }
            
                    """

        return prompt, question  
    

    # hàm chạy toàn bộ pipeline 
    def generate_result(self): 
        model_pipeline = self.load_model()
        db = self.load_db()
        prompt, question = self.load_prompt()
        

        db = db.similarity_search(question)
        print(db)
        context = ''

        for i in range(max(len(db), 3)): 
            context += db[i].page_content

        prompt = prompt.format(context=  context, question = question)

        output = model_pipeline(prompt = prompt, max_tokens= 3000, temperature=0.1)

        return output["choices"][0]["text"].strip()


if __name__ == '__main__': 
    
    model_id = 'server/rag/model/vinallama-7b-chat_q5_0.gguf'
    input_path = './server/rag/input/'

    # test load model 
    def test_response_model(): 

        start = time()
        qa_generate = Question_Generate(model_id,  input_path)
        response = qa_generate.generate_result()
        print(response)

        finish = time()

        print(f"Run time: {finish - start}")

    test_response_model()