import os 
from dotenv import load_dotenv
import transformers
from torch import cuda, bfloat16
from langchain_community.llms import huggingface_pipeline
from time import time 
load_dotenv()

HUGGINGFACE_API = os.getenv("HUGGING_FACE_API") 

model_id = 'vilm/vinallama-2.7b-chat'

class Pretrain_model: 

    def __init__(self, model_id): 
        self.bnb_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=bfloat16
                    )   
        
        self.model_config =transformers.AutoConfig.from_pretrained(
                        model_id,
                        token = HUGGINGFACE_API
                    )

        self.tokenizer  = transformers.AutoTokenizer.from_pretrained(model_id, token = HUGGINGFACE_API)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    config=self.model_config,
                    quantization_config=self.bnb_config,
                    device_map='auto',
                    token = HUGGINGFACE_API
                )
    



    def build_model_pipeline(self): 

        text_generation_pipeline = transformers.pipeline(
            model= self.model,
            tokenizer= self.tokenizer,
            task="text-generation",
            eos_token_id= self.tokenizer.eos_token_id,
            pad_token_id= self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=100,
        )

        my_pipeline = huggingface_pipeline.HuggingFacePipeline(pipeline=text_generation_pipeline)
        return my_pipeline 
    
    
        


if __name__ == '__main__': 


    def test_prepare(): 
        start = time()
        pretrain = Pretrain_model('vilm/vinallama-2.7b-chat')
        pipeline = pretrain.build_model_pipeline()

        finish = time()
        print(f'Prepare time is {finish - start}')

    test_prepare()


    def test_query(query: str): 
        start = time()

        Pretrain_model = Pretrain_model('vilm/vinallama-2.7b-chat')
        pipeline = Pretrain_model.build_model_pipeline()

        ans = pipeline(query)

        finish = time() 

        print(f'Prepare time is {finish - start}')

    test_query("Hình tam giác có bao nhiêu cạnh?")
    
