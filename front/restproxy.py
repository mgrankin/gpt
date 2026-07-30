from common import process_seq

seq_length = 1024

from text_generation import Client

client = Client("http://127.0.0.1:7003")

def get_sample(prompt, length:int, num_samples:int, allow_linebreak:bool, temperature:float):
    lm_text = '<LM>' + prompt
    result = client.generate(lm_text, do_sample=True, temperature=temperature, repetition_penalty=1.2, top_k=0, top_p=1.0, watermark=False,
                       max_new_tokens=length, ).generated_text.replace('\n', ' ')
    
    result = process_seq([result])
    return result
