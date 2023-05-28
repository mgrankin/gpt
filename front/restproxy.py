from common import process_seq

seq_length = 1024

from text_generation import Client

client = Client("http://127.0.0.1:7003")

def get_sample(prompt, length:int, num_samples:int, allow_linebreak:bool):
    lm_text = '<LM>' + prompt
    result = client.generate(lm_text, do_sample=True, temperature=.5, repetition_penalty=5.0, typical_p=0.9, top_k=10, top_p=0.95, watermark=False,
                       max_new_tokens=length, ).generated_text.replace('\n', ' ')
    
    result = process_seq([result])
    return result
