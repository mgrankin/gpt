import regex as re

from pydantic import BaseModel, Field
class Prompt(BaseModel):
    prompt:str = Field(..., title='Model prompt')
    model:str = Field('gpt3', title='Model type')
    length:int = Field(15, ge=1, le=150, title='Number of tokens generated in each sample')
    num_samples:int = Field(3, ge=1, le=5, title='Number of samples generated')
    allow_linebreak:bool = Field(False, title='Allow linebreak in a sample')

def fix_string(string) -> str:
    in_word = string
    in_between_words = ['-', '–']
    in_sentences = ['«', '(', '[', '{', '"', '„', '\'']

    for item in in_between_words:
        regex = r'\w[%s]\s\w' % item
        in_word = re.findall(regex, string)

        for x in in_word:
            a = x[:1]; b = x[3:4]
            string = string.replace(x, a + '-' + b)

    for item in in_sentences:
        string = string.replace(f' {item} ', f' {item}')

    return string

def process_seq(generated_sequences):
    reg_text = [re.match(r'[\w\W]*[\.!?]\n', item) for item in generated_sequences]
    reg_text2 = [re.match(r'[\w\W]*[\.!?]', item) for item in generated_sequences]
    result = [reg_item[0] if reg_item else reg_item2[0] if reg_item2 else item for reg_item, reg_item2, item in zip(reg_text, reg_text2, generated_sequences)]
    result = [fix_string(s) for s in result]
    return result 