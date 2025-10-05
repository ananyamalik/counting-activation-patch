import random
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class Model:

    def __init__(self, model_name, filename):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32,
                device_map = "auto"
                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.filename = filename

    def load_data(self):
        with open(self.filename, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    
    def make_prompt(self, category, items):
        return f"""Count the number of words in the following list that match the given category.
Type: {category}\nList: {items}\nAnswer (only a number in parentheses):"""

    def query_model(self, category, items, max_new_tokens=5):
        prompt = self.make_prompt(category, items)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        match = re.search(r"\((\d+)\)", text)
        if match:
            return int(match.group(1))
        else:
            return None

    def run_benchmark(self, num=10000):
        df = self.load_data()
        df_abridged = df[:num]
        output = []
        for i, row in tqdm(df_abridged.iterrows()):
            answer = self.query_model(row['type'], row['list'])
            output.append(answer)
        return output
    
    def calc_res(self, output, num=10000):
        df = self.load_data()
        df_abridged = df[:num]
        df_abridged['output'] = output
        df_abridged['result'] = df_abridged.apply(lambda row: row['answer'] == row['output'], axis=1)
        filtered_df = df_abridged[df_abridged['result'] == True]
        return len(filtered_df)/num

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    filename = "diverse_count.json"
    num = 10
    exp = Model(model_name, filename)
    output = exp.run_benchmark(num)
    accuracy = exp.calc_res(output, num)
    print(output)
    print(accuracy)
