import random
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import re
import numpy as np
import pickle
from tqdm import tqdm

class Model:

    def __init__(self, model_name, filename):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32,
                device_map = "auto", 
                output_hidden_states=True
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

    # def patched_model(self, token, layer, per_word_hidden_states, word_token_positions):


    def query_model(self, category, items):
        prompt = self.make_prompt(category, items)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # hidden_states = outputs.hidden_states[layer][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        word_token_positions = []
        token_idx = 0
        # for word in items:
        #     token_ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
        #     token_range = list(range(token_idx, token_idx+len(token_ids)))
        #     word_token_positions.append(token_range)
        #     token_idx += len(token_ids)
        last_token_idx = input_ids.shape[1] - 1

        per_layer_results = []
        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
            layer_hidden = layer_hidden[0]  # shape: [seq_len, hidden_dim]
            last_token_hidden = layer_hidden[last_token_idx].cpu()  # [hidden_dim]
            per_layer_results.append({
                "layer": layer_idx,
                "last_token_hidden": last_token_hidden
            })
        
        # per_layer_results = []
        # for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
        #     layer_hidden = layer_hidden[0] 
        #     per_word_hidden_states = []
        #     for pos in word_token_positions:
        #         per_word_hidden_states.append(layer_hidden[pos].mean(dim=0))

        #     per_layer_results.append({
        #         "layer": layer_idx,
        #         "hidden_states": layer_hidden.cpu(),
        #         "per_word_hidden_states": [x.cpu() for x in per_word_hidden_states]
        #     })

        return {
            "tokens": tokens,
            "last_token": last_token_idx,
            "layers": per_layer_results
        }

        
        # return hidden_states, word_token_positions
    
        # generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        # text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # match = re.search(r"\((\d+)\)", text)
        # if match:
        #     return int(match.group(1))
        # else:
        #     return None
    
    def run_model(self, num=10000):
        X = []
        y = []
        df = self.load_data()
        df_abridged = df[:num]
        embeddings_complete = []
        for i, row in tqdm(df_abridged.iterrows()):
            sent = {
                "id": i,
                "list": row["list"],
                "type": row["type"],
                "results": self.query_model(row["type"], row["list"])
            }
            embeddings_complete.append(sent)
        
        return embeddings_complete

    # def run_model(self, layer, num=10000):
    #     X = []
    #     y = []
    #     df = self.load_data()
    #     df_abridged = df[:num]
    #     output = []
    #     for i, row in tqdm(df_abridged.iterrows()):
    #         hidden_states, word_token_positions = self.query_model(row['type'], row['list'], layer)
    #         for pos, count in zip(word_token_positions, row['running_count']):
    #             X.append(hidden_states[pos].cpu().numpy())
    #             y.append(count)

    #     return np.stack(X), np.stack(y)
    
    def linear_probing(self, X, y):
        probe = LinearRegression()
        probe.fit(X,y)
        y_pred = probe.predict(X)
        return y_pred
    
    def calc_res(self, output, num=10000):
        df = self.load_data()
        df_abridged = df[:num]
        df_abridged['output'] = output
        df_abridged['result'] = df_abridged.apply(lambda row: row['answer'] == row['output'], axis=1)
        filtered_df = df_abridged[df_abridged['result'] == True]
        return len(filtered_df)/num
    
    def patch_last_token_and_generate(
            self,
            base_category,
            base_items,
            donor_hidden_states,
            layer_to_patch,
        ):
        base_prompt = self.make_prompt(base_category, base_items)
        inputs = self.tokenizer(base_prompt, return_tensors="pt").to(self.device)
        last_token_idx = inputs["input_ids"].shape[1] - 1
        answer_start_token_idx = (input_ids == tokenizer(":").input_ids[-1]).nonzero()[0]


        def patch_hook(module, input, output):
            output = output.clone()
            layer_dict = donor_hidden_states[layer_to_patch]
            donor_hidden = layer_dict["last_token_hidden"]
            
            last_idx = output.shape[1] - 1
            output[0, answer_start_token_idx, :] = donor_hidden.to(output.device).to(output.dtype)
            return output


        handle = self.model.model.layers[layer_to_patch].register_forward_hook(
            lambda module, inp, out: patch_hook(module, inp, out)
        )

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0
            )

        handle.remove()
        text_out = self.tokenizer.decode(generated[0])

        return text_out



    
    

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    filename = "diverse_count_with_running.json"
    num = 1000
    category = "fruit"
    items = ['apple', 'chair', 'banana']
    exp = Model(model_name, filename)
    output = exp.run_model(num)
    torch.save(output, f"output/embedding_last_token_{num}.pt")
    # with open(f"output/embedding_complete_{num}.pkl", "wb") as f:
    #     pickle.dump(output, f)
    # for layer in range(0,30):
    #     X, y = exp.run_model(layer=layer, num=num)
    #     y_pred = exp.linear_probing(X,y)
    #     print(f"Layer: {layer}\tr2_score: {r2_score(y, y_pred)}")
    # output = exp.run_benchmark(num)
    # accuracy = exp.calc_res(output, num)
    # print(output)
    # print(accuracy)
