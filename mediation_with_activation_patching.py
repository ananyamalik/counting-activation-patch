import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open('diverse_count.json', 'r') as f:
    source_data = json.load(f)

with open('confounding_data.json', 'r') as f:
    target_data = json.load(f)

def get_all_hidden_states(model, inputs):
    inputs_on_device = {k: v.to(model.device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs_on_device, output_hidden_states=True)
    return outputs.hidden_states

def patching_run(model, corrupted_inputs, clean_hidden_state_to_patch, layer_to_patch, patch_position):
    def patch_hook(module, args, kwargs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        hidden_states[:, patch_position, :] = clean_hidden_state_to_patch
        return output

    handle = model.model.layers[layer_to_patch].register_forward_hook(patch_hook, with_kwargs=True)

    with torch.no_grad():
        patched_outputs = model(**corrupted_inputs)

    handle.remove()

    return patched_outputs

def run_causal_mediation(source, target):
    tokenizer.padding_side = "left"
    source_prompt = f"""Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
    Type: {source['type']}
    List: {target['list']}
    Answer: ("""
    target_prompt = f"""Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
    Type: {target['type']}
    List: {target['list']}
    Answer: ("""

    source_answer = str(source['answer'])
    clean_answer = str(target['answer'])

    inputs = tokenizer(
        [source_prompt, target_prompt],
        return_tensors="pt",
        padding=True
    )
    
    source_inputs = {'input_ids': inputs['input_ids'][0:1], 'attention_mask': inputs['attention_mask'][0:1]}
    target_inputs = {'input_ids': inputs['input_ids'][1:2], 'attention_mask': inputs['attention_mask'][1:2]}

    patch_token_position = inputs['input_ids'].shape[1] - 1
    source_answer_token_id = tokenizer.convert_tokens_to_ids(source_answer)

    source_hidden_states = get_all_hidden_states(model, source_inputs)
    source_states_to_patch = [
        state[0, patch_token_position, :] for state in source_hidden_states
    ]

    target_inputs_on_device = {k: v.to(model.device) for k,v in target_inputs.items()}

    with torch.no_grad():
        target_outputs = model(**target_inputs_on_device)

    target_logits = target_outputs.logits[0, patch_token_position, :]
    target_prob = F.softmax(target_logits, dim=-1)[source_answer_token_id].item()

    patched_probabilities = []
    num_layers = len(model.model.layers)
    for layer_idx in range(num_layers):
        state_to_patch = source_states_to_patch[layer_idx]

        patched_outputs = patching_run(
            model,
            target_inputs_on_device,
            state_to_patch,
            layer_idx,
            patch_token_position
        )

        patched_logits = patched_outputs.logits[0, patch_token_position, :]
        patched_prob = F.softmax(patched_logits, dim=-1)[source_answer_token_id].item()
        patched_probabilities.append(patched_prob)

    return patched_probabilities, target_prob, num_layers

output = []

for i in tqdm(range(1000)):
    op = {}
    op['idx'] = i 
    op['patched_probabilities'], op['target_prob'], op['num_layers'] = run_causal_mediation(source_data[i], target_data[i])
    output.append(op)

torch.save(output, f"output/causal_output_1000.pt")