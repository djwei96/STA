import os
import torch
import pandas as pd
import statistics

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_ppl_from_token_ids(df, algo_type, model, device, max_length):
    df_ppl = df.copy(deep=True)
    all_ppl = []
    for index, row in tqdm(df_ppl.iterrows(), total=len(df_ppl)):
        full_output_ids = row[f'full_output_ids_{algo_type}']
        full_output_ids = [full_output_ids[0][-max_length:]]
        full_output_ids = torch.tensor(full_output_ids, dtype=torch.long).to(device)
        with torch.no_grad():
            ppl_outputs = model(input_ids=full_output_ids, labels=full_output_ids)
            loss = ppl_outputs.loss
            ppl = torch.exp(loss)
            ppl = ppl.item()

        df_ppl.at[index, f'ppl_{algo_type}'] = ppl
        all_ppl.append(ppl)
    avg_ppl = statistics.mean(all_ppl)
    return avg_ppl, df_ppl


def get_ppl_from_text(df, algo_type, model, tokenizer, device, max_length):
    df_ppl = df.copy(deep=True)
    all_ppl = []
    for index, row in tqdm(df_ppl.iterrows(), total=len(df_ppl)):
        prompt = row['prompt']
        output_text = row[f'output_{algo_type}']
        full_output_text = prompt + output_text
        full_outputs = tokenizer(full_output_text, return_tensors="pt", truncation=True, max_length=max_length)
        full_outputs = {k: v.to(device) for k, v in full_outputs.items()}
        with torch.no_grad():
            ppl_outputs = model(**full_outputs, labels=full_outputs['input_ids'])
            loss = ppl_outputs.loss
            ppl = torch.exp(loss)
            ppl = ppl.item()

        df_ppl.at[index, f'ppl_{algo_type}'] = ppl
        all_ppl.append(ppl)
    avg_ppl = statistics.mean(all_ppl)
    return avg_ppl, df_ppl


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    ppl_model_name = 'meta-llama/Llama-2-13b-hf'
    cache_dir = 'hf_models'
    ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_name, cache_dir=cache_dir)
    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_name, cache_dir=cache_dir)
    device = 'cuda'
    ppl_model.to(device)

    algo_type = 'sample'
    max_length = 2048
    file_path = 'results/C4/sample/2024-04-30-01:15:50/dataset_output.jsonl'
    df = pd.read_json(file_path, lines=True)
    avg_ppl, df_ppl = get_ppl_from_token_ids(df, algo_type, ppl_model, device, max_length)
    print('ppl from token ids: ', avg_ppl)
    avg_ppl, df_ppl = get_ppl_from_text(df, algo_type, ppl_model, ppl_tokenizer, device, max_length)
    print('ppl from text: ', avg_ppl)
