"""
MAUVE 
Adapted from the eval script in https://github.com/XiangLi1999/ContrastiveDecoding
requirement: pip install mauve-text (https://github.com/krishnap25/mauve)
"""

from transformers import AutoTokenizer
from tqdm import tqdm
import os
import mauve
import json
import statistics
import pandas as pd


def get_mauve_score_from_C4(df, algo_type, max_len=256, verbose=False, device_id=0, featurize_model_name="gpt2"):
    with open('data/C4/sampled_dataset.json', "r") as input_file:
        data = json.load(input_file)

    with open('data/C4/sampled_dataset_train.json', "r") as input_file:
        train_data = json.load(input_file)

    # get last 200 tokens as reference text from C4
    all_reference_text = []
    all_output_text = []
    for index, row in df.iterrows():
        full_text = data[index].strip()
        train_text = train_data[index].strip()
        reference_text = full_text.replace(train_text, "").strip()
        output_text = row[f'output_{algo_type}'].strip()
        if output_text != "":
            all_reference_text.append(reference_text)
            all_output_text.append(output_text)
    
    out = mauve.compute_mauve(
            p_text=all_reference_text,
            q_text=all_output_text,
            device_id=device_id,
            max_text_length=max_len,
            verbose=verbose,
            featurize_model_name=featurize_model_name,
            )
    return out.mauve

if __name__ == '__main__':
    algo_type = 'sample'
    file_path = 'results/C4/sample/2024-04-30-01:15:50/dataset_output.jsonl'
    df = pd.read_json(file_path, lines=True)
    avg_mauve_score = get_mauve_score_from_C4(df, algo_type)
    print('average mauve score: ', avg_mauve_score)
