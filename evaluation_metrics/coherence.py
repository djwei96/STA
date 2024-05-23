"""
Adapted from the eval script in https://github.com/XiangLi1999/ContrastiveDecoding
requirement: pip install simcse (https://github.com/princeton-nlp/SimCSE)
"""
import os
import numpy as np
import pandas as pd

from simcse import SimCSE


def get_coherence_score(df, algo_type, simcse_model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    simcse_model = SimCSE(simcse_model_name)
    similarities = simcse_model.similarity(df['prompt'].to_list(), df[f'output_{algo_type}'].to_list())
    similarities = np.array(similarities)
    avg_coherence_score = similarities.trace() / len(similarities) 

    return avg_coherence_score

if __name__ == '__main__':
    algo_type = 'sample'
    file_path = 'results/C4/sample/2024-04-30-01:15:50/dataset_output.jsonl'
    df = pd.read_json(file_path, lines=True)
    avg_coherence_score = get_coherence_score(df, algo_type)
    print(avg_coherence_score)
