import os
import sys
import json
import scipy
import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
from evaluation_metrics.detection import get_detection_score

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append('lm_watermarking')
from lm_watermarking.watermark_processor import WatermarkDetector

sys.path.append('sample_watermark')
from sample_watermark.sample_watermark_processor import SampleWatermarkDetector

from DiPmark.dipmark import (
    Dip_Reweight,
    WatermarkLogitsProcessor,
    PrevN_ContextCodeExtractor,
)


def filter_paraphrase_text(text):
    if not text:
        return None
    colon_index = text.find(':')
    if colon_index != -1:
        return text[colon_index + 1:].strip()
    else:
        return text.strip()


def compute_z_score(token_quantiles, T, gamma):
    # count refers to number of green tokens, T is total number of tokens
    observed_count = sum(1 for x in token_quantiles if x > gamma)
    numer = observed_count - gamma * T
    denom = sqrt(T * gamma * (1 - gamma))
    z_score = numer / denom
    return z_score


def compute_p_value(z_score):
    p_value = scipy.stats.norm.sf(z_score)
    return p_value


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    sample_threshold = True

    #algo_type = 'sample'
    #data_name = '2024-04-30-01:15:50'
    #data_name =  '2024-05-09-08:40:16' #'2024-05-09-08:40:16' 16 '2024-05-09-03:43:33' 4 '2024-05-09-05:19:55' (1.35)
    #data_name = '' '2024-05-09-20:33:58', '2024-05-10-06:16:39' (1.95)

    #algo_type = 'green'
    #data_name = '2024-05-10-15:35:38' #2024-04-30-04:37:25, 2024-04-30-15:40:53, 2024-04-30-20:06:29, 2024-05-10-15:35:38

    algo_type = 'dipmark'
    #data_name = '2024-05-11-16:14:38' #2024-05-11-11:33:54, 2024-05-11-16:14:38, 2024-05-11-20:52:51 (context = 5)
    data_name = '2024-05-12-04:19:38' #2024-05-12-04:19:38, 2024-05-12-08:53:19, 2024-05-12-13:23:03 (context = 1)

    print(data_name)

    attack_type = 'cp'
    if sample_threshold == True and algo_type == 'sample':
        parent_folder_path = f'attack_results/C4/{attack_type}/sample_threshold/{data_name}'
        config_file_path = f'results/C4/sample_threshold/{data_name}/config.json'
    else:
        parent_folder_path = f'attack_results/C4/{attack_type}/{algo_type}/{data_name}'
        config_file_path = f'results/C4/{algo_type}/{data_name}/config.json'
    with open(config_file_path, 'r') as f:
        config_dict = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config_dict['model_name_or_path'], cache_dir='hf_models')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if algo_type == 'sample':
        watermark_detector = SampleWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                           n_sample_per_token=config_dict['n_sample_per_token'],
                                           z_threshold=config_dict['detection_z_threshold'],
                                           gamma=config_dict['gamma'])
    elif algo_type == 'green':
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                     gamma=config_dict['gamma'],
                                     seeding_scheme=config_dict['seeding_scheme'],
                                     device=device,
                                     tokenizer=tokenizer,
                                     z_threshold=config_dict['detection_z_threshold'],
                                     normalizers=config_dict['normalizers'],
                                     ignore_repeated_bigrams=config_dict['ignore_repeated_bigrams'],
                                     select_green_tokens=config_dict['select_green_tokens'])
    elif algo_type == 'dipmark':
        pass
    else:
        print('wrong watermark algorithm')
        sys.exit(0)

    path = Path(parent_folder_path)
    folder_names = [dir.name for dir in path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)
    #folder_names = folder_names[:-1]
    print(folder_names)

    min_detect_length = 1
    all_z_scores_watermark = []
    all_z_scores_no_watermark = []

    for folder_name in folder_names:
        folder_path = f'{path}/{folder_name}'
        df = pd.read_json(os.path.join(folder_path, f'dataset_attack_output.jsonl'), lines=True)
        df.replace("nan", np.nan, inplace=True)
        df = df.dropna()
        for index, row in tqdm(df.iterrows(), total=len(df)):
            input_text_watermark = row[f'attack_output_{algo_type}'] 
            input_text_wo_watermark = row[f'attack_output_wo_{algo_type}'] 

            if algo_type == 'sample':
                try:
                    z_score_watermark, p_value_watermark = watermark_detector.detect(input_text_watermark, 2048, device, tokenizer)
                    z_score_wo_watermark, p_value_wo_watermark = watermark_detector.detect(input_text_wo_watermark, 2048, device, tokenizer)
                except:
                    z_score_watermark, p_value_watermark = 0, 0
                    z_score_wo_watermark, p_value_wo_watermark = 0, 0

            elif algo_type == 'green': 
                try:
                    output_dict_watermark = watermark_detector.detect(input_text_watermark)
                    z_score_watermark, p_value_watermark = output_dict_watermark['z_score'], output_dict_watermark['p_value']
                    output_dict_wo_watermark = watermark_detector.detect(input_text_wo_watermark)
                    z_score_wo_watermark, p_value_wo_watermark = output_dict_wo_watermark['z_score'], output_dict_wo_watermark['p_value']
                except:
                    z_score_watermark, p_value_watermark = 0, 0
                    z_score_wo_watermark, p_value_wo_watermark = 0, 0

            elif algo_type == 'dipmark':
                inputs_watermark = tokenizer(input_text_watermark, return_tensors="pt", 
                                                add_special_tokens=True, truncation=True, 
                                                max_length=2048)
                input_ids_watermark = inputs_watermark['input_ids'].to(device)

                token_quantiles_watermark = []
                try:
                    dipmark_wp = WatermarkLogitsProcessor(
                        b"15485863",
                        Dip_Reweight(config_dict['alpha']),
                        PrevN_ContextCodeExtractor(config_dict['prevN']))
                    
                    input_ids_for_quantile = input_ids_watermark.clone()
                    for i in range(input_ids_watermark.shape[1]):
                        current_token = input_ids_watermark[:, i]
                        current_token_quantile = dipmark_wp.get_green_token_quantile(input_ids_for_quantile, len(tokenizer.vocab), current_token)
                        token_quantiles_watermark.append(current_token_quantile[0].item())
                        input_ids_for_quantile =  torch.cat((input_ids_for_quantile, current_token.unsqueeze(0)), dim=-1)

                    z_score_watermark = compute_z_score(token_quantiles_watermark, input_ids_watermark.shape[1], config_dict['gamma'])
                    p_value_watermark = compute_p_value(z_score_watermark)
                except:
                    z_score_watermark, p_value_watermark = 0, 0                             
                            
                inputs_wo_watermark = tokenizer(input_text_wo_watermark, return_tensors="pt", 
                                                add_special_tokens=True, truncation=True, 
                                                max_length=2048)
                input_ids_wo_watermark = inputs_wo_watermark['input_ids'].to(device)
                token_quantiles_wo_watermark = []
                try:
                    dipmark_wp = WatermarkLogitsProcessor(
                        b"15485863",
                        Dip_Reweight(config_dict['alpha']),
                        PrevN_ContextCodeExtractor(config_dict['prevN']))

                    input_ids_for_quantile = input_ids_wo_watermark.clone()
                    for i in range(input_ids_wo_watermark.shape[1]):
                        current_token = input_ids_wo_watermark[:, i]
                        current_token_quantile = dipmark_wp.get_green_token_quantile(input_ids_for_quantile, len(tokenizer.vocab), current_token)
                        token_quantiles_wo_watermark.append(current_token_quantile[0].item())
                        input_ids_for_quantile =  torch.cat((input_ids_for_quantile, current_token.unsqueeze(0)), dim=-1)

                    z_score_wo_watermark = compute_z_score(token_quantiles_wo_watermark, input_ids_wo_watermark.shape[1], config_dict['gamma'])
                    p_value_wo_watermark = compute_p_value(z_score_wo_watermark)
                except:
                    z_score_watermark, p_value_watermark = 0, 0                             

            all_z_scores_watermark.append(z_score_watermark)
            all_z_scores_no_watermark.append(z_score_wo_watermark)
            df.at[index, f'z_score_{algo_type}'] = z_score_watermark
            df.at[index, f'z_score_wo_{algo_type}'] = z_score_wo_watermark

        detection_score_dict = {}
        target_fpr_list = [0.01, 0.05, 0.1]
        detection_z_score_thresholds = [1.5, 1.9] + np.arange(2, 4.1, 0.1).tolist()
        for detection_z_score_threshold in detection_z_score_thresholds:
            detection_z_score_threshold = round(detection_z_score_threshold, 1)
            per_detection_score = get_detection_score(df, algo_type, detection_z_score_threshold, target_fpr_list)
            detection_score_dict[f'z_score_{detection_z_score_threshold}'] = per_detection_score
        
        with open(os.path.join(folder_path, 'eval_results_detect.json'), 'w') as file:
            json.dump(detection_score_dict, file, indent=4)
        print(detection_score_dict)