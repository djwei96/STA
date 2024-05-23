import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import AutoTokenizer
from human_eval.data import read_problems

# A Watermark for Large Language Models
# https://github.com/jwkirchenbauer/lm-watermarking


def single_insertion(
    attack_len,
    min_attack_len,
    tokenized_wm_output,  # src
    tokenized_no_wm_output,  # dst
    seed = 42
):
    if seed is not None:
        torch.manual_seed(seed)
    
    flat_tokenized_wm_output = tokenized_wm_output.flatten()
    flat_tokenized_no_wm_output = tokenized_no_wm_output.flatten()
    indices_to_replace = torch.randperm(min_attack_len)[:attack_len]
    flat_tokenized_wm_output[indices_to_replace] = flat_tokenized_no_wm_output[indices_to_replace]
    new_tokenized_wm_output = flat_tokenized_wm_output.view(tokenized_wm_output.size())
    return new_tokenized_wm_output


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default='0',
    )
    parser.add_argument(
        "--algo_type",
        type=str,
        default='sample',
    )
    parser.add_argument(
        "--filter_long_text",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--sample_threshold",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default='C4',
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default='2024-04-30-01:15:50',
    )
    parser.add_argument(
        "--use_canonical_or_reference",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--min_attack_length",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--attack_insertion_length",
        type=str,
        default='25%',
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.dataset_type not in ['C4', 'human_eval']:
        raise ValueError('wrong dataset type')

    if args.dataset_type == 'human_eval':
        problems = read_problems()
        canonical_solutions = [problems[key]['canonical_solution'] for key in problems.keys()]

    folder_name = args.folder_name
    print(folder_name)
    algo_type = args.algo_type
    cache_dir = 'hf_models'

    if args.dataset_type == 'C4':
        if args.sample_threshold and args.algo_type == 'sample':
            input_folder_path = f'results/C4/sample_threshold/{folder_name}'
            output_folder_path = f'attack_results/C4/cp/sample_threshold/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'results/C4/sample_threshold/{folder_name}/config.json'
        else:
            input_folder_path = f'results/C4/{algo_type}/{folder_name}'
            output_folder_path = f'attack_results/C4/cp/{algo_type}/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'results/C4/{algo_type}/{folder_name}/config.json'
    elif args.dataset_type == 'human_eval':
        if args.sample_threshold and args.algo_type == 'sample':
            input_folder_path = f'code_results/human_eval/sample_threshold/{folder_name}'
            output_folder_path = f'attack_results/human_eval/cp/sample_threshold/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'code_results/human_eval/sample_threshold/{folder_name}/config.json'
        else:
            input_folder_path = f'code_results/human_eval/{algo_type}/{folder_name}'
            output_folder_path = f'attack_results/human_eval/cp/{algo_type}/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'code_results/human_eval/{algo_type}/{folder_name}/config.json'

    os.makedirs(output_folder_path)

    if args.filter_long_text:
        df = pd.read_json(os.path.join(input_folder_path, f'dataset_output_n_500.jsonl'), lines=True)
    else:
        df = pd.read_json(os.path.join(input_folder_path, f'dataset_output.jsonl'), lines=True)

    with open(config_file_path, 'r') as f:
        config_dict = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(config_dict['model_name_or_path'], cache_dir='hf_models')

    # fixed copy paste attack for output w/o watermark with max tokens only
    if "%" in args.attack_insertion_length:
        args.attack_insertion_length = int(args.attack_insertion_length[:-1])/100
    else:
        args.attack_insertion_length = int(args.attack_insertion_length)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        output_text_watermark = row[f'output_{algo_type}']
        if args.use_canonical_or_reference == True:
            if args.dataset_type == 'C4':
                output_text_wo_watermark = row[f'reference_{algo_type}']
            elif args.dataset_type == 'human_eval':
                output_text_wo_watermark = canonical_solutions[index%len(canonical_solutions)]
        else:
            output_text_wo_watermark = row[f'output_wo_{algo_type}']

        output_text_watermark_ids = tokenizer(output_text_watermark, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        output_text_wo_watermark_ids = tokenizer(output_text_wo_watermark, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        min_attack_length = args.min_attack_length
        if min(output_text_watermark_ids.shape[-1], output_text_wo_watermark_ids.shape[-1]) < min_attack_length:
            continue

        if isinstance(args.attack_insertion_length, int):
            attack_watermark_output_ids = single_insertion(args.attack_insertion_length, 
                                                   min_attack_length,
                                                   output_text_watermark_ids.clone(), 
                                                   output_text_wo_watermark_ids.clone())
            attack_wo_watermark_output_ids = single_insertion(args.attack_insertion_length, 
                                                   min_attack_length,
                                                   output_text_wo_watermark_ids.clone(), 
                                                   output_text_watermark_ids.clone())
        else:
            attack_watermark_output_ids = single_insertion(int(output_text_watermark_ids.shape[-1]*args.attack_insertion_length), 
                                                   min_attack_length,
                                                   output_text_watermark_ids.clone(), 
                                                   output_text_wo_watermark_ids.clone())
            attack_wo_watermark_output_ids = single_insertion(int(output_text_wo_watermark_ids.shape[-1]*args.attack_insertion_length), 
                                                   min_attack_length,
                                                   output_text_wo_watermark_ids.clone(), 
                                                   output_text_watermark_ids.clone())

        ''' 
        print(torch.equal(output_text_watermark_ids, attack_watermark_output_ids))
        print(torch.equal(output_text_wo_watermark_ids, attack_wo_watermark_output_ids))
        print(torch.equal(attack_watermark_output_ids, attack_wo_watermark_output_ids))
        print(output_text_watermark_ids)
        print(attack_watermark_output_ids)
        print((output_text_watermark_ids != attack_watermark_output_ids).sum().item())
        print(output_text_wo_watermark_ids)
        print(attack_wo_watermark_output_ids)
        print((output_text_wo_watermark_ids != attack_wo_watermark_output_ids).sum().item())
        ''' 

        df.at[index, f'attack_output_{algo_type}'] = tokenizer.decode(attack_watermark_output_ids, skip_special_tokens=True)
        df.at[index, f'attack_output_wo_{algo_type}'] = tokenizer.decode(attack_wo_watermark_output_ids, skip_special_tokens=True)

    df.to_json(os.path.join(output_folder_path, f'dataset_attack_output.jsonl'), orient='records', lines=True)
    with open(os.path.join(output_folder_path, 'attack_config.json'), 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    with open(os.path.join(output_folder_path, 'algo_config.json'), 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)
