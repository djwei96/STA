import argparse
import json
import nltk
import time
import os
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

from nltk.tokenize import sent_tokenize
from datetime import datetime
from human_eval.data import read_problems

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

nltk.download("punkt")

# A Watermark for Large Language Models
# https://github.com/jwkirchenbauer/lm-watermarking

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


def generate_dipper_paraphrases(
    input_text,
    prompt,
    model,
    tokenizer,
    max_new_tokens,
    do_sample,
    top_p,
    use_prompt,
    sent_interval,
    lex,
    order,
):
    lex_code = int(100 - lex)
    order_code = int(100 - order)
    input_text = " ".join(input_text.split())
    sentences = sent_tokenize(input_text)

    #TODO: lex_code: 60, order_code: 60
    output_text = ""
    for sent_idx in range(0, len(sentences), sent_interval):
        curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
        if use_prompt:
            prompt = " ".join(prompt.replace("\n", " ").split())
            final_input_text = f"lexical = {lex_code}, order = {order_code} {prompt} <sent> {curr_sent_window} </sent>"
        else:
            final_input_text = f"lexical = {lex_code}, order = {order_code} <sent> {curr_sent_window} </sent>"

        final_input = tokenizer([final_input_text], return_tensors="pt")
        final_input = {k: v.cuda() for k, v in final_input.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **final_input, do_sample=do_sample, top_p=top_p, top_k=None, max_new_tokens=max_new_tokens)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text += " " + outputs[0]
    return output_text


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
        default='4',
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default='C4',
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default='2024-05-11-11:33:54',
    )
    parser.add_argument(
        "--algo_type",
        type=str,
        default='dipmark',
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
        "--use_canonical_or_reference",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--min_attack_length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--do_sample",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--use_prompt",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--lex_diversity", #The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
        type=int,
        default=20,
    )
    parser.add_argument(
        "--order_diversity", #The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
        type=int,
        default=20,
    )
    parser.add_argument(
        "--sent_interval",
        type=int,
        default=3,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.dataset_type not in ['C4', 'human_eval']:
        raise ValueError('wrong dataset type')

    if args.dataset_type == 'human_eval':
        problems = read_problems()
        canonical_solutions = [problems[key]['canonical_solution'] for key in problems.keys()]

    # lexical diversity: 60, order diversity: 0
    # lexical diversity: 60, order diversity: 20
    # lexical diversity: 20, order diversity: 20
    # lexical diversity: 20, order diversity: 0
    assert args.lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
    assert args.order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

    folder_name = args.folder_name
    algo_type = args.algo_type

    if args.dataset_type == 'C4':
        if args.sample_threshold and args.algo_type == 'sample':
            input_folder_path = f'results/C4/sample_threshold/{folder_name}'
            output_folder_path = f'attack_results/C4/dipper/sample_threshold/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'results/C4/sample_threshold/{folder_name}/config.json'
        else:
            input_folder_path = f'results/C4/{algo_type}/{folder_name}'
            output_folder_path = f'attack_results/C4/dipper/{algo_type}/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'results/C4/{algo_type}/{folder_name}/config.json'
    elif args.dataset_type == 'human_eval':
        if args.sample_threshold and args.algo_type == 'sample':
            input_folder_path = f'code_results/human_eval/sample_threshold/{folder_name}'
            output_folder_path = f'attack_results/human_eval/dipper/sample_threshold/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'code_results/human_eval/sample_threshold/{folder_name}/config.json'
        else:
            input_folder_path = f'code_results/human_eval/{algo_type}/{folder_name}'
            output_folder_path = f'attack_results/human_eval/dipper/{algo_type}/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'code_results/human_eval/{algo_type}/{folder_name}/config.json'

    print(input_folder_path)
    os.makedirs(output_folder_path)
    
    if args.filter_long_text:
        df = pd.read_json(os.path.join(input_folder_path, f'dataset_output_n_500.jsonl'), lines=True)
    else:
        df = pd.read_json(os.path.join(input_folder_path, f'dataset_output.jsonl'), lines=True)
        args.min_attack_length = args.max_new_tokens - 25
        df[f'output_len_{algo_type}'] = df[f'stats_{algo_type}'].apply(len)
        df[f'output_len_wo_{algo_type}'] = df[f'stats_wo_{algo_type}'].apply(len)
        df = df[(df[f'output_len_{algo_type}'] >= args.min_attack_length) & (df[f'output_len_wo_{algo_type}'] >= args.min_attack_length)]

    print(len(df)) 
    #df = df.head(3)
    with open(config_file_path, 'r') as f:
        config_dict = json.load(f)

    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", cache_dir='hf_models')
    model = T5ForConditionalGeneration.from_pretrained("kalpeshk2011/dipper-paraphraser-xxl", cache_dir='hf_models')
    model.cuda()
    model.eval()

    for index, row in tqdm(df.iterrows(), total=len(df)):
        prompt=row['prompt']
        output_text_watermark = row[f'output_{algo_type}']
        if args.use_canonical_or_reference == True:
            if args.dataset_type == 'C4':
                output_text_wo_watermark = row[f'reference_{algo_type}']
            elif args.dataset_type == 'human_eval':
                output_text_wo_watermark = canonical_solutions[index%len(canonical_solutions)]
        else:
            output_text_wo_watermark = row[f'output_wo_{algo_type}']

        attack_watermark_output_text = generate_dipper_paraphrases(input_text=output_text_watermark, 
                                                       prompt=prompt,
                                                       model=model,
                                                       tokenizer=tokenizer, 
                                                       max_new_tokens=args.max_new_tokens,
                                                       do_sample=args.do_sample,
                                                       top_p=args.top_p,
                                                       use_prompt=args.use_prompt,
                                                       lex=args.lex_diversity,
                                                       order=args.order_diversity,
                                                       sent_interval=args.sent_interval)
        attack_wo_watermark_output_text = generate_dipper_paraphrases(input_text=output_text_wo_watermark, 
                                                       prompt=prompt,
                                                       model=model,
                                                       tokenizer=tokenizer, 
                                                       max_new_tokens=args.max_new_tokens,
                                                       do_sample=args.do_sample,
                                                       top_p=args.top_p,
                                                       use_prompt=args.use_prompt,
                                                       lex=args.lex_diversity,
                                                       order=args.order_diversity,
                                                       sent_interval=args.sent_interval)
        df.at[index, f'attack_output_{algo_type}'] = attack_watermark_output_text # strip()
        df.at[index, f'attack_output_wo_{algo_type}'] = attack_wo_watermark_output_text # strip()

        print('attack with wartermark: \n')
        print(attack_watermark_output_text)
        print('\n')
        print('attack without wartermark: \n')
        print(attack_wo_watermark_output_text)

    df.to_json(os.path.join(output_folder_path, f'dataset_attack_output.jsonl'), orient='records', lines=True)
    with open(os.path.join(output_folder_path, 'attack_config.json'), 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    with open(os.path.join(output_folder_path, 'algo_config.json'), 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)
