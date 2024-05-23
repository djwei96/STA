import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
from human_eval.data import read_problems

from tqdm import tqdm
from datetime import datetime
from openai import OpenAI


attack_prompts = {
    "0": "paraphrase the following paragraphs:\n",
    "1": "paraphrase the following paragraphs and try your best not to use the same bigrams from the original paragraphs\n",
    "2": "paraphrase the following paragraphs and try to keep the similar length to the original paragraphs\n",
    "3": "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n",
    "4": "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
}


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
        "--max_new_tokens",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default='C4',
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default='2024-04-29-15:53:51',
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
        "--use_canonical_or_reference",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--min_attack_length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0, # 0.4
    )
    parser.add_argument(
        "--attack_prompt_name",
        type=str,
        default='4',
    )
    args = parser.parse_args()
    if args.dataset_type not in ['C4', 'human_eval']:
        raise ValueError('wrong dataset type')

    if args.dataset_type == 'human_eval':
        problems = read_problems()
        canonical_solutions = [problems[key]['canonical_solution'] for key in problems.keys()]

    folder_name = args.folder_name
    algo_type = args.algo_type

    if args.dataset_type == 'C4':
        if args.sample_threshold and args.algo_type == 'sample':
            input_folder_path = f'results/C4/sample_threshold/{folder_name}'
            output_folder_path = f'attack_results/C4/gpt/sample_threshold/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'results/C4/sample_threshold/{folder_name}/config.json'
        else:
            input_folder_path = f'results/C4/{algo_type}/{folder_name}'
            output_folder_path = f'attack_results/C4/gpt/{algo_type}/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'results/C4/{algo_type}/{folder_name}/config.json'
    elif args.dataset_type == 'human_eval':
        if args.sample_threshold and args.algo_type == 'sample':
            input_folder_path = f'code_results/human_eval/sample_threshold/{folder_name}'
            output_folder_path = f'attack_results/human_eval/gpt/sample_threshold/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'code_results/human_eval/sample_threshold/{folder_name}/config.json'
        else:
            input_folder_path = f'code_results/human_eval/{algo_type}/{folder_name}'
            output_folder_path = f'attack_results/human_eval/gpt/{algo_type}/{folder_name}/att-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
            config_file_path = f'code_results/human_eval/{algo_type}/{folder_name}/config.json'

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
    #df = df.head(2)
    with open(config_file_path, 'r') as f:
        config_dict = json.load(f)

    if args.attack_prompt_name in attack_prompts:
        attack_prompt = attack_prompts[args.attack_prompt_name]
    else:
        print('wrong template name')
        sys.exit(0)

    client = OpenAI(api_key = "")
    total_attack_reviews_num = 0
    once_time = time.time()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # attack watermark
        total_attack_reviews_num += 1
        output_text_watermark = row[f'output_{algo_type}']
        messages_watermark=[
                {"role": "user", "content": attack_prompt + output_text_watermark},
                ]
        try:
            response = client.chat.completions.create(
                model = 'gpt-3.5-turbo',
                messages = messages_watermark,
                temperature=args.temperature,
                #max_tokens = args.max_new_tokens
            )
        except Exception as e:
            print('Error:', e)
            print('Number of', total_attack_reviews_num, 'not generated')

        if total_attack_reviews_num % 10 == 0:
            print('Current execution time of ', total_attack_reviews_num, ' : ', time.time() - once_time)
            once_time = time.time()
            time.sleep(10)

        df.at[index, f'attack_output_{algo_type}'] = response.choices[0].message.content
        df.at[index, f'attack_output_{algo_type}_token_num'] =  response.usage.completion_tokens

        # attack wo watermark
        total_attack_reviews_num += 1
        if args.use_canonical_or_reference == True:
            if args.dataset_type == 'C4':
                output_text_wo_watermark = row[f'reference_{algo_type}']
            elif args.dataset_type == 'human_eval':
                output_text_wo_watermark = canonical_solutions[index%len(canonical_solutions)]
        else:
            output_text_wo_watermark = row[f'output_wo_{algo_type}']

        messages_wo_watermark=[
                {"role": "user", "content": attack_prompt + output_text_wo_watermark},
                ]
        try:
            response = client.chat.completions.create(
                model = 'gpt-3.5-turbo',
                messages = messages_wo_watermark,
                temperature=args.temperature,
                #max_tokens = args.max_new_tokens
            )
        except Exception as e:
            print('Error:', e)
            print('Number of', total_attack_reviews_num, 'not generated')

        if total_attack_reviews_num % 10 == 0:
            print('Current execution time of ', total_attack_reviews_num, ' : ', time.time() - once_time)
            once_time = time.time()
            time.sleep(10)

        df.at[index, f'attack_output_wo_{algo_type}'] = response.choices[0].message.content
        df.at[index, f'attack_output_wo_{algo_type}_token_num'] =  response.usage.completion_tokens

        #if total_attack_reviews_num == 2:
        #    break 

    time.sleep(10)
    df.to_json(os.path.join(output_folder_path, f'dataset_attack_output.jsonl'), orient='records', lines=True)
    with open(os.path.join(output_folder_path, 'attack_config.json'), 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    with open(os.path.join(output_folder_path, 'algo_config.json'), 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)
    
