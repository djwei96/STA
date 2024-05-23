import os
import json
import torch
import pandas as pd
import argparse
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation_metrics.detection import get_detection_score
from evaluation_metrics.entropy import get_entropy_percentile
from evaluation_metrics.mrr import get_mrr_score
from evaluation_metrics.ppl import get_ppl_from_token_ids, get_ppl_from_text
from evaluation_metrics.repetition_diversity import get_diversity_score
from evaluation_metrics.coherence import get_coherence_score
from evaluation_metrics.mauve_score import get_mauve_score_from_C4


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
        default='9',
    )
    parser.add_argument(
        "--algo_type",
        type=str,
        default='sample', # sample, green, dipmark, delta
    )
    parser.add_argument(
        "--sample_threshold",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--algo_use_watermark",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--ppl_use_token_ids",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--filter_long_text",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--filter_long_text_num",
        type=int,
        default=500,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = args.max_length
    ppl_use_token_ids = args.ppl_use_token_ids
    filter_long_text = args.filter_long_text
    filter_long_text_num = args.filter_long_text_num
    cache_dir = 'hf_models'

    algo_type = args.algo_type 
    algo_use_watermark = args.algo_use_watermark
    if algo_use_watermark:
        algo_type_watermark = algo_type   
    else:
        algo_type_watermark = f'wo_{algo_type}'   

    if args.algo_type == 'sample' and args.sample_threshold:
        parent_folder_path = Path(f'results/C4/sample_threshold')
    else:
        parent_folder_path = Path(f'results/C4/{algo_type}')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)
    #folder_names = folder_names[6:]
    print(folder_names)

    with open('data/C4/sampled_dataset.json', "r") as input_file:
        data = json.load(input_file)

    with open('data/C4/sampled_dataset_train.json', "r") as input_file:
        train_data = json.load(input_file)

    # evaluate all folders of given algorithm
    for folder_name in folder_names:
        print('current folder: ', folder_name)
        folder_path = os.path.join(parent_folder_path, folder_name)
        with open(os.path.join(folder_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        df = pd.read_json(os.path.join(folder_path, f'dataset_output.jsonl'), lines=True)

        # get last 200 tokens as reference text from C4
        for index, row in df.iterrows():
            full_text = data[index].strip()
            train_text = train_data[index].strip()
            reference_text = full_text.replace(train_text, "") # strip()
            df.loc[index, f'reference_{algo_type}'] = reference_text
        df.to_json(os.path.join(folder_path, 'dataset_output.jsonl'), orient='records', lines=True)
        
        # if filter first 500 generated long text with around 200 tokens
        if filter_long_text:
            df[f'output_len_{algo_type_watermark}'] = df[f'stats_{algo_type_watermark}'].apply(len)
            df = df.sort_values(by=f'output_len_{algo_type_watermark}', ascending=False, kind='stable')
            df = df.head(filter_long_text_num)
            df.to_json(os.path.join(folder_path, f'dataset_output_n_{filter_long_text_num}.jsonl'), orient='records', lines=True)
        else:
            filter_long_text_num = 1000
        print('number of generated text', len(df))

        # claculate ppl 
        ppl_model_name, avg_ppl = 'no', None
        gen_model_name = config_dict['model_name_or_path']
        if gen_model_name == 'meta-llama/Llama-2-7b-hf':
            ppl_model_name = 'meta-llama/Llama-2-13b-hf'
        elif gen_model_name == 'meta-llama/Llama-2-7b-chat-hf':
            #ppl_model_name = 'meta-llama/Llama-2-13b-chat-hf'
            continue
        else:
            continue
        
        ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_name, cache_dir=cache_dir)
        ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_name, truncation_side='left', cache_dir=cache_dir)
        ppl_model.eval()
        ppl_model.to(device)

        if ppl_use_token_ids:
            avg_ppl, df_ppl = get_ppl_from_token_ids(df, algo_type_watermark, ppl_model, device, max_length)
        else:
            avg_ppl, df_ppl = get_ppl_from_text(df, algo_type_watermark, ppl_model, ppl_tokenizer, device, max_length)

        # calculate detection score
        detection_score_dict = {}
        if algo_type_watermark in ['sample', 'sample_threshold', 'green', 'dipmark']:
            target_fpr_list = [0.01, 0.05, 0.1]
            detection_z_score_thresholds = [1.5, 1.9] + np.arange(2, 4.1, 0.1).tolist()
            for detection_z_score_threshold in detection_z_score_thresholds:
                detection_z_score_threshold = round(detection_z_score_threshold, 1)
                per_detection_score = get_detection_score(df, algo_type_watermark, detection_z_score_threshold, target_fpr_list)
                detection_score_dict[f'z_score_{detection_z_score_threshold}'] = per_detection_score

        # calculate diversity, coherence and mauve
        try:
            avg_diversity, avg_log_diversity = get_diversity_score(df, algo_type_watermark)
            avg_coherence = get_coherence_score(df, algo_type_watermark)
            avg_mauve_score = get_mauve_score_from_C4(df, algo_type_watermark)
        except:
            avg_diversity, avg_log_diversity = None, None
            avg_coherence = None
            avg_mauve_score = None

        # calcuate entropy percdentile and mrr
        try:
            entropy_5th_percentile = get_entropy_percentile(df, algo_type_watermark, 5)
            entropy_8th_percentile = get_entropy_percentile(df, algo_type_watermark, 8)
            entropy_10th_percentile = get_entropy_percentile(df, algo_type_watermark, 10)
            avg_mrr = get_mrr_score(df, algo_type_watermark, entropy_10th_percentile)
        except:
            entropy_5th_percentile = None
            entropy_8th_percentile = None
            entropy_10th_percentile = None
            avg_mrr = None

        # record evaluation results
        term_width = 80
        print('-'*term_width)
        results = {'algorithm type': algo_type,
                   'algorithm use watermark': algo_use_watermark,
                   'detection score': detection_score_dict,
                   'entropy 5th percentile': entropy_5th_percentile,
                   'entropy 8th percentile': entropy_8th_percentile,
                   'entropy 10th percentile': entropy_10th_percentile,
                   'average mrr': avg_mrr,
                   'ppl use token ids': ppl_use_token_ids,
                   'ppl model name': ppl_model_name,
                   'average ppl': avg_ppl,
                   'average diversity': avg_diversity,
                   'average log diversity': avg_log_diversity,
                   'average coherence score': avg_coherence,
                   'average mauve score': avg_mauve_score,
                   }
        df_ppl.to_json(os.path.join(folder_path, 
                                    f'df_eval_results_{algo_type_watermark}_n_{filter_long_text_num}_ppl_{ppl_model_name.split('/')[-1]}_id_{ppl_use_token_ids}.jsonl'), 
                                    orient='records', 
                                    lines=True)
        with open(os.path.join(folder_path, 
                               f'eval_results_{algo_type_watermark}_n_{filter_long_text_num}_ppl_{ppl_model_name.split('/')[-1]}_id_{ppl_use_token_ids}.json'), 'w') as file:
            json.dump(results, file, indent=4)
        print(results)
