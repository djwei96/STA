import os
import json
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from evaluation_metrics.avg_bert_score import get_average_bert_score


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
        default='1',
    )
    parser.add_argument(
        "--algo_type",
        type=str,
        default='dipmark', # sample, green, dipmark, delta, oracle
    )
    parser.add_argument(
        "--sample_threshold",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    cache_dir = 'hf_models'
    algo_type = args.algo_type
    sample_threshold = args.sample_threshold
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if sample_threshold == True and algo_type == 'sample':
        parent_folder_path = Path(f'results/C4/sample_threshold')
    else:
        parent_folder_path = Path(f'results/C4/{algo_type}')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)

    #folder_names = ['2024-05-12-00:11:03']
    for folder_name in tqdm(folder_names):
        folder_path = os.path.join(parent_folder_path, folder_name)
        df = pd.read_json(os.path.join(folder_path, f'dataset_output_n_500.jsonl'), lines=True)
        with open(os.path.join(folder_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)

        output_text_watermark = df[f'output_{algo_type}'].to_list()
        output_reference = df[f'reference_{algo_type}'].to_list()

        average_precision, average_recall, average_f1 = get_average_bert_score(output_text_watermark, output_reference)
        
        term_width = 80
        print('-'*term_width)
        results = {'average precision': average_precision,
                   'average recall': average_recall,
                   'average f1': average_f1
                   }
        with open(os.path.join(folder_path, f'eval_bert_score.json'), 'w') as file:
            json.dump(results, file, indent=4)
