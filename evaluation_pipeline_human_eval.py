import os
import json
import pandas as pd
import argparse
import numpy as np

from pathlib import Path
from evaluation_metrics.detection import get_detection_score
from evaluation_metrics.entropy import get_entropy_percentile
from evaluation_metrics.pass_k import get_pass_score


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
        default='sample', # sample, green, dipmark, delta, oracle
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
        default=4096,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    max_length = args.max_length

    algo_type = args.algo_type 
    algo_use_watermark = args.algo_use_watermark
    if algo_use_watermark:
        algo_type_watermark = algo_type   
    else:
        algo_type_watermark = f'wo_{algo_type}'   

    if args.algo_type == 'sample' and args.sample_threshold:
        parent_folder_path = Path(f'code_results/human_eval/sample_threshold')
    else:
        parent_folder_path = Path(f'code_results/human_eval/{algo_type}')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)
    folder_names = folder_names[:]

    # evaluate all folders of given algorithm
    for folder_name in folder_names:
        print('current folder: ', folder_name)
        folder_path = os.path.join(parent_folder_path, folder_name)

        try:
            df = pd.read_json(os.path.join(folder_path, f'dataset_output.jsonl'), lines=True)
        except:
            df = None
            print(f'empty dataset output of {algo_type_watermark}')

        # calcualte pass score
        pass_k, pass_variance = None, None
        pass_k, pass_variance = get_pass_score(folder_path, algo_type_watermark)

        # calculate detection score
        detection_score_dict = {}
        if algo_type_watermark in ['sample', 'sample_threshold', 'green', 'dipmark']:
            target_fpr_list = [0.01, 0.05, 0.1]
            detection_z_score_thresholds = [1.5, 1.9] + np.arange(2, 4.1, 0.1).tolist()
            for detection_z_score_threshold in detection_z_score_thresholds:
                detection_z_score_threshold = round(detection_z_score_threshold, 1)
                per_detection_score = get_detection_score(df, algo_type_watermark, detection_z_score_threshold, target_fpr_list)
                detection_score_dict[f'z_score_{detection_z_score_threshold}'] = per_detection_score

        # calcuate entropy percdentile and mrr
        try:
            entropy_5th_percentile = get_entropy_percentile(df, algo_type_watermark, 5)
            entropy_8th_percentile = get_entropy_percentile(df, algo_type_watermark, 8)
            entropy_10th_percentile = get_entropy_percentile(df, algo_type_watermark, 10)
        except:
            entropy_5th_percentile = None
            entropy_8th_percentile = None
            entropy_10th_percentile = None
            print(f'empty entropy of {algo_type_watermark}')

        # record evaluation results
        term_width = 80
        print('-'*term_width)
        results = {'algorithm type': algo_type,
                   'algorithm use watermark': algo_use_watermark,
                   'pass score': pass_k,
                   'pass variance': pass_variance,
                   'detection score': detection_score_dict,
                   'entropy 5th percentile': entropy_5th_percentile,
                   'entropy 8th percentile': entropy_8th_percentile,
                   'entropy 10th percentile': entropy_10th_percentile,
                   }
        with open(f'{folder_path}/eval_results_{algo_use_watermark}.json', 'w') as file:
            json.dump(results, file, indent=4)
        print(results)
