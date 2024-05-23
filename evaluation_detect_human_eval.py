import os
import sys
import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from scipy.stats import norm

from human_eval.data import read_problems
from evaluation_metrics.detection import get_tpr_and_f1_at_fpr 

sys.path.append('lm_watermarking')
from lm_watermarking.watermark_processor import WatermarkDetector

sys.path.append('sample_watermark')
from sample_watermark.sample_watermark_processor import SampleWatermarkDetector


def get_classification_score(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 'N/A'
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 'N/A'
    fpr = fp / (fp + tn) if (tp + fn) != 0 else 'N/A'
    fnr = fn / (fn + tp) if (tp + fn) != 0 else 'N/A'
    return {'f1': f1, 'auc': auc, 'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr}

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    algo_type = 'sample'
    sample_threshold = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_canonical = True
    use_pass_1 = False
    problems = read_problems()
    canonical_solutions = [problems[key]['canonical_solution'] for key in problems.keys()]
    batch_size = len(canonical_solutions)

    if sample_threshold == True and algo_type == 'sample':
        parent_folder_path = Path(f'code_results/human_eval/sample_threshold')
    else:
        parent_folder_path = Path(f'code_results/human_eval/{algo_type}')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)

    for folder_name in tqdm(folder_names):
        folder_path = os.path.join(parent_folder_path, folder_name)
        df_watermark = pd.read_json(os.path.join(folder_path, f'full_completions_{algo_type}.jsonl'), lines=True)
        df_wo_watermark = pd.read_json(os.path.join(folder_path, f'full_completions_wo_{algo_type}.jsonl'), lines=True)
        total_rows = len(df_watermark)
        assert total_rows % batch_size == 0

        with open(os.path.join(os.path.join(folder_path, 'config.json')), 'r') as f:
            config_dict = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(config_dict['model_name_or_path'], cache_dir='hf_models')
    
        if algo_type == 'sample':
            watermark_detector = SampleWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                               n_sample_per_token=config_dict['n_sample_per_token'],
                                               z_threshold=None,
                                               gamma=config_dict['gamma'])
        elif algo_type == 'green':
            watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                         gamma=config_dict['gamma'],
                                         seeding_scheme=config_dict['seeding_scheme'],
                                         device=device,
                                         tokenizer=tokenizer,
                                         z_threshold=None,
                                         normalizers=config_dict['normalizers'],
                                         ignore_repeated_bigrams=config_dict['ignore_repeated_bigrams'],
                                         select_green_tokens=config_dict['select_green_tokens'])
    
        # loop through each batch
        all_z_scores_watermark = []
        all_p_values_watermark = []
        all_z_scores_wo_watermark = []
        all_p_values_wo_watermark = []
        all_completions_watermark = df_watermark['completion'].to_list()
        all_completions_wo_watermark = df_wo_watermark['completion'].to_list()
        assert len(all_completions_watermark) == len(all_completions_wo_watermark)
        assert len(all_completions_watermark)%164 == 0

        for i in tqdm(range(0, total_rows)):
            completion_watermark = all_completions_watermark[i]
            if use_canonical:
                completion_wo_watermark = canonical_solutions[i%164]
            else:
                completion_wo_watermark = all_completions_wo_watermark[i]
    
            if algo_type == 'sample':
                try:
                    z_score_watermark, p_value_watermark = watermark_detector.detect(completion_watermark, 2048, device, tokenizer)
                except:
                    z_score_watermark = 0
                    p_value_watermark = 0
                all_z_scores_watermark.append(z_score_watermark)
                all_p_values_watermark.append(p_value_watermark)
    
                try:
                    z_score_wo_watermark, p_value_wo_watermark = watermark_detector.detect(completion_wo_watermark, 2048, device, tokenizer)
                except:
                    z_score_wo_watermark = 0
                    p_value_wo_watermark = 0
                all_z_scores_wo_watermark.append(z_score_wo_watermark)
                all_p_values_wo_watermark.append(p_value_wo_watermark)
            elif algo_type == 'green': 
                try:
                    z_score_watermark, p_value_watermark = watermark_detector.detect(completion_watermark)
                except:
                    z_score_watermark = 0
                    p_value_watermark = 0
                all_z_scores_watermark.append(z_score_watermark)
                all_p_values_watermark.append(p_value_watermark)
    
                try:
                    z_score_wo_watermark, p_value_wo_watermark = watermark_detector.detect(completion_wo_watermark)
                except:
                    z_score_wo_watermark = 0
                    p_value_wo_watermark = 0
                all_z_scores_wo_watermark.append(z_score_wo_watermark)
                all_p_values_wo_watermark.append(p_value_wo_watermark)
            else:
                sys.exit(0)

            if use_pass_1 and total_rows % 164 == 0:
                break # only keep pass@1
    
        # calculate overall score
        target_fpr_list = [0.01, 0.05, 0.1]
        detection_z_score_thresholds = [1.5, 1.9, 2]
        all_eval_results_dict = {}
        for detection_z_score_threshold in detection_z_score_thresholds:
            eval_results_dict = {}
            all_predicted_scores = all_z_scores_watermark +  all_z_scores_wo_watermark

            # label 1: watermark / label 0: no watermark
            predicted_watermark_labels = [1 if value > detection_z_score_threshold else 0 for value in all_z_scores_watermark]
            predicted_wo_watermark_labels = [1 if value > detection_z_score_threshold else 0 for value in all_z_scores_wo_watermark]
            all_predicted_labels = predicted_watermark_labels + predicted_wo_watermark_labels
    
            ture_watermark_labels = [1]*len(predicted_watermark_labels)
            ture_wo_watermark_labels = [0]*len(predicted_wo_watermark_labels)
            all_true_labels = ture_watermark_labels + ture_wo_watermark_labels
            eval_results_dict['all'] = get_classification_score(all_true_labels, all_predicted_labels)

            #TODO: under devlopment, need to check
            all_predicted_probs = [norm.cdf(z) for z in all_predicted_scores]
            for target_fpr in target_fpr_list:
                target_eval_results = get_tpr_and_f1_at_fpr(all_true_labels, all_predicted_probs, target_fpr)
                eval_results_dict[f'target_fpr@{target_fpr}'] = target_eval_results

            all_batch_eval_results = []
            for i in range(0, len(predicted_watermark_labels), batch_size):
                batch_predicted_watermark_labels = predicted_watermark_labels[i:i+batch_size]
                batch_predicted_wo_watermark_labels = predicted_wo_watermark_labels[i:i+batch_size]
                batch_true_watermark_labels = [1]*len(batch_predicted_watermark_labels)
                batch_true_wo_watermark_labels = [0]*len(batch_predicted_wo_watermark_labels)
                batch_true_labels = batch_true_watermark_labels + batch_true_wo_watermark_labels
                batch_predicted_labels = batch_predicted_watermark_labels + batch_predicted_wo_watermark_labels
                assert len(batch_predicted_watermark_labels) == len(batch_predicted_wo_watermark_labels) == batch_size
                all_batch_eval_results.append(get_classification_score(batch_true_labels, batch_predicted_labels))
            df_all_batch = pd.DataFrame(all_batch_eval_results)
            eval_results_dict['all_batch'] = df_all_batch.mean().to_dict()

            for target_fpr in target_fpr_list:
                all_batch_target_results = []
                for i in range(0, len(predicted_watermark_labels), batch_size):
                    batch_z_scores_watermark_scores = all_z_scores_watermark[i:i+batch_size]
                    batch_z_scores_wo_watermark_scores = all_z_scores_wo_watermark[i:i+batch_size]
                    batch_predicted_scores = batch_z_scores_watermark_scores + batch_z_scores_wo_watermark_scores
                    batch_true_watermark_labels = [1]*len(batch_z_scores_watermark_scores)
                    batch_true_wo_watermark_labels = [0]*len(batch_z_scores_wo_watermark_scores)
                    batch_true_labels = batch_true_watermark_labels + batch_true_wo_watermark_labels
                    batch_predicted_probs = [norm.cdf(z) for z in batch_predicted_scores]
                    assert len(batch_z_scores_watermark_scores) == len(batch_z_scores_wo_watermark_scores) == batch_size
                    all_batch_target_results.append(get_tpr_and_f1_at_fpr(batch_true_labels, batch_predicted_probs, target_fpr))
                df_target_batch = pd.DataFrame(all_batch_target_results)
                eval_results_dict[f'target_fpr_batch@{target_fpr}'] = df_target_batch.mean().to_dict()
            
            # record all eval results under a z_score
            all_eval_results_dict[f'z_score_{detection_z_score_threshold}'] = eval_results_dict

        print(all_eval_results_dict)
        with open(f'{folder_path}/eval_detect_results_canonical_{use_canonical}_pass_1_{use_pass_1}.json', 'w') as file:
            json.dump({'detection score': all_eval_results_dict}, file, indent = 4)
