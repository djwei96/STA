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

from evaluation_metrics.detection import get_tpr_and_f1_at_fpr 

sys.path.append('lm_watermarking')
from lm_watermarking.watermark_processor import WatermarkDetector

sys.path.append('sample_watermark')
from sample_watermark.sample_watermark_processor import SampleWatermarkDetector

# must run evaluation_pipeline_C4.py first
if __name__ == '__main__':
    algo_type = 'sample'
    sample_threshold = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_reference = True
    filter_long_text = True
    filter_long_text_num = 500

    if sample_threshold == True and algo_type == 'sample':
        parent_folder_path = Path(f'results/C4/sample_threshold')
    else:
        parent_folder_path = Path(f'results/C4/{algo_type}')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)

    for folder_name in tqdm(folder_names):
        folder_path = os.path.join(parent_folder_path, folder_name)

        if filter_long_text:
            df = pd.read_json(os.path.join(folder_path, f'dataset_output_n_{filter_long_text_num}.jsonl'), lines=True)
        else:
            df = pd.read_json(os.path.join(folder_path, 'dataset_output.jsonl'), lines=True)
            filter_long_text_num = 1000

        output_text_watermark = df[f'output_{algo_type}'].to_list()
        if use_reference:
            output_text_wo_watermark = df[f'reference_{algo_type}'].to_list()
        else:
            output_text_wo_watermark = df[f'output_wo_{algo_type}'].to_list()

        if filter_long_text:
            assert len(output_text_watermark) == len(output_text_wo_watermark) == filter_long_text_num
        else:
            assert len(output_text_watermark) == len(output_text_wo_watermark) == 1000

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
        for i in range(0, len(df)):
            if algo_type == 'sample':
                try:
                    z_score_watermark, p_value_watermark = watermark_detector.detect(output_text_watermark[i].strip(), 2048, device, tokenizer)
                except:
                    z_score_watermark = 0
                    p_value_watermark = 0
                all_z_scores_watermark.append(z_score_watermark)
                all_p_values_watermark.append(p_value_watermark)
    
                try:
                    z_score_wo_watermark, p_value_wo_watermark = watermark_detector.detect(output_text_wo_watermark[i].strip(), 2048, device, tokenizer)
                except:
                    z_score_wo_watermark = 0
                    p_value_wo_watermark = 0
                all_z_scores_wo_watermark.append(z_score_wo_watermark)
                all_p_values_wo_watermark.append(p_value_wo_watermark)
            elif algo_type == 'green': 
                try:
                    z_score_watermark, p_value_watermark = watermark_detector.detect(output_text_watermark[i].strip())
                except:
                    z_score_watermark = 0
                    p_value_watermark = 0
                all_z_scores_watermark.append(z_score_watermark)
                all_p_values_watermark.append(p_value_watermark)
    
                try:
                    z_score_wo_watermark, p_value_wo_watermark = watermark_detector.detect(output_text_wo_watermark[i].strip())
                except:
                    z_score_wo_watermark = 0
                    p_value_wo_watermark = 0
                all_z_scores_wo_watermark.append(z_score_wo_watermark)
                all_p_values_wo_watermark.append(p_value_wo_watermark)
            else:
                sys.exit(0)
            
        # calculate overall score
        target_fpr_list = [0.01, 0.05, 0.1]
        detection_z_score_thresholds = [1.5, 1.9, 2]
        all_eval_results_dict = {}
        for detection_z_score_threshold in detection_z_score_thresholds:
            eval_results_dict = {}
            all_predicted_scores = all_z_scores_watermark +  all_z_scores_wo_watermark

            # label 1: watermark / label 0: wo watermark
            predicted_watermark_labels = [1 if value > detection_z_score_threshold else 0 for value in all_z_scores_watermark]
            predicted_wo_watermark_labels = [1 if value > detection_z_score_threshold else 0 for value in all_z_scores_wo_watermark]
            all_predicted_labels = predicted_watermark_labels + predicted_wo_watermark_labels
    
            ture_watermark_labels = [1]*len(predicted_watermark_labels)
            ture_wo_watermark_labels = [0]*len(predicted_wo_watermark_labels)
            all_true_labels = ture_watermark_labels + ture_wo_watermark_labels

            f1 = f1_score(all_true_labels, all_predicted_labels)
            auc = roc_auc_score(all_true_labels, all_predicted_labels)
            tn, fp, fn, tp = confusion_matrix(all_true_labels, all_predicted_labels).ravel()
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 'N/A'
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 'N/A'
            fpr = fp / (fp + tn) if (tp + fn) != 0 else 'N/A'
            fnr = fn / (fn + tp) if (tp + fn) != 0 else 'N/A'
            eval_results_dict['all'] = {'f1': f1, 'auc': auc, 'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr}

            #TODO: under devlopment, need to check
            all_predicted_probs = [norm.cdf(z) for z in all_predicted_scores]
            for target_fpr in target_fpr_list:
                target_eval_results = get_tpr_and_f1_at_fpr(all_true_labels, all_predicted_probs, target_fpr)
                eval_results_dict[f'target_fpr@{target_fpr}'] = target_eval_results
            all_eval_results_dict[f'z_score_{detection_z_score_threshold}'] = eval_results_dict

        print(all_eval_results_dict)
        with open(f'{folder_path}/eval_detect_results_reference_{use_reference}_n_{filter_long_text_num}.json', 'w') as file:
            json.dump({'detection score': all_eval_results_dict}, file, indent = 4)
