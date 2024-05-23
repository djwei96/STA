import os
import json
import pandas as pd
from pathlib import Path


# collect entropy threshold results for human eval
if __name__ == '__main__':
    use_detection = False
    use_batch = False
    use_pass_1 = False
    use_canonical = True
    parent_folder_path = Path(f'code_results/human_eval/sample_threshold')

    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    print(folder_names)
    folder_names = sorted(folder_names)
    #folder_names = folder_names[:-2]

    all_results = []
    for folder_name in folder_names:
        result_dict = {}
        folder_path = os.path.join(parent_folder_path, folder_name)
        try:
            with open(os.path.join(folder_path, f'eval_results_True.json'), 'r') as file:
                result = json.load(file)
            with open(os.path.join(folder_path, f'config.json'), 'r') as file:
                config_dict = json.load(file)
        except:
            continue

        print('current folder: ', folder_name)
        result_dict['algo_type'] = 'sample_threshold'
        result_dict['use_entropy_threshold'] = config_dict['use_entropy_threshold']
        result_dict['entropy_threshold'] = config_dict['entropy_threshold']
        result_dict['n_sample_per_token_above_entropy_threshold'] = config_dict['n_sample_per_token_above_entropy_threshold']
        result_dict['pass@1'] = result['pass score']['pass@1']
        result_dict['pass@5'] = result['pass score']['pass@5']
        result_dict['pass@10'] = result['pass score']['pass@10']

        if use_detection:
            detection_result = pd.read_json(os.path.join(folder_path, f'eval_detect_results_canonical_{use_canonical}_pass_1_{use_pass_1}.json'), lines=True)
        else:
            detection_result = result

        detection_z_thresholds = [1.9, 2.0, 2.5]
        for detection_z_threshold in detection_z_thresholds:
            if use_detection and use_batch:
                result_dict[f'z_{detection_z_threshold}_all_f1'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all_batch']['f1']
                result_dict[f'z_{detection_z_threshold}_all_auc'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all_batch']['auc']
                result_dict[f'z_{detection_z_threshold}_all_tpr'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all_batch']['tpr']
                result_dict[f'z_{detection_z_threshold}_all_tnr'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all_batch']['tnr']
                result_dict[f'z_{detection_z_threshold}_all_fpr'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all_batch']['fpr']
                result_dict[f'z_{detection_z_threshold}_all_fnr'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all_batch']['fnr']
            else:
                result_dict[f'z_{detection_z_threshold}_all_f1'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all']['f1']
                result_dict[f'z_{detection_z_threshold}_all_auc'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all']['auc']
                result_dict[f'z_{detection_z_threshold}_all_tpr'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all']['tpr']
                result_dict[f'z_{detection_z_threshold}_all_tnr'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all']['tnr']
                result_dict[f'z_{detection_z_threshold}_all_fpr'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all']['fpr']
                result_dict[f'z_{detection_z_threshold}_all_fnr'] = detection_result['detection score'][f'z_score_{detection_z_threshold}']['all']['fnr']

        df_results = pd.DataFrame([result_dict])
        df_results.to_excel(os.path.join(folder_path, f'eval_results_sample_threshold.xlsx'), index=False)
        all_results.append(result_dict)
    
    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_excel(os.path.join(parent_folder_path, f'pool/all_eval_results_sample_threshold_human_eval.xlsx'), index=False)