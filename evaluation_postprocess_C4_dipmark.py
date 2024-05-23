import os
import json
import pandas as pd
from pathlib import Path


# collect entropy threshold results for human eval
if __name__ == '__main__':
    use_detection = False
    ppl_use_token_ids = True
    filter_long_text_num = 500
    parent_folder_path = Path(f'results/C4/dipmark')

    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    print(folder_names)
    folder_names = sorted(folder_names)
    #folder_names = folder_names[:-2]

    all_results = []
    for folder_name in folder_names:
        result_dict = {}
        folder_path = os.path.join(parent_folder_path, folder_name)
        try:
            with open(os.path.join(folder_path, f'eval_results_dipmark_n_{filter_long_text_num}_ppl_Llama-2-13b-hf_id_{ppl_use_token_ids}.json'), 'r') as file:
                result = json.load(file)
            with open(os.path.join(folder_path, f'config.json'), 'r') as file:
                config_dict = json.load(file)
        except:
            continue

        print('current folder: ', folder_name)
        result_dict['algo_type'] = 'dipmark'
        result_dict['folder_name'] = folder_name
        result_dict['alpha'] = config_dict['alpha']
        result_dict['prevN'] = config_dict['prevN']
        result_dict['ppl'] = result['average ppl']
        result_dict['diversity'] = result['average diversity']
        result_dict['log diversity'] = result['average log diversity']
        result_dict['coherence'] = result['average coherence score']
        result_dict['mauve'] = result['average mauve score']

        if use_detection:
            #detection_result = pd.read_json(os.path.join(folder_path, f'eval_detect_results_canonical_{use_canonical}_pass_1_{use_pass_1}.json'), lines=True)
            break
        else:
            detection_result = result

        detection_z_thresholds = [1.9, 2.0, 4.0]
        for detection_z_threshold in detection_z_thresholds:
            if use_detection:
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
        df_results.to_excel(os.path.join(folder_path, f'eval_results_n_{filter_long_text_num}_id_{ppl_use_token_ids}_extra_detection_{use_detection}.xlsx'), index=False)
        all_results.append(result_dict)
    
    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_excel(os.path.join(parent_folder_path, f'pool/all_eval_results_n_{filter_long_text_num}_id_{ppl_use_token_ids}_extra_detection_{use_detection}.xlsx'), index=False)