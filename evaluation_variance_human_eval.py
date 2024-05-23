import os
import statistics
import json
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    algo_type = 'sample'
    sample_threshold = False
    
    if sample_threshold == True and algo_type == 'sample':
        parent_folder_path = Path(f'code_results/human_eval/sample_threshold')
    else:
        parent_folder_path = Path(f'code_results/human_eval/{algo_type}')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)
    folder_names = folder_names[:-1]   

    for folder_name in tqdm(folder_names):
        folder_path = os.path.join(parent_folder_path, folder_name)   
        with open(os.path.join(folder_path, 'eval_ppl_pass_k_10.json')) as f:
            ppl_resutls = json.load(f)
        all_ppl = ppl_resutls['all ppl values']

        all_variances = []
        outlier_count = 0
        for i in range(164):    
            variance = statistics.variance(all_ppl[i::164])
            if variance > 100:
                #print(all_variances)
                outlier_count += 1
                continue
            else:
                all_variances.append(variance)
        avg_variance = statistics.mean(all_variances)
        print(avg_variance)
        print(outlier_count)
        ppl_resutls['average variance'] = avg_variance

        with open(os.path.join(folder_path, 'eval_ppl_pass_k_10.json'), 'w') as f:
            json.dump(ppl_resutls, f, indent=4)