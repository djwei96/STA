import os
import json
import numpy as np
from pathlib import Path

def group_passed_data(file_path):
    with open(file_path, 'r') as file:
        line_counter = 0
        current_group = []

        for line in file:
            data = json.loads(line)
            passed_value = data.get('passed')
            if not passed_value:
                current_group.append(0)
            else:
                current_group.append(1)
            line_counter += 1
            
    return current_group

if __name__ == '__main__':
    algo_type = 'sample' 
    sample_threshold = False

    if algo_type == 'sample' and sample_threshold:
        parent_folder_path = Path(f'code_results/human_eval/sample_threshold')
    else:
        parent_folder_path = Path(f'code_results/human_eval/{algo_type}')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)
    #folder_names = folder_names[-4:]

    # evaluate all folders of given algorithm
    for folder_name in folder_names:
        print('current folder: ', folder_name)
        folder_path = os.path.join(parent_folder_path, folder_name)
        file_path = os.path.join(folder_path, f'filtered_codes_{algo_type}.jsonl_results.jsonl')

        grouped_passed_data = group_passed_data(file_path)

        average_number_pass = 0
        count = 0
        for i in range(164):
            tmp = grouped_passed_data[i::164]
            if 1 in tmp:
                count += 1
                average_number_pass += np.sum(tmp)

        results = {'total': int(average_number_pass),
                   'count': count,
                   'average': average_number_pass / count
                   }
        with open(os.path.join(folder_path, f'eval_pass_number.json'), 'w') as file:
            json.dump(results, file, indent=4) 