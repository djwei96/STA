import os
import re
import pandas as pd
import json
import statistics
from human_eval.evaluation import evaluate_functional_correctness


def fix_indents(completion: str) -> str:
    return completion.replace("\t", "    ")


def filter_code(completion):
    completion = completion.strip()
    filtered_code = fix_indents(completion)

    pattern = r"```[ ]*python[ ]*\n(.*?)```"
    match = re.search(pattern, filtered_code, re.DOTALL)
    if match:
        filtered_code = match.group(1).strip()

    pattern = r"```\n(.*?)```"
    match = re.search(pattern, filtered_code, re.DOTALL)
    if match:
        filtered_code = match.group(1).strip()

    return filtered_code


def get_pass_score(folder_path, algo_type):
    df = pd.read_json(os.path.join(folder_path, f'full_completions_{algo_type}.jsonl'), lines=True)
    df['completion'] = df['completion'].apply(filter_code)
    df.to_json(os.path.join(folder_path, f'filtered_codes_{algo_type}.jsonl'), lines=True, orient='records')
    pass_k = evaluate_functional_correctness(os.path.join(folder_path, f'filtered_codes_{algo_type}.jsonl'), k=[1, 5, 10])

    pass_counts = []
    line_count = 0
    with open(os.path.join(folder_path, f'filtered_codes_{algo_type}.jsonl_results.jsonl'), 'r') as f:
        for _ in range(10):
            pass_count = 0
            for _ in range(164):
                line_count += 1
                line = f.readline()
                if not line:  # End of file
                    break
                data = json.loads(line)
                if data['passed']:
                    pass_count += 1
            pass_counts.append(pass_count)
    pass_variance = statistics.variance(pass_counts)
    print(line_count)
    return pass_k, pass_variance

