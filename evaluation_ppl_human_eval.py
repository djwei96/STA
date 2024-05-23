import os
import sys
import json
import torch
import argparse
import statistics
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from human_eval.data import read_problems


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
        default='0',
    )
    parser.add_argument(
        "--pass_k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--algo_type",
        type=str,
        default='oracle', # sample, green, dipmark, delta, oracle
    )
    parser.add_argument(
        "--sample_threshold",
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
    cache_dir = 'hf_models'
    algo_type = args.algo_type
    sample_threshold = args.sample_threshold
    device = "cuda" if torch.cuda.is_available() else "cpu"

    problems = read_problems()
    prompts = [problems[key]['prompt'] for key in problems.keys()]
    batch_size = len(prompts)
    
    if sample_threshold == True and algo_type == 'sample':
        parent_folder_path = Path(f'code_results/human_eval/sample_threshold')
    else:
        parent_folder_path = Path(f'code_results/human_eval/{algo_type}')
    folder_names = [dir.name for dir in parent_folder_path.iterdir() if dir.is_dir()]
    folder_names = sorted(folder_names)
    #folder_names = folder_names[-5:]

    for folder_name in tqdm(folder_names):
        print(folder_name)
        folder_path = os.path.join(parent_folder_path, folder_name)
        with open(os.path.join(folder_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        df_watermark = pd.read_json(os.path.join(folder_path, f'full_completions_{algo_type}.jsonl'), lines=True)
        #df_watermark = df_watermark.head(batch_size*2)
        completions_watermark = df_watermark['completion'].to_list()
        total_rows = len(df_watermark)
        assert total_rows % batch_size == 0
        
        # claculate ppl 
        model_name, avg_ppl = 'no', None
        gen_model_name = config_dict['model_name_or_path']
        if gen_model_name == 'codellama/CodeLlama-7b-Instruct-hf':
            model_name = 'codellama/CodeLlama-13b-Instruct-hf'
        else:
            continue
        
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left', cache_dir=cache_dir)
        model.eval()
        model.to(device)

        # assert 
        all_ppl = []
        for i in tqdm(range(0, len(completions_watermark))):
            prompt = prompts[i%batch_size]
            input_text = prompt + completions_watermark[i]
            inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, max_length=args.max_length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                ppl_outputs = model(**inputs, labels=inputs['input_ids'])
                loss = ppl_outputs.loss
                ppl = torch.exp(loss)
                ppl = ppl.item()
            all_ppl.append(ppl)

        assert len(all_ppl) // batch_size == args.pass_k
        all_avg_ppl = statistics.mean(all_ppl)

        all_batch_mean_ppl = []
        for i in range(0, len(all_ppl), batch_size):
            batch_ppl = all_ppl[i:i+batch_size]
            print(len(batch_ppl))
            all_batch_mean_ppl.append(statistics.mean(batch_ppl))

        term_width = 80
        print('-'*term_width)
        results = {'pass_k': args.pass_k,
                   'all average ppl': all_avg_ppl,
                   'all ppl values': all_ppl,
                   'all batch mean ppl values': all_batch_mean_ppl
                   }
        with open(os.path.join(folder_path, f'eval_ppl_pass_k_{args.pass_k}.json'), 'w') as file:
            json.dump(results, file, indent=4)
