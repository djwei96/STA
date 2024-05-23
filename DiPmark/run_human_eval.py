import os
import sys
import json
import torch
import argparse
import scipy

from tqdm import tqdm
from math import sqrt
from datetime import datetime
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from scipy.stats import entropy
from human_eval.data import write_jsonl, read_problems

from dipmark import (
    Dip_Reweight,
    WatermarkLogitsProcessor,
    PrevN_ContextCodeExtractor,
)

from dipmark import patch_model


parser = argparse.ArgumentParser()
parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="codellama/CodeLlama-7b-Instruct-hf",
    )
parser.add_argument(
        "--hash_key",
        type=int,
        default=15485863,
        help="The key for hashing.",
    )
parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )
parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
    )
parser.add_argument(
        "--prompt_template_name",
        type=str,
        default='instruct',
        help="Prompt template to wrap intput.",
    )
parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default='9',
    )
parser.add_argument(
        "--temperature",
        type=float,
        default=1,
    )
parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )
parser.add_argument(
        "--top_k",
        type=int,
        default=0,
    )
parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
    )
parser.add_argument(
        "--pass_k",
        type=int,
        default=5,
    )
parser.add_argument(
        "--prevN",
        type=int,
        default=5,
    )
parser.add_argument(
        "--generation_seed",
        type=int,
        default=42,
    )


def compute_z_score(token_quantiles, T, gamma):
    # count refers to number of green tokens, T is total number of tokens
    observed_count = sum(1 for x in token_quantiles if x > gamma)
    numer = observed_count - gamma * T
    denom = sqrt(T * gamma * (1 - gamma))
    z_score = numer / denom
    return z_score


def compute_p_value(z_score):
    p_value = scipy.stats.norm.sf(z_score)
    return p_value


if __name__ == '__main__':
    args = parser.parse_args()
    hash_key = str(args.hash_key)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    hf_cache_dir = 'hf_models'
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir=hf_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=hf_cache_dir)
    prompt_max_length = 4096 - args.max_new_tokens
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    patch_model(model)

    result_dir = f'code_results/human_eval/dipmark/{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'config.json'), 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)

    with open('prompt_template_code.json', 'r') as file:
        prompt_templates = json.load(file)
    prompt_template_name = args.prompt_template_name
    if prompt_template_name in prompt_templates:
        prompt_template = prompt_templates[prompt_template_name]
    else:
        print('wrong template name')
        sys.exit(0)

    torch.manual_seed(args.generation_seed)

    problems = read_problems()
    #problems = {k: problems[k] for k in list(problems)[:2]}
    full_completions_watermark = []
    full_completions_wo_watermark = []
    for per_pass in tqdm(range(args.pass_k)): 
        if per_pass == 0: 
            output_file = open(os.path.join(result_dir, 'dataset_output.jsonl'), "w") 

        for task_id in tqdm(problems):
            dipmark_wp = WatermarkLogitsProcessor(
                hash_key.encode(),
                Dip_Reweight(args.alpha),
                PrevN_ContextCodeExtractor(args.prevN),
            )
        
            per_problem = problems[task_id]['prompt']
            input_text = prompt_template.format(per_problem)
            inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, max_length=prompt_max_length, truncation=True)
            input_ids = inputs['input_ids'].to(device)

            # watermark 
            output_ids_watermark = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p = args.top_p,
                top_k = args.top_k,
                num_beams = 1,
                logits_warper = LogitsProcessorList([dipmark_wp]),
                pad_token_id = tokenizer.eos_token_id
            )

            output_ids_wo_watermark = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p = args.top_p,
                top_k = args.top_k,
                num_beams = 1,
                pad_token_id = tokenizer.eos_token_id
            )
        
            full_output_ids_watermark = output_ids_watermark.clone()
            output_ids_watermark = output_ids_watermark[:, input_ids.shape[-1]:]
            decoded_output_watermark = tokenizer.batch_decode(output_ids_watermark, skip_special_tokens=True)[0]
            full_completions_watermark.append({"task_id": task_id, "completion": decoded_output_watermark})

            token_quantiles_watermark = []
            input_ids_for_quantile = input_ids.clone()
            for i in range(output_ids_watermark.shape[1]):
                current_token = output_ids_watermark[:, i]
                current_token_quantile = dipmark_wp.get_green_token_quantile(input_ids_for_quantile, len(tokenizer.vocab), current_token)
                token_quantiles_watermark.append(current_token_quantile[0].item())
                input_ids_for_quantile =  torch.cat((input_ids_for_quantile, current_token.unsqueeze(0)), dim=-1)
            try:
                z_score_watermark = compute_z_score(token_quantiles_watermark, output_ids_watermark.shape[1], args.gamma)
                p_value_watermark = compute_p_value(z_score_watermark)
            except:
                z_score_watermark = None
                p_value_watermark = None

            full_output_ids_wo_watermark = output_ids_wo_watermark.clone()
            output_ids_wo_watermark = output_ids_wo_watermark[:, input_ids.shape[-1]:]
            decoded_output_wo_watermark = tokenizer.batch_decode(output_ids_wo_watermark, skip_special_tokens=True)[0]
            full_completions_wo_watermark.append({"task_id": task_id, "completion": decoded_output_wo_watermark})

            token_quantiles_wo_watermark = []
            input_ids_for_quantile = input_ids.clone()
            for i in range(output_ids_wo_watermark.shape[1]):
                current_token = output_ids_wo_watermark[:, i]
                current_token_quantile = dipmark_wp.get_green_token_quantile(input_ids_for_quantile, len(tokenizer.vocab), current_token)
                token_quantiles_wo_watermark.append(current_token_quantile[0].item())
                input_ids_for_quantile =  torch.cat((input_ids_for_quantile, current_token.unsqueeze(0)), dim=-1)
            try:
                z_score_wo_watermark = compute_z_score(token_quantiles_wo_watermark, output_ids_wo_watermark.shape[1], args.gamma)
                p_value_wo_watermark = compute_p_value(z_score_wo_watermark)
            except:
                z_score_wo_watermark = None
                p_value_wo_watermark = None

            if per_pass == 0:
                output_original_scores = dipmark_wp.output_original_scores
                output_modified_scores = dipmark_wp.output_modified_scores
                assert len(output_original_scores) == len(output_modified_scores) == output_ids_watermark.shape[-1]
                #print(torch.equal(output_original_scores[0], output_modified_scores[0]))
        
                output_stats = []
                for i in range(len(output_original_scores)):
                    next_token_id = output_ids_watermark[0][i].unsqueeze(-1).unsqueeze(-1)
                    next_token_logits = output_original_scores[i]
                    next_token_dist = torch.softmax(next_token_logits, dim=-1)
                    next_token_entropy = entropy(next_token_dist.cpu().detach().numpy().tolist()[0])
                    sorted_indices = torch.argsort(next_token_dist, descending=True)
                    next_token_rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[1].item() + 1
                    next_token_prob = next_token_dist.squeeze()[next_token_id].item()
                    output_stats.append({'next_token_id': next_token_id.item(), 'next_token_rank': next_token_rank, 
                                         'next_token_prob': next_token_prob, 'next_token_entropy': next_token_entropy})
            
                term_width = 80
                print("-"*term_width)
                print('Output with dipmark ')
                print(decoded_output_watermark)
                print('z score: ', z_score_watermark, 'p value: ', p_value_watermark)
                print("-"*term_width)
                print('Output without dipmark')
                print(decoded_output_wo_watermark)
                print('z score: ', z_score_wo_watermark, 'p value: ', p_value_wo_watermark)
        
                per_gen_results = {'prompt': input_text, 
                                   'output_dipmark': decoded_output_watermark, 'stats_dipmark': output_stats,
                                   'full_output_ids_dipmark': full_output_ids_watermark.detach().tolist(),
                                   'z_score_dipmark': z_score_watermark, 'p_value_dipmark': p_value_watermark,
                                   'output_wo_dipmark': decoded_output_wo_watermark, 
                                   'full_output_ids_wo_dipmark': full_output_ids_wo_watermark.detach().tolist(),
                                   'z_score_wo_dipmark': z_score_wo_watermark, 'p_value_wo_dipmark': p_value_wo_watermark,
                                   } 

                output_file.write(json.dumps(per_gen_results) + '\n') 

        if per_pass == 0: 
            output_file.close()

    write_jsonl(os.path.join(result_dir, "full_completions_dipmark.jsonl"), full_completions_watermark)
    write_jsonl(os.path.join(result_dir, "full_completions_wo_dipmark.jsonl"), full_completions_wo_watermark)
