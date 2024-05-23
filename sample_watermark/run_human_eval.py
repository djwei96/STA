import os
import sys
import json
import numpy
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM)

from sample_watermark_processor import SampleWatermark, SampleWatermarkDetector

from human_eval.data import write_jsonl, read_problems


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


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="codellama/CodeLlama-7b-Instruct-hf",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--hash_key1",
        type=int,
        default=15485863,
        help="The first key for hashing.",
    )
    parser.add_argument(
        "--hash_key2",
        type=int,
        default=17624813,
        help="The second key for hashing.",
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        default='instruct',
        help="Prompt template to wrap intput.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--pass_k",
        type=int,
        default=1,
        help="Number of passes.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--n_sample_per_token",
        type=int,
        default=1,
        help="Sample times.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=42,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=1.0,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=2.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default='7',
    )
    args = parser.parse_args()
    return args


def load_model(args):
    """Load and return the model and tokenizer"""
    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom","llama"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer, device


def load_model_cache(args, hf_cache_dir):
    """Load and return the model and tokenizer from cache"""
    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom","llama"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, cache_dir=hf_cache_dir)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto', cache_dir=hf_cache_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', cache_dir=hf_cache_dir)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=hf_cache_dir)
    return model, tokenizer, device

    
def main(args): 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    hf_cache_dir = 'hf_models'
    model, tokenizer, device = load_model_cache(args, hf_cache_dir)
    #model, tokenizer, device = load_model(args)

    sample_watermark = SampleWatermark(vocab=list(tokenizer.get_vocab().values()),
                                       n_sample_per_token=args.n_sample_per_token,
                                       hash_key1=args.hash_key1,
                                       hash_key2=args.hash_key2)
    sample_detector = SampleWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                              n_sample_per_token=args.n_sample_per_token,
                                              z_threshold=args.detection_z_threshold,
                                              gamma=args.gamma,
                                              hash_key1=args.hash_key1,
                                              hash_key2=args.hash_key2)

    result_dir = f'code_results/human_eval/sample/{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    os.makedirs(result_dir)

    args.prompt_max_length = 4096 - args.max_new_tokens
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
    full_completions_sample = []
    full_completions_wo_sample = []
    for per_pass in range(args.pass_k): 
        if per_pass == 0: 
            output_file = open(os.path.join(result_dir, 'dataset_output.jsonl'), "w") 

        for task_id in tqdm(problems):
            per_problem = problems[task_id]['prompt']
            input_text = prompt_template.format(per_problem)
            inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, max_length=args.prompt_max_length, truncation=True)
            input_ids = inputs['input_ids'].to(device)
    
            output_ids_sample, output_stats_sample = sample_watermark.generate_with_watermark(input_ids, args, model, device, tokenizer)
            full_output_ids_sample = output_ids_sample.clone().detach().tolist()
            output_ids_sample = output_ids_sample[:,input_ids.shape[-1]:]
            decoded_output_sample = tokenizer.batch_decode(output_ids_sample, skip_special_tokens=True)[0]
            full_completions_sample.append({"task_id": task_id, "completion": decoded_output_sample})
    
            output_ids_wo_sample, output_stats_wo_sample = sample_watermark.generate_without_watermark(input_ids, args, model, device, tokenizer)
            full_output_ids_wo_sample = output_ids_wo_sample.clone().detach().tolist()
            output_ids_wo_sample = output_ids_wo_sample[:,input_ids.shape[-1]:]
            decoded_output_wo_sample = tokenizer.batch_decode(output_ids_wo_sample, skip_special_tokens=True)[0]
            full_completions_wo_sample.append({"task_id": task_id, "completion": decoded_output_wo_sample})

            if per_pass == 0: 
                try:
                    z_score_sample, p_value_sample = sample_detector.detect(decoded_output_sample, args.prompt_max_length, device, tokenizer)
                except:
                    z_score_sample, p_value_sample = None, None
                try:
                    z_score_wo_sample, p_value_wo_sample = sample_detector.detect(decoded_output_wo_sample, args.prompt_max_length, device, tokenizer)
                except:
                    z_score_wo_sample, p_value_wo_sample = None, None
    
                term_width = 80
                print("-"*term_width)
                print('Output with sampling ')
                print(decoded_output_sample)
                print('z score: ', z_score_sample, 'p value: ', p_value_sample)
                print("-"*term_width)
                print('Output without sampling')
                print(decoded_output_wo_sample)
                print('z score: ', z_score_wo_sample, 'p value: ', p_value_wo_sample)
    
                # Record results
                per_results = {'prompt': input_text, 
                        'output_sample': decoded_output_sample, 'stats_sample': output_stats_sample, 
                        'full_output_ids_sample': full_output_ids_sample,
                        'z_score_sample': z_score_sample, 'p_value_sample': p_value_sample,
                        'output_wo_sample': decoded_output_wo_sample, 'stats_wo_sample': output_stats_wo_sample, 
                        'z_score_wo_sample': z_score_wo_sample, 'p_value_wo_sample': p_value_wo_sample,
                        'full_output_ids_wo_sample': full_output_ids_wo_sample,
                        } 
                #output_file.write(json.dumps(per_results, indent=4) + '\n') 
                output_file.write(json.dumps(per_results) + '\n') 

        if per_pass == 0: 
            output_file.close()

    write_jsonl(os.path.join(result_dir, "full_completions_sample.jsonl"), full_completions_sample)
    write_jsonl(os.path.join(result_dir, "full_completions_wo_sample.jsonl"), full_completions_wo_sample)
    return None

if __name__ == "__main__":
    args = parse_args()
    main(args)
