import os
import sys
import json
import numpy
import argparse
import pandas as pd

from argparse import Namespace
from datetime import datetime
from pprint import pprint
from functools import partial
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import entropy

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from accelerate import dispatch_model, infer_auto_device_map
from peft import get_peft_config, get_peft_model 

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
        "--hash_key",
        type=int,
        default=15485863,
        help="The key for hashing.",
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
        default=5,
        help="Number of passes.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
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
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
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
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=2.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
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
    """Load and return the model and tokenizer"""
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


def generate(prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens,
                                                    hash_key=args.hash_key)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k = 0,
            top_p = 1.0,
            temperature=args.sampling_temp,
            num_beams=args.n_beams,
            pad_token_id=tokenizer.eos_token_id
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams,
            pad_token_id=tokenizer.eos_token_id
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )

    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 4096-args.max_new_tokens
    
    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, 
                           truncation=True, max_length=args.prompt_max_length)
    tokd_input.to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    full_output_ids_without_watermark = generate_without_watermark(**tokd_input)
    full_output_ids_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = full_output_ids_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = full_output_ids_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    output_original_scores, output_modified_scores = watermark_processor.get_output_scores()
    #print(torch.all(output_original_scores[0] == output_modified_scores[0]))

    output_stats_with_watermark = []
    assert len(output_original_scores) == len(output_modified_scores) == output_with_watermark.shape[-1]
    for i in range(len(output_original_scores)):
        next_token_id = output_with_watermark[0][i].unsqueeze(-1).unsqueeze(-1)
        next_token_logits = output_original_scores[i]
        next_token_dist = torch.softmax(next_token_logits, dim=-1)
        next_token_entropy = entropy(next_token_dist.cpu().detach().numpy().tolist()[0])
        sorted_indices = torch.argsort(next_token_dist, descending=True)
        next_token_rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[1].item() + 1
        next_token_prob = next_token_dist.squeeze()[next_token_id].item()
        output_stats_with_watermark.append({'next_token_id': next_token_id.item(), 'next_token_rank': next_token_rank, 
                                            'next_token_prob': next_token_prob, 'next_token_entropy': next_token_entropy})

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            full_output_ids_without_watermark.detach().tolist(),
            full_output_ids_with_watermark.detach().tolist(),
            output_stats_with_watermark,
            args) 


def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
    else:
        score_dict = None
        print('Error')
    return score_dict['z_score'], score_dict['p_value'] 

    
def main(args): 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    hf_cache_dir = 'hf_models'
    model, tokenizer, device = load_model_cache(args, hf_cache_dir)
    #model, tokenizer, device = load_model(args)

    result_dir = f'code_results/human_eval/green/{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
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
    full_completions_green = []
    full_completions_wo_green = []
    for per_pass in tqdm(range(args.pass_k)): 
        if per_pass == 0: 
            output_file = open(os.path.join(result_dir, 'dataset_output.jsonl'), "w") 

        for task_id in tqdm(problems):
            per_problem = problems[task_id]['prompt']
            input_text = prompt_template.format(per_problem)
    
            # Benchmark of green token list
            _, _, output_wo_green, output_green, full_output_ids_wo_green, full_output_ids_green, output_stats_green, _ = generate(input_text, args, model=model, device=device, tokenizer=tokenizer)
            full_completions_green.append({"task_id": task_id, "completion": output_green})
            full_completions_wo_green.append({"task_id": task_id, "completion": output_wo_green})

            if per_pass == 0: 
                try:
                    z_score_green, p_value_green = detect(output_green, args, device=device, tokenizer=tokenizer)
                except:
                    z_score_green, p_value_green = None, None
                try:
                    z_score_wo_green, p_value_wo_green = detect(output_wo_green, args, device=device, tokenizer=tokenizer)
                except:
                    z_score_wo_green, p_value_wo_green = None, None
        
                term_width = 80
                print("-"*term_width)
                print("Output with green token list:")
                print(output_green)
                print('z score: ', z_score_green, 'p value: ', p_value_green)
                print("-"*term_width)
                print("Output without green token list:")
                print(output_wo_green)
                print('z score: ', z_score_wo_green, 'p value: ', p_value_wo_green)
        
                # Record results
                per_results = {'prompt': input_text, 
                            'output_green': output_green, 'stats_green': output_stats_green, 
                            'full_output_ids_green': full_output_ids_green,
                            'z_score_green': z_score_green, 'p_value_green': p_value_green, 
                            'output_wo_green': output_wo_green, 
                            'full_output_ids_wo_green': full_output_ids_wo_green,
                            'z_score_wo_green': z_score_wo_green, 'p_value_wo_green': p_value_wo_green
                            } 
                output_file.write(json.dumps(per_results) + '\n') 
                #torch.cuda.empty_cache()
        
        if per_pass == 0: 
            output_file.close()
    
    write_jsonl(os.path.join(result_dir, "full_completions_green.jsonl"), full_completions_green)
    write_jsonl(os.path.join(result_dir, "full_completions_wo_green.jsonl"), full_completions_wo_green)
    return None

if __name__ == "__main__":
    args = parse_args()
    main(args)
