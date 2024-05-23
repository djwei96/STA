import os
import json
import torch
import argparse

from tqdm import tqdm
from datetime import datetime
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from dipmark import (
    Dip_Reweight,
    WatermarkLogitsProcessor,
    PrevN_ContextCodeExtractor,
)

from scipy.stats import entropy

from dipmark import patch_model

parser = argparse.ArgumentParser()
parser.add_argument(
        "--model_name_or_path",
        type=str,
        #default='facebook/opt-1.3b',
        default='meta-llama/Llama-2-7b-hf',
    )
parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )
parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default='3',
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
        default=200,
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


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    hf_cache_dir = 'hf_models'
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir=hf_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=hf_cache_dir)
    prompt_max_length = 2048 - args.max_new_tokens
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    patch_model(model)
    
    with open('data/C4/sampled_dataset_train.json', "r") as input_file:
        data = json.load(input_file)
    
    result_dir = f'results/C4/dipmark/{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'config.json'), 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    
    torch.manual_seed(args.generation_seed)
    output_file = open(os.path.join(result_dir, 'dataset_output.jsonl'), "w")
    for input_text in tqdm(data):
        dipmark_wp = WatermarkLogitsProcessor(
            b"15485863",
            Dip_Reweight(args.alpha),
            PrevN_ContextCodeExtractor(args.prevN),
        )
    
        input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, 
                            truncation=True, max_length=prompt_max_length)
        input_ids = input_ids['input_ids'].to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p = args.top_p,
            top_k = args.top_k,
            num_beams = 1,
            logits_warper=LogitsProcessorList([dipmark_wp]),
        )
    
        #print(torch.equal(input_ids, output_ids[:, :input_ids.shape[-1]]))
        full_output_ids = output_ids.clone()
        output_ids = output_ids[:, input_ids.shape[-1]:]
        output_original_scores = dipmark_wp.output_original_scores
        output_modified_scores = dipmark_wp.output_modified_scores
        #print(output_ids.shape)
    
        assert len(output_original_scores) == len(output_modified_scores) == output_ids.shape[-1]
        #print(torch.equal(output_original_scores[0], output_modified_scores[0]))
    
        output_stats = []
        for i in range(len(output_original_scores)):
            next_token_id = output_ids[0][i].unsqueeze(-1).unsqueeze(-1)
            next_token_logits = output_original_scores[i]
            next_token_dist = torch.softmax(next_token_logits, dim=-1)
            next_token_entropy = entropy(next_token_dist.cpu().detach().numpy().tolist()[0])
            sorted_indices = torch.argsort(next_token_dist, descending=True)
            next_token_rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[1].item() + 1
            next_token_prob = next_token_dist.squeeze()[next_token_id].item()
            output_stats.append({'next_token_id': next_token_id.item(), 'next_token_rank': next_token_rank, 
                                                'next_token_prob': next_token_prob, 'next_token_entropy': next_token_entropy})
    
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(output_text)
    
        per_gen_results = {'prompt': input_text, 
                       'output_dipmark': output_text[0], 
                       'stats_dipmark': output_stats,
                       'full_output_ids_dipmark': full_output_ids.detach().tolist()
                    } 
        output_file.write(json.dumps(per_gen_results) + '\n') 
    output_file.close()