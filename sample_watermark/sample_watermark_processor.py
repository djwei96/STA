from __future__ import annotations
from math import sqrt

import scipy.stats
from scipy.stats import entropy

import torch
from torch.nn import functional as F


class SampleWatermark:
    def __init__(
        self,
        vocab: list[int] = None,
        n_sample_per_token: int = 32, 
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key1: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        hash_key2: int = 17624813,
    ):

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.n_sample_per_token = n_sample_per_token
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key1 = hash_key1
        self.hash_key2 = hash_key2

    def _seed_rng(self, now_token_id: torch.LongTensor, next_token_id: torch.LongTensor) -> None:
        self.rng.manual_seed(self.hash_key1*now_token_id + self.hash_key2*next_token_id)
        return

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        cur_len,
        eos_token_id,
        batch_size = 1,
        num_beams = 1,
        min_length = 0,
        repetition_penalty = 1.0,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores, batch_size, num_beams, input_ids, repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        return scores
    
    def top_k_top_p_filtering(
        self,
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
        ):
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
        
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def generate_with_watermark(self, input_ids, args, model, device, tokenizer):
        self.rng = torch.Generator(device=device)
        cur_len = 0
        output_stats = []
        past_key_values = None
        output_ids = input_ids.clone()

        with torch.no_grad():
            now_token_id = input_ids[0][-1]
            for _ in range(args.max_new_tokens):
                output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values # save past values for fast decoding
    
                next_token_logits = output.logits[:, -1, :] # or detach()
                next_token_dist = torch.softmax(next_token_logits, dim=-1)
    
                next_token_score = self.postprocess_next_token_scores(
                    scores=next_token_logits,
                    input_ids=input_ids,
                    cur_len = cur_len,
                    eos_token_id=tokenizer.eos_token_id,
                    #min_length=args.max_new_tokens-200,
                    num_beams=1,
                )
    
                if args.sampling_temp != 1.0:
                    next_token_score = next_token_score / args.sampling_temp
                next_token_logscores = self.top_k_top_p_filtering(next_token_score, top_k=0, top_p=1.0)
                next_token_probs = F.softmax(next_token_logscores, dim=-1) #TODO: change next_token_probs to next_token_dist
    
                for j in range(self.n_sample_per_token+1):
                    if args.use_sampling:
                        next_token_id = torch.multinomial(next_token_probs, num_samples=1)
                        #next_token_id = torch.multinomial(next_token_dist, num_samples=1)
                    else:
                        next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)
                        #next_token_id = torch.argmax(next_token_dist, dim=-1).unsqueeze(-1)
                    
                    self._seed_rng(now_token_id.item(), next_token_id.item())
                    cipher = torch.rand(1, device=self.rng.device, generator=self.rng)
                    if cipher.item() < args.gamma:
                        break 
    
                if next_token_id.squeeze().item() == tokenizer.eos_token_id:
                    break
    
                input_ids = next_token_id # use past_key_values to speed up decoding
                now_token_id = next_token_id
                output_ids = torch.cat([output_ids, next_token_id], dim=-1)
                cur_len += 1

                # record results
                next_token_entropy = entropy(next_token_dist.cpu().detach().numpy().tolist()[0])
                sorted_indices = torch.argsort(next_token_dist, descending=True)
                next_token_rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[1].item() + 1
                next_token_prob = next_token_dist.squeeze()[next_token_id].item()
                output_stats.append({'next_token_id': next_token_id.item(), 
                                     'next_token_rank': next_token_rank, 
                                     'next_token_prob': next_token_prob, 
                                     'next_token_entropy': next_token_entropy, 
                                     'sample_time': j})
            return output_ids, output_stats
    
    def generate_without_watermark(self, input_ids, args, model, device, tokenizer):
        cur_len = 0
        output_stats = []
        past_key_values = None
        output_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values # cache for fast decoding
    
                next_token_logits = output.logits[:, -1, :]
                next_token_dist = torch.softmax(next_token_logits, dim=-1)
    
                next_token_score = self.postprocess_next_token_scores(
                    scores=next_token_logits,
                    input_ids=input_ids,
                    cur_len = cur_len,
                    eos_token_id=tokenizer.eos_token_id,
                    #min_length=args.max_new_tokens-200,
                    num_beams=1,
                )
    
                if args.sampling_temp != 1.0:
                    next_token_score = next_token_score / args.sampling_temp
                next_token_logscores = self.top_k_top_p_filtering(next_token_score, top_k=0, top_p=1.0)
                next_token_probs = F.softmax(next_token_logscores, dim=-1) #TODO: change next_token_probs to next_token_dist
    
                if args.use_sampling:
                    next_token_id = torch.multinomial(next_token_probs, num_samples=1)
                    #next_token_id = torch.multinomial(next_token_dist, num_samples=1)
                else:
                    next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)
                    #next_token_id = torch.argmax(next_token_dist, dim=-1).unsqueeze(-1)
    
                if next_token_id.squeeze().item() == tokenizer.eos_token_id:
                    break
    
                input_ids = next_token_id # use past_key_values to speed up decoding
                output_ids = torch.cat([output_ids, next_token_id], dim=-1)
                cur_len += 1

                # record results
                next_token_entropy = entropy(next_token_dist.cpu().detach().numpy().tolist()[0])
                sorted_indices = torch.argsort(next_token_dist, descending=True)
                next_token_rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[1].item() + 1
                next_token_prob = next_token_dist.squeeze()[next_token_id].item()
                output_stats.append({'next_token_id': next_token_id.item(), 
                                     'next_token_rank': next_token_rank, 
                                     'next_token_prob': next_token_prob, 
                                     'next_token_entropy': next_token_entropy})
            return output_ids, output_stats
    

class SampleWatermarkDetector():
    def __init__(
        self,
        vocab: list[int] = None,
        n_sample_per_token: int = 1, 
        z_threshold: int = 2, 
        gamma = 0.5,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key1: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        hash_key2: int = 17624813,
    ):
        
        device = 'cuda'
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.n_sample_per_token = n_sample_per_token
        self.hash_key1 = hash_key1
        self.hash_key2 = hash_key2

        self.z_threshold = z_threshold
        self.gamma = gamma
        self.seeding_scheme = seeding_scheme
        self.rng = torch.Generator(device=device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

    def _seed_rng(self, now_token_id: torch.LongTensor, next_token_id: torch.LongTensor) -> None:
        self.rng.manual_seed(self.hash_key1*now_token_id + self.hash_key2*next_token_id)
        return

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z_score = numer / denom
        return z_score

    def _compute_p_value(self, z_score):
        p_value = scipy.stats.norm.sf(z_score)
        return p_value

    def detect(self, input_text, prompt_max_length, device, tokenizer):
        prompt = input_text
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=prompt_max_length).to(device)
        
        result = []
        for i in range(input_ids.shape[1] - 1):
            self._seed_rng(input_ids[0][i].item(), input_ids[0][i+1].item())
            cipher = torch.rand(1, device=self.rng.device, generator=self.rng)

            if cipher.item() < self.gamma:
                result.append(1)
            else:
                result.append(0) 
        #print(result)
        z_score = self._compute_z_score(result.count(1), len(result))
        p_value = self._compute_p_value(z_score)
        return z_score, p_value
