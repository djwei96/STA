import os
import numpy as np
import pandas as pd
import statistics


def get_mrr_score(df, algo_type, low_entropy_threshold):
    def calculate_mrr(rank_list):
        reciprocal_ranks = [1.0 / rank for rank in rank_list if rank > 0]
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        return mrr

    stats = df[f'stats_{algo_type}']
    all_mrr = [] 
    for stat in stats:
        per_gen_token_entropies = [data['next_token_entropy'] for data in stat]
        per_gen_token_ranks = [data['next_token_rank'] for data in stat]
        #print(len(per_gen_token_ranks))
        per_gen_low_entropy_mask = np.array(per_gen_token_entropies) < low_entropy_threshold
        per_gen_low_entropy_filtered_rank = np.array(per_gen_token_ranks)[per_gen_low_entropy_mask]
        #print(len(per_gen_low_entropy_filtered_rank))
        if len(per_gen_low_entropy_filtered_rank) > 0:
            per_mrr = calculate_mrr(per_gen_low_entropy_filtered_rank)
            all_mrr.append(per_mrr)
    avg_mrr = statistics.mean(all_mrr)
    return avg_mrr
