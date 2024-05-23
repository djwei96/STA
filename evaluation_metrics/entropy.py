import os
import pandas as pd


def get_entropy_percentile(df, algo_type, n_percentile):
    stats = df[f'stats_{algo_type}']
    all_token_entropies = [] 
    for stat in stats:
        per_gen_token_entropies = [data['next_token_entropy'] for data in stat]
        all_token_entropies.extend(per_gen_token_entropies)
    all_token_entropies = sorted(all_token_entropies)
    entropy_percentile = all_token_entropies[int(len(all_token_entropies)/n_percentile)]
    return entropy_percentile
