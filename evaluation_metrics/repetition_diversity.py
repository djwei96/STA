'''
Adapted from https://github.com/yxuansu/SimCTG/blob/main/simctg/evaluation.py
as used in Contrastive Decoding https://github.com/XiangLi1999/ContrastiveDecoding
'''
import os
import math
import statistics
import pandas as pd


def eval_text(text, ngram):
    token_list = text.strip().split()
    start_idx, end_idx = 0, ngram
    total_num = 0
    ngram_set = set()
    while end_idx < len(token_list):
        one_ngram_list = token_list[start_idx:end_idx]
        assert len(one_ngram_list) == ngram
        one_ngram = " ".join(one_ngram_list)
        total_num += 1
        ngram_set.add(one_ngram)
        start_idx += 1
        end_idx += 1
    return len(ngram_set), total_num


def eval_one_instance(text, ngram_list):
    res_dict = {}
    for n in ngram_list:
        n_unique, n_total = eval_text(text, n)
        res_dict[n] = {"unique": n_unique, "total": n_total}
    unique_token_set = set(text.strip().split())
    return res_dict, unique_token_set


def measure_repetition_and_diversity(input_text):
    """
    input text: a string
    """
    ngram_list = [2, 3, 4]
    pred_res_dict = {}
    for n in ngram_list:
        pred_res_dict[n] = {}
        pred_res_dict[n]["unique"] = 0
        pred_res_dict[n]["total"] = 0

    pred_unique_token_set = set()
    # for text in text_list:
    stripped_text = input_text.strip("\n").strip()
    one_pred_res_dict, one_pred_uni_token_set = eval_one_instance(stripped_text, ngram_list)

    # unique token set
    pred_unique_token_set = pred_unique_token_set.union(one_pred_uni_token_set)
    # ngram statistic
    for n in ngram_list:
        pred_res_dict[n]["unique"] += one_pred_res_dict[n]["unique"]
        pred_res_dict[n]["total"] += one_pred_res_dict[n]["total"]

    # prediction result
    pred_seq_2 = 1 - (pred_res_dict[2]["unique"] / pred_res_dict[2]["total"])
    # pred_seq_2 = round(pred_seq_2 * 100, 2)
    pred_seq_3 = 1 - (pred_res_dict[3]["unique"] / pred_res_dict[3]["total"])
    # pred_seq_3 = round(pred_seq_3 * 100, 2)
    pred_seq_4 = 1 - (pred_res_dict[4]["unique"] / pred_res_dict[4]["total"])
    # pred_seq_4 = round(pred_seq_4 * 100, 2)
    pred_div = (1 - pred_seq_2 / 100) * (1 - pred_seq_3 / 100) * (1 - pred_seq_4 / 100)

    pred_log_div = -math.log(max(1 - pred_div, math.exp(-20)))  # this is our addition
    # defining 20 manually as the maximal value

    # return pred_seq_2, pred_seq_3, pred_seq_4, pred_div
    # return a dictionary with the ngram repetition levels and diversity
    return {
        "repetition_2": pred_seq_2,
        "repetition_3": pred_seq_3,
        "repetition_4": pred_seq_4,
        "diversity": pred_div,
        "log_diversity": pred_log_div,
    }


def get_diversity_score(df, algo_type):
    all_diversity = []
    all_log_diversity = []
    for _, row in df.iterrows():
        input_text = row[f'output_{algo_type}']
        try:
            results_dict = measure_repetition_and_diversity(input_text)
            all_diversity.append(results_dict['diversity'])
            all_log_diversity.append(results_dict['log_diversity'])
        except:
            None

    avg_diversity = statistics.mean(all_diversity)
    avg_log_diversity = statistics.mean(all_log_diversity)
    return avg_diversity, avg_log_diversity


if __name__ == '__main__':
    algo_type = 'sample'
    file_path = 'results/C4/sample/2024-04-30-01:15:50/dataset_output.jsonl'
    df = pd.read_json(file_path, lines=True)
    avg_diversity, avg_log_diversity = get_diversity_score(df, algo_type)
    print('average diversity: ', avg_diversity)
    print('average log diversity: ', avg_log_diversity)
