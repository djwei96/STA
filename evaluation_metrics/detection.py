import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve

#TODO: under development, need to check
def get_tpr_and_f1_at_fpr(y_true, y_scores, target_fpr):
    """
    Computes TPR and F1-score at a specific FPR, and the best F1-score across all thresholds.

    Parameters:
    y_true (list or array): True binary labels.
    y_scores (list or array): Predicted probabilities for the positive class.
    target_fpr (float): The target false positive rate (default is 0.01 for 1% FPR).

    Returns:
    dict: TPR and F1-score at the specified FPR, and the best F1-score.
    """
    # Compute ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    
    # Find the threshold closest to the target FPR and corresponding TPR
    idx_target_fpr = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target_fpr = tpr[idx_target_fpr]
    
    # Compute PR Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    
    # Compute F1-scores for each point on the PR Curve
    valid_scores = (precision + recall) != 0  # Avoid division by zero
    f1_scores = np.where(valid_scores, 2 * (precision * recall) / (precision + recall), 0)
    
    # F1-score at the same threshold as the target FPR
    if roc_thresholds[idx_target_fpr] in pr_thresholds:
        f1_at_target_fpr = f1_scores[np.where(pr_thresholds >= roc_thresholds[idx_target_fpr])[0][0]]
    else:
        f1_at_target_fpr = 'N/A'  # In case the exact threshold isn't found in PR thresholds
    
    # Best F1-score across all thresholds
    best_f1 = np.max(f1_scores)
    
    return {
        'tpr': tpr_at_target_fpr,
        'f1': f1_at_target_fpr,
        'best_f1': best_f1
    }


def get_detection_score(df, algo_type, detection_z_score_threshold, target_fpr_list=[0.01, 0.05, 0.1]):
    '''
    positive label: machine/watermark -> 1
    negative label: no watermark -> 0
    '''
    detection_df = df.copy(deep=True)
    detection_df = detection_df.dropna(subset=[f'z_score_{algo_type}', f'z_score_wo_{algo_type}']) 
    predicted_watermark_scores = detection_df[f'z_score_{algo_type}'].tolist()
    predicted_wo_watermark_scores = detection_df[f'z_score_wo_{algo_type}'].tolist()
    all_predicted_scores = predicted_watermark_scores + predicted_wo_watermark_scores

    predicted_watermark_labels = [1 if value > detection_z_score_threshold else 0 for value in predicted_watermark_scores]
    predicted_wo_watermark_labels = [1 if value > detection_z_score_threshold else 0 for value in predicted_wo_watermark_scores]
    all_predicted_labels = predicted_watermark_labels + predicted_wo_watermark_labels

    ture_watermark_labels = [1]*len(predicted_watermark_scores)
    ture_wo_watermark_labels = [0]*len(predicted_wo_watermark_scores)
    all_true_labels = ture_watermark_labels + ture_wo_watermark_labels

    eval_results_dict = {}
    # calculate overall score
    f1 = f1_score(all_true_labels, all_predicted_labels)
    auc = roc_auc_score(all_true_labels, all_predicted_labels)
    tn, fp, fn, tp = confusion_matrix(all_true_labels, all_predicted_labels).ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 'N/A'
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 'N/A'
    fpr = fp / (fp + tn) if (tp + fn) != 0 else 'N/A'
    fnr = fn / (fn + tp) if (tp + fn) != 0 else 'N/A'
    eval_results_dict['all'] = {'f1': f1, 'auc': auc, 'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr}

    # calculate score at a given fpr rate
    all_predicted_probs = [norm.cdf(z) for z in all_predicted_scores]
    for target_fpr in target_fpr_list:
        target_eval_results = get_tpr_and_f1_at_fpr(all_true_labels, all_predicted_probs, target_fpr)
        eval_results_dict[f'target_fpr@{target_fpr}'] = target_eval_results
    return eval_results_dict
