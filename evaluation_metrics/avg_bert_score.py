import os
import bert_score

def get_average_bert_score(list1, list2):
    # Load BERT score model
    scorer = bert_score.BERTScorer(model_type='bert-base-uncased', lang='en')
    
    # Compute BERT scores for each pair of texts
    all_scores = []
    for text1, text2 in zip(list1, list2):
        score = scorer.score([text1], [text2])
        all_scores.append(score)
    
    # Extract the P, R, F1 scores and calculate average
    precision_scores = [score[0].item() for score in all_scores]
    recall_scores = [score[1].item() for score in all_scores]
    f1_scores = [score[2].item() for score in all_scores]
    
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)
    
    return average_precision, average_recall, average_f1

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    list1 = ["This is the first sentence.", "Another sentence here."]
    list2 = ["This is the second sentence.", "Yet another sentence."]
    avg_precision, avg_recall, avg_f1 = get_average_bert_score(list1, list2)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1 Score:", avg_f1)
