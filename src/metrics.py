import os
from typing import List, Dict

import numpy as np
import torch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import util as st_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)
semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def predict_top_k_tokens(prompt, end_before_padding, model, k=1, device='cuda'):
    inputs = prompt.to(device)
    with torch.no_grad():
        logits = model(inputs)
        next_token_logits = logits[0, end_before_padding-1, :]
        top_k_values, top_k_indices = torch.topk(next_token_logits, k)
        top_k_tokens = [idx.item() for idx in top_k_indices]
    return top_k_tokens


def calculate_metrics(true_labels: List[int], predicted_labels: List[List[int]], k: int, true_texts: List[str],
                      pred_texts: List[List[str]]) -> Dict[str, float]:
    true_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true in pred[:k])   # This is a relaxed evaluation since we are considering the top-k predictions, and it is essentially a multi-label classification problem
    false_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true not in pred[:k])
    false_negatives = len(true_labels) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate Mean Average Precision and Mean Reciprocal Rank
    ap_sum = 0
    rr_sum = 0
    for true, pred in zip(true_labels, predicted_labels):
        if true in pred:
            rank = pred.index(true) + 1
            ap_sum += 1 / rank
            rr_sum += 1 / rank
    map_score = ap_sum / len(true_labels)
    mrr_score = rr_sum / len(true_labels)

    # Calculate Semantic Similarity
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    true_embeddings = sentence_model.encode(true_texts)
    pred_embeddings = [sentence_model.encode(texts) for texts in pred_texts]

    semantic_similarities = []
    cosine_similarities_incorrect = []
    for true_emb, pred_emb_list, true_label, pred_label in zip(true_embeddings, pred_embeddings, true_labels,
                                                               predicted_labels):
        similarities = util.cos_sim(true_emb, pred_emb_list)
        semantic_similarities.append(similarities[0][0].item())  # Take the similarity with the top prediction

        if true_label not in pred_label:
            # Calculate cosine similarity for incorrect predictions
            cosine_sim = util.cos_sim(true_emb, pred_emb_list[0]).item()  # Compare with top prediction
            cosine_similarities_incorrect.append(cosine_sim)

    avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities)
    avg_cosine_similarity_incorrect = sum(cosine_similarities_incorrect) / len(
        cosine_similarities_incorrect) if cosine_similarities_incorrect else 0

    sklearn_accuracy = accuracy_score(true_labels, predicted_labels)
    sklearn_precision = precision_score(true_labels, predicted_labels, average='weighted')
    sklearn_recall = recall_score(true_labels, predicted_labels, average='weighted')
    sklearn_f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return {
        "accuracy": true_positives / len(true_labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map": map_score,
        "mrr": mrr_score,
        "avg_semantic_similarity": avg_semantic_similarity,
        "avg_cosine_similarity_incorrect": avg_cosine_similarity_incorrect,
        "sklearn_accuracy": sklearn_accuracy,
        "sklearn_precision": sklearn_precision,
        "sklearn_recall": sklearn_recall,
        "sklearn_f1": sklearn_f1
    }


def cosine_semantic_similarity(sent1: str, sent2: str, model: SentenceTransformer) -> float:
    embeddings = model.encode([sent1, sent2])
    return 1 - cosine(embeddings[0], embeddings[1])


def semantic_similarity(sent1: str, sent2: str) -> float:
    """
    Calculate the semantic similarity between two sentences
    :param sent1:
    :param sent2:
    :return: the cosine similarity between the two sentences based on the sentence-transformers library
    The function is adapted from the following source: https://huggingface.co/tasks/sentence-similarity
    """
    embedding_1 = semantic_model.encode(sent1, convert_to_tensor=True)
    embedding_2 = semantic_model.encode(sent2, convert_to_tensor=True)
    return st_utils.pytorch_cos_sim(embedding_1, embedding_2).item()


def mean_reciprocal_rank(true_label: str, predictions: List[str]) -> float:
    if true_label in predictions:
        return 1.0 / (predictions.index(true_label) + 1)
    return 0.0


def top_k_accuracy(true_labels: str, predictions: List[str], k: int) -> float:
    correct_predictions = 0
    for true_label, prediction in zip(true_labels, predictions):
        true_label = true_label.strip()  # Strip extra spaces from true label
        # top_k_pred = prediction.strip()
        top_k_pred = [pred.strip() for pred in prediction[:k]]  # Strip spaces from predictions

        if true_label in top_k_pred:
            correct_predictions += 1
    return correct_predictions / len(true_labels)


def top_k_precision_recall_f1(true_labels: List[str], predictions: List[List[str]], k: int) -> dict:
    true_positives = 0
    total_predictions = 0
    total_actual = len(true_labels)

    for true_label, prediction in zip(true_labels, predictions):
        true_label = true_label.strip()  # Strip extra spaces from true label
        # top_k_pred = prediction.strip()
        top_k_pred = [pred.strip() for pred in prediction[:k]]  # Strip spaces from top-k predictions
        # print('true_label', true_label)
        # print('top_k_pred', top_k_pred)

        if true_label in top_k_pred:
            true_positives += 1
        total_predictions += k

    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calculate_metrics(true_labels: List[str], predicted_labels: List[List[str]], k: int) -> dict:
    accuracy = top_k_accuracy(true_labels, predicted_labels, k)
    precision, recall, f1 = top_k_precision_recall_f1(true_labels, predicted_labels, k)

    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    semantic_similarities = [cosine_semantic_similarity(true, pred[0], sentence_model) for true, pred in
                             zip(true_labels, predicted_labels)]
    mrr = np.mean([mean_reciprocal_rank(true, pred) for true, pred in zip(true_labels, predicted_labels)])

    return {
        'accuracy': accuracy,
        'precision_at_k': precision,
        'recall_at_k': recall,
        'f1_at_k': f1,
        'mean_semantic_similarity': np.mean(semantic_similarities),
        'mean_reciprocal_rank': mrr
    }


def summary_metrics(combined_results):
    # generate summary statistics of the results
    clean_accuracies = [result['clean_metrics']['accuracy'] for result in combined_results]
    grouped_accuracies = [result['grouped_metrics']['accuracy'] for result in combined_results]
    clean_precision = [result['clean_metrics']['precision_at_k'] for result in combined_results]
    grouped_precision = [result['grouped_metrics']['precision_at_k'] for result in combined_results]
    clean_recall = [result['clean_metrics']['recall_at_k'] for result in combined_results]
    grouped_recall = [result['grouped_metrics']['recall_at_k'] for result in combined_results]
    clean_f1 = [result['clean_metrics']['f1_at_k'] for result in combined_results]
    grouped_f1 = [result['grouped_metrics']['f1_at_k'] for result in combined_results]
    # clean_map = [result['clean_metrics']['map'] for result in combined_results]
    # grouped_map = [result['grouped_metrics']['map'] for result in combined_results]
    clean_mrr = [result['clean_metrics']['mean_reciprocal_rank'] for result in combined_results]
    grouped_mrr = [result['grouped_metrics']['mean_reciprocal_rank'] for result in combined_results]
    clean_semantic_similarity = [result['clean_metrics']['mean_semantic_similarity'] for result in combined_results]
    grouped_semantic_similarity = [result['grouped_metrics']['mean_semantic_similarity'] for result in combined_results]
    # average them and log them
    avg_clean_accuracy = np.mean(clean_accuracies)
    avg_grouped_accuracy = np.mean(grouped_accuracies)
    avg_clean_precision = np.mean(clean_precision)
    avg_grouped_precision = np.mean(grouped_precision)
    avg_clean_recall = np.mean(clean_recall)
    avg_grouped_recall = np.mean(grouped_recall)
    avg_clean_f1 = np.mean(clean_f1)
    avg_grouped_f1 = np.mean(grouped_f1)
    # avg_clean_map = np.mean(clean_map)
    # avg_grouped_map = np.mean(grouped_map)
    avg_clean_mrr = np.mean(clean_mrr)
    avg_grouped_mrr = np.mean(grouped_mrr)
    avg_clean_semantic_similarity = np.mean(clean_semantic_similarity)
    avg_grouped_semantic_similarity = np.mean(grouped_semantic_similarity)
    # calculate the standard deviation
    std_clean_accuracy = np.std(clean_accuracies)
    std_grouped_accuracy = np.std(grouped_accuracies)
    std_clean_precision = np.std(clean_precision)
    std_grouped_precision = np.std(grouped_precision)
    std_clean_recall = np.std(clean_recall)
    std_grouped_recall = np.std(grouped_recall)
    std_clean_f1 = np.std(clean_f1)
    std_grouped_f1 = np.std(grouped_f1)
    # std_clean_map = np.std(clean_map)
    # std_grouped_map = np.std(grouped_map)
    std_clean_mrr = np.std(clean_mrr)
    std_grouped_mrr = np.std(grouped_mrr)
    std_clean_semantic_similarity = np.std(clean_semantic_similarity)
    std_grouped_semantic_similarity = np.std(grouped_semantic_similarity)

    summary = {
        'avg_clean_accuracy': avg_clean_accuracy,
        'avg_grouped_accuracy': avg_grouped_accuracy,
        'avg_clean_precision': avg_clean_precision,
        'avg_grouped_precision': avg_grouped_precision,
        'avg_clean_recall': avg_clean_recall,
        'avg_grouped_recall': avg_grouped_recall,
        'avg_clean_f1': avg_clean_f1,
        'avg_grouped_f1': avg_grouped_f1,
        # 'avg_clean_map': avg_clean_map,
        # 'avg_grouped_map': avg_grouped_map,
        'avg_clean_mrr': avg_clean_mrr,
        'avg_grouped_mrr': avg_grouped_mrr,
        'avg_clean_semantic_similarity': avg_clean_semantic_similarity,
        'avg_grouped_semantic_similarity': avg_grouped_semantic_similarity,
        'std_clean_accuracy': std_clean_accuracy,
        'std_grouped_accuracy': std_grouped_accuracy,
        'std_clean_precision': std_clean_precision,
        'std_grouped_precision': std_grouped_precision,
        'std_clean_recall': std_clean_recall,
        'std_grouped_recall': std_grouped_recall,
        'std_clean_f1': std_clean_f1,
        'std_grouped_f1': std_grouped_f1,
        # 'std_clean_map': std_clean_map,
        # 'std_grouped_map': std_grouped_map,
        'std_clean_mrr': std_clean_mrr,
        'std_grouped_mrr': std_grouped_mrr,
        'std_clean_semantic_similarity': std_clean_semantic_similarity,
        'std_grouped_semantic_similarity': std_grouped_semantic_similarity
    }
    return summary
