def f1_accessed_columns(ground_truth, prediction):
    gt = set(ground_truth)
    pred = set(prediction)

    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def recall_correlated_columns(ground_truth, prediction):
    if len(ground_truth) == 0:
        return 1.0  # If no ground truth, consider recall as perfect
    true_positives = sum(1 for gt in ground_truth if gt in prediction)
    total_ground_truth = len(ground_truth)
    return true_positives / total_ground_truth if total_ground_truth > 0 else 0.0
