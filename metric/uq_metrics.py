"""
Functions for uncertainty quantification
"""

import numpy as np
from sklearn.metrics import roc_auc_score


# CAGRI: To-be implemented
# - class-wise calibration error
# - adaptive calibration error

# To-be re-factored
# accuracy as to calculate any top-k accuracy
# auc in native numpy


# accuracy
def accuracy(y_true, y_pred):
    """Accuracy classification score.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    Returns
    -------
    score : float
    """
    return np.mean(y_true == y_pred)


# accuracytop5
# def get_acc5(preds, targets, **args):
#    preds = torch.Tensor(preds)
#    targets = torch.LongTensor(targets)
#    return accuracy(preds, targets, topk=(5,))[0].item()/100.


# ece
def ece(y_true, y_pred, confs, proba_pred, n_bins=15, min_pred=0):
    """
    Compute ECE
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array
        Ground truth (correct) labels.
    preds : 1d array-like, or label indicator array
        Predicted labels, as returned by a classifier.
    confs : 1d array-like
        Predicted labels'probabilities, as normalized by a classifier.
        confidences, predictions = torch.max(softmaxes, 1)
    Returns
    -------
    score : float
    """
    # if min_pred == "minpred":
    #     min_pred = min(proba_pred)
    # else:
    #     assert min_pred >= 0

    # bin_boundaries = np.linspace(min_pred, 1., n_bins + 1)
    bin_boundaries = np.linspace(0, 1., n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    acc_in_bin_list = []
    avg_confs_in_bins = []
    list_prop_bin = []
    ece = 0.0
    accurate = (y_true == y_pred)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confs > bin_lower, confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        list_prop_bin.append(prop_in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accurate[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            acc_in_bin_list.append(accuracy_in_bin)
            avg_confs_in_bins.append(avg_confidence_in_bin)
            ece += np.abs(delta) * prop_in_bin

        else:
            avg_confs_in_bins.append(None)
            acc_in_bin_list.append(None)

    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece, avg_confs_in_bins, acc_in_bin_list


# class-wise ece

# entropy

def entropy(preds):
    return np.sum(-preds * np.log(preds), axis=1)


# log likelihood
def ll(y_true, preds):
    """
    Compute log likelihood
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array
        Ground truth (correct) labels.
    preds : predicted_probs. size (n_samples, c_classes) nd array-like
        Normalized logits, or predicted probabilities for all samples.
    Returns
    -------
    score : float
    """
    preds_target = preds[np.arange(len(y_true)), y_true]
    return np.log(1e-12 + preds_target).mean()


# auc
# CAGRI: To-do: implement this function
def auc(accurate, confs):
    """
    auc(AUROC):Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases
        expect labels with shape (n_samples,) while the multilabel case expects
        binary label indicators with shape (n_samples, n_classes).
    preds : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

    Returns
    -------
    auc : float
    """

    return roc_auc_score(accurate, confs)


# brier
def brier(y_true, preds):
    """
    Compute brier score
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases
        expect labels with shape (n_samples,) while the multilabel case expects
        binary label indicators with shape (n_samples, n_classes).
    preds : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

    Returns
    -------
    auc : float
    """
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(y_true)), y_true] = 1.0

    return np.mean(np.sum((preds - one_hot_targets) ** 2, axis=1))


# collect all metrics

def metrics_evaluate(y_true, preds, **args):
    y_pred = preds.argmax(axis=1)  # model's class prediction
    confs = preds.max(axis=1)  # highest predicted probability
    # preds_target =  preds[len(y_true),y_true] # the probability predicted for the true_class
    accurate = (y_true == y_pred)

    score_acc = accuracy(y_true, y_pred)
    score_ece, _, _ = ece(y_true, y_pred, confs)
    # score_classece =
    # score_ace =
    score_entropy = entropy(preds).mean()
    score_nll = -ll(y_true, preds)
    score_auc = auc(accurate, confs)
    score_brier = brier(y_true, preds)

    # getting scores
    metrics_dict = {'accuracy': score_acc,
                    'ece': score_ece,
                    # 'class-ece': ,
                    # 'ace': ,
                    'entropy': score_entropy,
                    'nll': score_nll,
                    'auc': score_auc,
                    'brier': score_brier}
    print(metrics_dict)
    return metrics_dict

if __name__ == "__main__":

   metrics_evaluate()