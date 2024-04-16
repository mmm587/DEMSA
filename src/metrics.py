import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    non_zeros = np.array([i for i, e in enumerate(truths) if e != 0 or (not exclude_zero)])
    test_preds_a7 = np.clip(results, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(truths,  a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(results, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(truths,  a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(results - truths))  
    corr = np.corrcoef(results, truths)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    f_score = f1_score((results[non_zeros] > 0), (truths[non_zeros] > 0), average='weighted')

    binary_truth = (truths[non_zeros] > 0)
    binary_preds = (results[non_zeros] > 0)

    acc = accuracy_score(binary_truth, binary_preds)
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", acc)

    print("-" * 50)
    return acc, mae, corr, f_score


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:

        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(results[:, emo_ind], axis=1)
            test_truth_i = truths[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:

        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(results, axis=1)
        test_truth_i = truths
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)

    return acc, f1

