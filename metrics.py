from sklearn.metrics import confusion_matrix
import numpy as np


def get_confusion_matrix(predictions, ground_truth, label_values):
    confusion_mat = confusion_matrix(
        ground_truth,
        predictions,
        range(len(label_values)))

    return confusion_mat


def accuracy(predictions, ground_truth):
    return 100 * np.mean(predictions==ground_truth)


def accuracy_using_cfm(confusion_mat):
    total = sum(sum(confusion_mat))
    accuracy = sum([confusion_mat[x][x] for x in range(len(confusion_mat))])
    accuracy *= 100 / float(total)

    return accuracy

def iou(predictions, ground_truth):
    inter = predictions[:,0,:,:]*ground_truth

    union = predictions[:,0,:,:] + ground_truth - inter

    return 100*inter.sum()/union.sum()


def F1_score(confusion_mat, label_values):
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * confusion_mat[i, i] / ( 2*confusion_mat[i, i]+np.sum(confusion_mat[i, :]) + np.sum(confusion_mat[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))


def get_kappa_coefficient(confusion_mat):
    total = np.sum(confusion_mat)
    pa = np.trace(confusion_mat) / float(total)
    pe = np.sum(np.sum(confusion_mat, axis=0) * np.sum(confusion_mat, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))


def get_values_of_all_metrics(predictions, ground_truth, label_values):
    confusion_mat = get_confusion_matrix(predictions, ground_truth, label_values)

    print("Confusion matrix :")
    print(confusion_mat)

    print("---")

    # Compute global accuracy
    acc = accuracy_using_cfm(confusion_mat)
    print('Global Accuracy:', acc)
    print("---")

    F1_score(confusion_mat, label_values)
    print("---")

    # Compute kappa coefficient
    get_kappa_coefficient(confusion_mat)

    return acc
