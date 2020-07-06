import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_predict,cross_val_score,ShuffleSplit
from sklearn.metrics import roc_curve,auc, roc_auc_score,accuracy_score, recall_score, f1_score, confusion_matrix, log_loss
from sklearn.metrics import balanced_accuracy_score,precision_score,average_precision_score, precision_recall_curve

def plotPerformance(trueLabels, anomalyScores, returnPreds = False):
    #"""
    #trueLabels is a series, in notebook is y_train or a derived series
    #anomaly socres are the prediction made by the classifier
    #it can be the corresponding y_preds
    #in notebook is predictionsBasedOnKFolds.loc[:,1]]

    #"""
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, thresholds = precision_recall_curve(preds['trueLabel'],preds['anomalyScore'])
    average_precision = average_precision_score(preds['trueLabel'],preds['anomalyScore'])
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    corte = np.argmax(fscore)
    print('Optimal Threshold=%f, F-Score=%.3f' % (thresholds[corte], fscore[corte]))

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')
    plt.scatter(recall[corte], precision[corte], marker='o', color='blue', label='Best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['anomalyScore'])
    areaUnderROC = auc(fpr, tpr)
    J = tpr - fpr
    optim = np.argmax(J)
    best_thresh = thresholds[optim]
    print('Optimal Threshold=%f' % (best_thresh))

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # optimal cut
    plt.scatter(fpr[optim], tpr[optim], marker='o', color='blue', label='Best')
    #
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")
    plt.show()
    
    if returnPreds==True:
        return preds

