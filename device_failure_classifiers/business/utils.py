def plot_pr():
    preds = pd.concat([y_train_rlg2,predictionsBasedOnKFolds.loc[:,1]], axis=1)
    preds.columns = ['trueLabel','prediction']
    predictionsBasedOnKFoldsLogisticRegression = preds.copy()

    precision, recall, thresholds = precision_recall_curve(preds['trueLabel'],
                                                       preds['prediction'])

    average_precision = average_precision_score(preds['trueLabel'],
                                            preds['prediction'])

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

    plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
          average_precision))

def plotResults(trueLabels, anomalyScores, returnPreds = False):
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
    
    plt.title('Precision-Recall curve: Average Precision = \
    {0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = roc_curve(preds['trueLabel'], \
                                     preds['anomalyScore'])
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: \
    Area under the curve = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")
    plt.show()
    
    if returnPreds==True:
        return preds