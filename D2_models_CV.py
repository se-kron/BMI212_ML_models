import numpy as np
import pandas as pd
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

'''
#########################################################################################
# BMI 212 MADC
# D2_models_CV.py
#
# Code run cross validation on a set of machine learning models using data prepared
# via prep_data.py
#
# Author: Sebastian Kronmueller (sekron)
#########################################################################################
'''

def run_crossval(modality, classifier, classifier_name, datafolder, plotfolder):
    ''' Function to run cross validation for a specific modality using data in datafolder
        Saves results into plotfolder
    '''

    tprs = []
    aucs = []
    y_pred = []
    y_test = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    # Run through folds
    for i in range(5):

        # Load data and labels
        df = pd.read_csv(datafolder + modality + '_fold_' + str(i) + '_data.csv',index_col=0)
        labels = pd.read_csv(datafolder + modality + '_fold_' + str(i) + '_labels.csv',index_col=0)

        # Prep integrated data to ensure consistent indexing
        df = df.transpose()
        df.index.rename('file_accession', inplace=True)
        labels.index.rename('file_accession', inplace=True)
        df = pd.merge(df, labels,left_index=True,right_index=True)

        train = np.argwhere(df['test_data'].to_numpy()==0).flatten()
        test = np.argwhere(df['test_data'].to_numpy()==1).flatten()

        y = df['AD_label'].to_numpy()
        X = df.iloc[:,:-4].to_numpy()

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        classifier = classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(classifier, X[test], y[test], name="ROC fold {}".format(i), alpha=0.3, lw=1,ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        y_pred.append(classifier.predict(X[test].tolist()))
        y_test.append(y[test].tolist())

    y_pred = list(itertools.chain(*y_pred))
    y_test = list(itertools.chain(*y_test))

    accuracy = accuracy_score(y_test,y_pred)
    print(modality + ' ' + classifier_name + " accuracy: ", accuracy)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=2, alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC " + modality + ' ' + classifier_name)
    ax.legend(loc="lower right")
    plt.savefig(plotfolder + modality + '_' + classifier_name + "_ROC_curve.png")
    plt.close()


def main():

    # Folder for data in CV-fold split and result plots
    # Change to do full genome instead of exome only
    datafolder = './cv_data_pvalue_fixed/'
    plotfolder = './cv_data_pvalue_fixed_plots/'

    # Set classifiers
    SVM = svm.SVC(kernel="linear", probability=False, random_state=1)
    RandomForest = RandomForestClassifier(max_features=None)
    LogRegL2 = LogisticRegression(penalty='l2',random_state=2,solver='lbfgs',max_iter=1000)
    LogRegL1 = LogisticRegression(penalty='l1',random_state=2,solver='liblinear',max_iter=1000)
    classifiers = [SVM,RandomForest,LogRegL2,LogRegL1]
    classifier_names = ['SVM','RandomForest','LogRegL2','LogRegL1']

    # Set modalities
    modalities = ['ChIP_CTCF', 'ChIP_H3K27ac', 'ChIP_H3K27me3', 'ChIP_H3K4me3', 'DNase']

    for modality in modalities:
        for i in range(len(classifiers)):
            classifier = classifiers[i]
            classifier_name = classifier_names[i]
            run_crossval(modality, classifier,classifier_name, datafolder, plotfolder)

    print('Done.')

if __name__ == "__main__":
    main()
