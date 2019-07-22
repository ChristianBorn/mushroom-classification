#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import pandas
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
import sklearn.naive_bayes as NB
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def measure_time(start=0):
    #Funktion zur Zeitmessung in Sekunden
    if start == 0:
        start = time.time()
        return start
    else:
        end = time.time()
        elapsed = end - start
        return elapsed

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def classification_lr(with_predicted_values=False, instant_save=False):
    '''
    Funktion für logistische Regression
    Ergebnisse werden als .txt und in Diagrammen gespeichert
    Nimmt zwei Argumente, die standardmäßig False sind
    with_predicted_values bestimmt, auf welcher Datenbasis die Funktion ausgeführt wird
    instant_save bestimmt, ob die erstellten Grafiken direkt gespeichert oder zur Laufzeit gezeigt werden
    '''
    if with_predicted_values == False:
        read_input = pandas.read_csv('mushrooms_encoded.csv', sep=',', encoding='utf-8')
        mode = '\nOhne vorhergesagte fehlende Werte'
        logtitle = 'removed_missing_lr.txt'
        figtitle = 'instant_save/LR_without_predicted'
        print('Training Logistic Regression on dataset without predicted values')
    else:
        read_input = pandas.read_csv('mushrooms_encoded_predicted.csv', sep=',', encoding='utf-8')
        mode = '\nMit vorhergesagten fehlenden Werten'
        logtitle = 'predicted_missing_lr.txt'
        figtitle = 'instant_save/LR_with_predicted'
        print('Training Logistic Regression on dataset with predicted values')
    feature_names = read_input.columns.values[2:]
    X = read_input.iloc[:,2:]  
    Y = read_input.iloc[:,1]
    c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    regularization_modes = ['l1', 'l2']
    print('Number of C values to try: '+str(len(c_values)))
    accs = {'l1': [], 'l2': []}
    
    for reg in regularization_modes:
        count = 1
        for val in c_values:
            # Aufteilung in Trainigs- und Testset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
            print('C value '+str(count)+' of '+str(len(c_values)))
            count += 1
            lr = LogisticRegression(penalty=reg,C=val)
            lr.fit(X_train, Y_train)
            pred = lr.predict(X_test)
            accs[reg].append(accuracy_score(Y_test, pred))
    max_acc_l1 = max(accs['l1'])
    max_c_l1 = c_values[accs['l1'].index(max_acc_l1)]
    print('Best accuracy for l1: '+str(max_acc_l1))
    print('Best C value for l1: '+str(max_c_l1))
    max_acc_l2 = max(accs['l2'])
    max_c_l2 = c_values[accs['l2'].index(max_acc_l2)]
    print('Best accuracy for l2: '+str(max_acc_l2))
    print('Best C value for l2: '+str(max_c_l2))
    cv = ShuffleSplit(n_splits=100, test_size=0.4, random_state=0)
    print('Plotting learning curve...')
    start = measure_time()
    plot_learning_curve(LogisticRegression(penalty='l1', C=max_c_l1), 'Lernkurve Logistische Regression\nmit Parametern l1, C='+str(max_c_l1)+mode, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    elapsed_time_l1 = measure_time(start)
    if instant_save == True:
        plt.savefig(figtitle+'_l1'+'.png',bbox_inches='tight')
    else:
        plt.show()
    print('Plotting learning curve...')
    start = measure_time()
    plot_learning_curve(LogisticRegression(penalty='l2', C=max_c_l2), 'Lernkurve Logistische Regression\nmit Parametern l2, C='+str(max_c_l2)+mode, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    elapsed_time_l2 = measure_time(start)
    if instant_save == True:
        plt.savefig(figtitle+'_l2'+'.png',bbox_inches='tight')
    else:
        plt.show()
    logentry = 'Bester Durchlauf l1:\n Präzision: '+str(max_acc_l1)+'\n C-Value: '+str(max_c_l1)+'\n Benötigte Zeit in sek: '+str(elapsed_time_l1)+'\n'+'Bester Durchlauf l2:\n Präzision: '+str(max_acc_l2)+'\n Benötigte Zeit in sek: '+str(elapsed_time_l2)+'\n C-Value: '+str(max_c_l2)      
    with open(logtitle, 'w') as file:
        file.write(logentry)
    print('Plotting accuracy for C values...')
    plt.clf()
    plt.plot(c_values, accs['l1'], label='l1')
    plt.plot(c_values, accs['l2'], label='l2')
    plt.legend(loc=4, prop={'size':15})
    plt.xlabel('Werte für Regularisierung C')
    plt.ylabel('Präzision der Vorhersage')
    plt.title('Präzision für Regularisierungswerte zwischen 0,00001 und 100'+mode)
    plt.xticks(np.arange(0,105,5))
    if instant_save == True:
        plt.savefig(figtitle+'_Regularisierung.png', bbox_inches='tight')
    else:
        plt.show()
    print('[+] Logistic Regression completed\n\n')

def classification_svm(with_predicted_values=False, instant_save=False):
    '''
    Funktion zum Training von SVMs
    Ergebnisse werden als .txt und in Diagrammen gespeichert
    Nimmt zwei Argumente, die standardmäßig False sind
    with_predicted_values bestimmt, auf welcher Datenbasis die Funktion ausgeführt wird
    instant_save bestimmt, ob die erstellten Grafiken direkt gespeichert oder zur Laufzeit gezeigt werden
    '''
    if with_predicted_values == False:
        read_input = pandas.read_csv('mushrooms_encoded.csv', sep=',', encoding='utf-8')
        mode = '\nOhne vorhergesagte fehlende Werte'
        logtitle = 'removed_missing_svm.txt'
        figtitle = 'instant_save/SVM_without_predicted'
        print('Training SVM on dataset without predicted values')
    else:
        read_input = pandas.read_csv('mushrooms_encoded_predicted.csv', sep=',', encoding='utf-8')
        mode = '\nMit vorhergesagten fehlenden Werten'
        logtitle = 'predicted_missing_svm.txt'
        figtitle = 'instant_save/SVM_with_predicted'
        print('Training SVM on dataset with predicted values')
    feature_names = read_input.columns.values[2:]
    X = read_input.iloc[:,2:]  
    Y = read_input.iloc[:,1]
    #c_values = np.arange(0.0001, 10, 0.001)
    c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    print('Number of C values to try: '+str(len(c_values)))
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    print('Number of kernels to try: '+str(len(kernels)))
    accs = {'linear': [], 'poly': [], 'rbf': [], 'sigmoid': []}
    times = {}
    cv = ShuffleSplit(n_splits=100, test_size=0.4, random_state=0)
    for kernel in kernels:
        print('Training SVM with kernel: '+kernel)
        count = 1
        for val in c_values:
            # Aufteilung in Trainigs- und Testset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
            print(str(count)+'/'+str(len(c_values)))
            count += 1
            model = SVC(C=val, kernel=kernel,verbose=0)
            model.fit(X_train, Y_train)
            pred = model.predict(X_test)
            accs[kernel].append(accuracy_score(Y_test, pred))
        best_c = c_values[accs[kernel].index(max(accs[kernel]))]
        print('Best accuracy for '+kernel+': '+str(max(accs[kernel])))
        print('Best C value for '+kernel+': '+str(best_c))
        print('Plotting learning curve...')
        start = measure_time()
        plot_learning_curve(SVC(C=best_c, kernel=kernel,verbose=0), 'Lernkurve SVM mit kernel: '+kernel+' und C='+str(best_c)+mode, X, Y, ylim=(0, max(accs[kernel])+0.1), cv=cv, n_jobs=4)
        elapsed_time = measure_time(start)
        times[kernel] = elapsed_time
        if instant_save == True:
            plt.savefig(figtitle+'_'+kernel+'.png',bbox_inches='tight')
        else:
            plt.show()
    logentry = 'Kernel: linear\n'+' Beste Präzision: '+str(max(accs['linear']))+'\n Bester C-Wert: '+str(c_values[accs['linear'].index(max(accs['linear']))])+'\n Zeit: '+str(times['linear'])+'\n'
    logentry = logentry+'Kernel: poly\n'+' Beste Präzision: '+str(max(accs['poly']))+'\n Bester C-Wert: '+str(c_values[accs['poly'].index(max(accs['poly']))])+'\n Zeit: '+str(times['poly'])+'\n'
    logentry = logentry+'Kernel: rbf\n'+' Beste Präzision: '+str(max(accs['rbf']))+'\n Bester C-Wert: '+str(c_values[accs['rbf'].index(max(accs['rbf']))])+'\n Zeit: '+str(times['rbf'])+'\n'
    logentry = logentry+'Kernel: sigmoid\n'+' Beste Präzision: '+str(max(accs['sigmoid']))+'\n Bester C-Wert: '+str(c_values[accs['sigmoid'].index(max(accs['sigmoid']))])+'\n Zeit: '+str(times['sigmoid'])
    with open(logtitle, 'w') as file:
        file.write(logentry)
    print('Plotting accuracy for C values...')
    plt.clf()
    for elem in accs.keys():
        plt.plot(c_values, accs[elem], label=elem)
    plt.legend(loc=4, prop={'size':15})
    plt.xlabel('Werte für Regularisierung C')
    plt.ylabel('Präzision der Vorhersage')
    plt.title('Präzision für Regularisierungswerte zwischen 0,00001 und 1000')
    plt.xticks(np.arange(0,1000,20))
    if instant_save == True:
        plt.savefig(figtitle+'_Regularisierung.png',bbox_inches='tight')
    else:
        plt.show()
    print('Plotting accuracy for C values (Snapshot)...')
    plt.clf()
    for elem in accs.keys():
        plt.plot(c_values[:-2], accs[elem][:-2], label=elem)
    plt.legend(loc=4, prop={'size':15})
    plt.xlabel('Werte für Regularisierung C')
    plt.ylabel('Präzision der Vorhersage')
    plt.title('Präzision für Regularisierungswerte zwischen 0,00001 und 10')
    plt.xticks(np.arange(0,10.5,0.5))
    if instant_save == True:
        plt.savefig(figtitle+'_Regularisierung.png',bbox_inches='tight')
    else:
        plt.show()
    print('SVM Completed\n\n')

def classification_knn(X=pandas.DataFrame([]),Y=pandas.DataFrame([]),with_predicted_values=False, instant_save=False, data_transformation=False):
    '''
    Funktion zum Training von KNNs
    Ergebnisse werden als .txt und in Diagrammen gespeichert
    X und Y sind standardmäßg leer, sind sie nicht leer, werden anstatt die Datenbasis zu laden diese zur Berechnung verwendet
    with_predicted_values bestimmt, auf welcher Datenbasis die Funktion ausgeführt wird
    instant_save bestimmt, ob die erstellten Grafiken direkt gespeichert oder zur Laufzeit gezeigt werden
    '''
    if not X.empty and not Y.empty:
        mode = '\nAuf unbearbeiteter Datenbasis'
        logtitle = 'initial_prediction.txt'
        figtitle = 'instant_save/KNN_initial_prediction'
        print('Training KNN on initial raw Dataset')
    elif with_predicted_values == False:
        read_input = pandas.read_csv('mushrooms_encoded.csv', sep=',', encoding='utf-8')
        mode = '\nOhne vorhergesagte fehlende Werte'
        logtitle = 'removed_missing_knn.txt'
        figtitle = 'instant_save/KNN_without_predicted'
        print('Training KNN on dataset without predicted values')
    else:
        read_input = pandas.read_csv('mushrooms_encoded_predicted.csv', sep=',', encoding='utf-8')
        mode = '\nMit vorhergesagten fehlenden Werten'
        logtitle = 'predicted_missing_knn.txt'
        figtitle = 'instant_save/KNN_with_predicted'
        print('Training KNN on dataset with predicted values')
    if X.empty and Y.empty:
        feature_names = read_input.columns.values[2:]
        X = read_input.iloc[:,2:]  
        Y = read_input.iloc[:,1]
    algorithms = ['ball_tree', 'kd_tree', 'brute']
    weights = ['uniform', 'distance']
    accs = {}
    cv = ShuffleSplit(n_splits=100, test_size=0.4, random_state=0)
    for algorithm in algorithms:
        print('\nTraining model with algorithm: '+algorithm)
        for weight in weights:
            print('Training model with weight: '+weight)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
            model = KNeighborsClassifier(weights=weight,algorithm=algorithm, p=2)
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            accs[algorithm+','+weight] = [accuracy_score(Y_test, pred)]
            if data_transformation == False:
                start = measure_time()
                plot_learning_curve(KNeighborsClassifier(weights=weight,algorithm=algorithm), 'Lernkurve KNN mit '+algorithm+', Gewichtung '+weight+mode, X, Y, ylim=(.8, 1+0.01), cv=cv, n_jobs=4)
                elapsed_time = measure_time(start)
                accs[algorithm+','+weight].append(elapsed_time)
            print('Accuracy for KNN with '+algorithm+' and weight '+weight+': '+str(accuracy_score(Y_test, pred)))
    best_pair_log = max(accs, key=accs.get)
    print('\nBest accuracy of '+str(accs[best_pair_log])+' achieved with pair: '+best_pair_log)
    best_pair = max(accs, key=accs.get).split(',')
    best_algorithm = best_pair[0]
    best_weight = best_pair[1]
    print('Plotting learning curve...')
    start = measure_time()
    plot_learning_curve(KNeighborsClassifier(weights=best_weight,algorithm=best_algorithm), 'Lernkurve KNN mit '+best_algorithm+', Gewichtung '+best_weight+mode, X, Y, ylim=(.8, 1+0.01), cv=cv, n_jobs=4)
    elapsed_time = measure_time(start)
    if instant_save == True:
        plt.savefig(figtitle+'.png',bbox_inches='tight')
    else:
        plt.show()
    logentry = 'Alle Präzisionswerte: '+str(accs)+'\n Zeit für bestes Paar '+best_pair_log+' in Sek: '+str(elapsed_time)
    with open(logtitle, 'w') as file:
        file.write(logentry)
    print('KNN completed\n\n')
    return best_pair


def classification_bayes(with_predicted_values=False, instant_save=False):
    '''
    Funktion zum Training von Naive Bayes
    Ergebnisse werden als .txt und in Diagrammen gespeichert
    with_predicted_values bestimmt, auf welcher Datenbasis die Funktion ausgeführt wird
    instant_save bestimmt, ob die erstellten Grafiken direkt gespeichert oder zur Laufzeit gezeigt werden
    '''
    if with_predicted_values == False:
        read_input = pandas.read_csv('mushrooms_encoded.csv', sep=',', encoding='utf-8')
        mode = '\nOhne vorhergesagte fehlende Werte'
        logtitle = 'removed_missing_bayes.txt'
        figtitle = 'instant_save/bayes_without_predicted'
        print('Training Naive Bayes on dataset without predicted values')
    else:
        read_input = pandas.read_csv('mushrooms_encoded_predicted.csv', sep=',', encoding='utf-8')
        mode = '\nMit vorhergesagten fehlenden Werten'
        logtitle = 'predicted_missing_bayes.txt'
        figtitle = 'instant_save/bayes_with_predicted'
        print('Training Naive Bayes on dataset with predicted values')
    feature_names = read_input.columns.values[2:]
    X = read_input.iloc[:,2:]  
    Y = read_input.iloc[:,1]
    accs = {}
    times = {}
    models = {'Gaussian': NB.GaussianNB(), 'Multinomial': NB.MultinomialNB(), 'Bernoulli': NB.BernoulliNB()}
    cv = ShuffleSplit(n_splits=100, test_size=0.4, random_state=0)
    for model in models.keys():
        print('Training '+model+' Naive Bayes')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
        clf = models[model]
        clf.fit(X_train, Y_train)
        pred = clf.predict(X_test)
        accs[model] = accuracy_score(Y_test, pred)
        print('Accuracy for '+model+' Naive Bayes: '+str(accuracy_score(Y_test, pred)))
        print('Plotting learning curve...')
        start = measure_time()
        plot_learning_curve(models[model], 'Lernkurve '+model+' Naive Bayes'+mode, X, Y, ylim=(0.65, 1.01), cv=cv, n_jobs=4)
        elapsed_time = measure_time(start)
        times[model] = elapsed_time
        if instant_save == True:
            plt.savefig(figtitle+'_'+model+'.png',bbox_inches='tight')
        else:
            plt.show()
    best_model = max(accs, key=accs.get)
    print('Highest Accuracy of '+str(accs[best_model])+' achieved with '+best_model+' Naive Bayes')
    logentry = 'Alle Präzisionswerte: '+str(accs)+'\n Alle Zeiten in Sek: '+str(times)
    with open(logtitle, 'w') as file:
        file.write(logentry)
    

def main():
    #Klassifikation ohne vorhergesagte Werte
    classification_lr(with_predicted_values=False, instant_save=False)
    classification_svm(with_predicted_values=False, instant_save=False)
    classification_knn(with_predicted_values=False, instant_save=False)
    classification_bayes(with_predicted_values=False, instant_save=False)
    #Klassifikation mit vorhergesagten Werte
    classification_lr(with_predicted_values=True, instant_save=False)
    classification_svm(with_predicted_values=True, instant_save=False)
    classification_knn(with_predicted_values=True, instant_save=False)
    classification_bayes(with_predicted_values=True, instant_save=False)
    
if __name__ == '__main__':
    main()