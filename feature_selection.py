#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def selection_SelectKBest(X,Y,feature_names):
    test = SelectKBest(score_func=chi2,k='all')
    fit = test.fit(X,Y)
    features = fit.transform(X)
    y_pos = numpy.arange(len(feature_names))
    plt.bar(y_pos,numpy.log10(fit.scores_), width=.35)
    plt.ylabel('Score')
    plt.xlabel('Feature')
    plt.xticks(y_pos + .35 / 2, feature_names, rotation='vertical')
    plt.title('Zusammenhang der Merkmale mit abhängiger Variable y')
    plt.show()

def main():
    read_input = pandas.read_csv('mushrooms_encoded.csv', sep=',', encoding='utf-8')
    feature_names = read_input.columns.values[2:]
    array = read_input.values
    X = array[:,2:]
    Y = array[:,1]
    selection_SelectKBest(X,Y,feature_names)    

if __name__ == '__main__':
    main()