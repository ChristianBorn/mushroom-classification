#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas
import numpy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
import classification
import feature_selection
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

def predict_missing(dataset):
    '''
    Funktion zur Vorhersage fehlender Werte
    Nimmt die eingelesen Datenbasis entgegen und füllt fehlende Werte per Vorhersage durch einen KNN aus
    '''
    #Bei der Vorhersage von stalk-root wurde class herausgehalten, da es die eigentliche Zielvariable später ist
    dataset = dataset.replace(numpy.NaN, '?')
    nan_index = (dataset[dataset['stalk-root'] == '?'].index)[0]
    le = LabelEncoder()
    for elem in dataset.columns:
        dataset[elem] = le.fit_transform(dataset[elem])
    #? wurde durch 0 kodiert
    dataset['stalk-root'].replace(dataset.iloc[nan_index,11], numpy.NaN, inplace=True)
    #print(dataset.iloc[nan_index,11])
    dataset_short = dataset.dropna()
    #Entfernen des ursprünglichen Zielwerts
    dataset_training = dataset_short.drop('class', 1)
    Y = dataset_training.iloc[:,10]
    X = dataset_training.drop('stalk-root', 1)
    #Finden der besten Parameter für Datensatz
    parameters = classification.classification_knn(X=X,Y=Y, data_transformation=True)
    best_algorithm = parameters[0]
    best_weight = parameters[1]
    print(' Best algorithm: '+best_algorithm+'\n Best weight: '+best_weight)
    # Filter rows with missing values
    to_predict = dataset[dataset.isnull().any(axis=1)]
    to_predict_X = to_predict.drop('class', 1)
    #Spalte mit fehlenden Werten ist die neue Zielvariable und wird später vorhergesagt
    to_predict_X = to_predict_X.drop('stalk-root', 1)
    #Training des KNN mit gefundenen Parametern auf dataset
    model = KNeighborsClassifier(weights=best_weight,algorithm=best_algorithm, p=2)
    model.fit(X,Y)
    #Vorhersage fehlender Werte auf Basis der korrespondierenden, vorhandenen Werte
    pred = model.predict(to_predict_X)
    to_predict['stalk-root'] = pred
    result = pandas.concat([dataset_short, to_predict])
    result['stalk-root'] = result['stalk-root'].astype(int)
    #Die Datenbasis mit vorhergesagten Werten wird in eine CSV-geschrieben
    result.to_csv('mushrooms_encoded_predicted.csv', ',', encoding='utf-8')
    print(' Number of missing values in new dataset: '+str(result.isnull().sum().sum()))
    print(' Dimensions of new dataset: '+str(result.shape))
    print(' Checking distribution of classes of new dataset')
    print(result['class'].value_counts(True))

def main():
    with open('mushrooms.csv', 'r') as file:
        read_input = pandas.read_csv(file, sep=',', encoding='utf-8')
    #Auf fehlende Werte überprüfen
    with open('dataset traits.txt', 'w') as file:
        file.write(str(read_input.describe(include='all')))
    #Einige Zeilen enthalten '?', was als fehlender Wert behandelt wird
    print('Missing values by rows:')
    print((read_input == '?').sum())
    read_input = read_input.replace('?', numpy.NaN)
    print('Checking for zero values:')
    if read_input.isnull().sum().sum() == 0:
        print('[+] No zero values detected')
    else:
        #Behandlung fehlender Werte
        print('[-] Zero values detected!')
        print('Number of missing values in original dataset: '+str(read_input.isnull().sum().sum()))
        print('[+] Creating dataset with predicted missing values:')
        predict_missing(read_input)
        print('[+] Missing values predicted')
        read_input.dropna(inplace=True)
    #Anzahl der Datensätze und Merkmale zählen
    print('Checking instances and dimensionality:')
    instances = read_input.shape[0]
    dimensions = read_input.shape[1]
    print('[+] number of instances: '+ str(instances))
    print('[+] number of dimensions: ' + str(dimensions))
    #Kodieren kategorieller Daten
    le = LabelEncoder()
    for elem in read_input.columns:
        read_input[elem] = le.fit_transform(read_input[elem])
    #Schreiben der bearbeiteten Daten in CSV
    read_input.to_csv('mushrooms_encoded.csv', ',', encoding='utf-8')
    print('[+] Removed rows with missing values from original dataset')
    #Verteilung der Klassen errechnen
    print('Checking distribution of classes')
    print(read_input['class'].value_counts(True))
    print('[+] Data transformation completed succesfully\n')

if __name__ == '__main__':
    main()
    feature_selection.main()
    classification.main()