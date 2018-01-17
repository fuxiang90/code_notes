from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData


import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB


    gnb = GaussianNB()
    clf = gnb.fit(features_train, labels_train)


    ### fit the classifier on the training features and labels


    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test,pred)
    return accuracy

features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy


print  submitAccuracy()
