#!/usr/bin/python

""" lecture and example code for decision tree unit """

from class_vis import prettyPicture, output_image
from classifyDT import classify
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracies(acc):
    return {"acc": round(acc, 3)}

### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)




test_pred = clf.predict(features_test)


#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_test, test_pred)
print submitAccuracies(accuracy)

### be sure to compute the accuracy on the test set



