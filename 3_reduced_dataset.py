#!/usr/bin/python
import nltk
import re,time,os,sys
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
import idx2numpy


best_accuracy_adaboost = {}
best_accuracy_forest = {}
best_number_of_tree_adaboost = {}
best_number_of_tree_forest = {}
def find_best_adaboost_classifier(number_of_trees,max_depth):
	adaboost_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth),n_estimators = number_of_trees)
	
	adaboost_classifier.fit(training_image_data_hog, training_label_data)

	adaboost_classifier_accuracy = adaboost_classifier.score(testing_image_data_hog, testing_label_data)
	print "\Adaboost classifier accuracyfor max_depth="+str(max_depth)+" and number of trees = "+ str(number_of_trees)+" is "+ str(adaboost_classifier_accuracy)
	
	if max_depth in best_accuracy_adaboost:
		if  best_accuracy_adaboost[max_depth] < adaboost_classifier_accuracy:
			best_accuracy_adaboost[max_depth] = adaboost_classifier_accuracy
			best_number_of_tree_adaboost[max_depth] = number_of_trees
	else :
		# print "\nEntered else case in adaboost"
		best_accuracy_adaboost[max_depth] = adaboost_classifier_accuracy
		best_number_of_tree_adaboost[max_depth] = number_of_trees

def find_best_random_forest_classifier(number_of_trees,max_depth):
	random_forest_classifier = RandomForestClassifier(n_estimators=number_of_trees, max_depth=max_depth)
	random_forest_classifier.fit(training_image_data_hog, training_label_data)
	random_forest_classifier_accuracy = random_forest_classifier.score(testing_image_data_hog, testing_label_data)
	print "\nRandom Forest accuracy with max_depth="+str(max_depth)+" and number of trees = "+str(number_of_trees)+ " is "+ str(random_forest_classifier_accuracy)

	if max_depth in best_accuracy_forest:
		if  best_accuracy_forest[max_depth] < random_forest_classifier_accuracy:
			best_accuracy_forest[max_depth] = random_forest_classifier_accuracy
			best_number_of_tree_forest[max_depth] = number_of_trees
	else :
		# print "\nEntered else case in forest"
		best_accuracy_forest[max_depth] = random_forest_classifier_accuracy
		best_number_of_tree_forest[max_depth] = number_of_trees

if __name__ == '__main__':
	print "Building Data set"	
	training_image_data = idx2numpy.convert_from_file("train-images.idx3-ubyte")
	training_image_data_hog = [hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3, 3))
					for img in training_image_data]
	training_label_data = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
	testing_image_data = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
	testing_image_data_hog = [hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
					for img in testing_image_data]
	testing_label_data = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")
	print "Dataset is complete"

	training_image_data_hog = training_image_data_hog[:1000]
	training_label_data = training_label_data[:1000]
	testing_label_data = testing_label_data[:100]
	testing_image_data_hog = testing_image_data_hog[:100]

	depth_array = [5,6,7]
	number_of_trees_array = [310,350,390]
	for depth in depth_array:
		for number_of_trees in number_of_trees_array :
			find_best_random_forest_classifier(number_of_trees,depth)
			find_best_adaboost_classifier(number_of_trees,depth)

	for key, value in best_accuracy_adaboost.iteritems():
		print "Best accuracy in adaboost with depth = "+str(key)+" is "+str(value)

	for key, value in best_accuracy_forest.iteritems():
		print "Best accuracy in forest with depth = "+str(key)+" is "+str(value)

	for key, value in best_number_of_tree_adaboost.iteritems():
		print "Best number of tree in adaboost with depth = "+str(key)+" is "+str(value)

	for key, value in best_number_of_tree_forest.iteritems():
		print "Best number of tree in forest with depth = "+str(key)+" is "+str(value)

