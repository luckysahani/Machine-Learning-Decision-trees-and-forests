#!/usr/bin/python
import nltk
import re,time,os,sys
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from skimage.feature import hog
import idx2numpy

def Random_Forest_Classifier(number_of_trees=100):
	# print '\nTraining data'
	start_time = time.time()
	# random_forest_classifier = DecisionTreeClassifier(max_features=None, max_depth=None)
	random_forest_classifier = RandomForestClassifier(n_estimators=number_of_trees)
	random_forest_classifier.fit(training_image_data_hog, training_label_data)
	end_time = time.time() - start_time
	# print "It took "+ str(end_time) + " to train the classifier"
	# print 'Training Completed'

	# print '\nTesting data '
	start_time = time.time()
	random_forest_classifier_accuracy = random_forest_classifier.score(testing_image_data_hog, testing_label_data)
	end_time = time.time() - start_time
	# print "It took "+ str(end_time) + " to test the data "

	# print '\n# printing Accuracy'
	# print "\nTesting for Random Forest Classifier with Number of Trees ="+str(number_of_trees)
	# print "-------------------------------------------------"
	print "\nRandom Forest accuracy with number of trees = "+str(number_of_trees)+ " is "+ str(random_forest_classifier_accuracy)

	return random_forest_classifier_accuracy
 

if __name__ == '__main__':
	train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
	training_image_data_hog = [hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
						for img in train_images]
	training_label_data = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
	test_images = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
	testing_image_data_hog = [hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
					   for img in test_images]
	testing_label_data = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")
	
	number_of_trees = [1, 3, 5, 7,  10, 15, 20, 25, 50, 100, 250,500]
	sum_of_accuracy = 0.0
	for tree_count in number_of_trees:
		sum_of_accuracy +=Random_Forest_Classifier(tree_count)
	average_accuracy = sum_of_accuracy/len(number_of_trees)
	print "\nRandom Forest average accuracy : "+ str(average_accuracy)
