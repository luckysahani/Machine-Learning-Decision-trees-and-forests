#!/usr/bin/python
import nltk
import re,time,os,sys
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog
import idx2numpy

def SingleDecisionTreeClassifier():
	print "Creating Dataset from MNIST Data"
	start_time = time.time()
	# training_image_data, training_label_data = loadlocal_mnist(
	# 	images_path=os.getcwd()+'/train-images.idx3-ubyte', 
	# 	labels_path=os.getcwd()+'/train-labels.idx1-ubyte')
	# testing_image_data, testing_label_data = loadlocal_mnist(
	# 	images_path=os.getcwd()+'/t10k-images.idx3-ubyte', 
	# 	labels_path=os.getcwd()+'/t10k-labels.idx1-ubyte')
	training_image_data = idx2numpy.convert_from_file("train-images.idx3-ubyte")
	training_image_data_hog = [hog(img)
					for img in training_image_data]
	training_label_data = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
	testing_image_data = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
	testing_image_data_hog = [hog(img)
					for img in testing_image_data]
	testing_label_data = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to make the HOG Images"

	print '\nTraining data'
	start_time = time.time()
	single_decision_tree_classifier = DecisionTreeClassifier()
	single_decision_tree_classifier.fit(training_image_data_hog, training_label_data)
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to train the classifier"
	print 'Training Completed'

	print '\nTesting data '
	start_time = time.time()
	single_decision_tree_classifier_accuracy = single_decision_tree_classifier.score(testing_image_data_hog, testing_label_data)
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to test the data "

	print '\nPrinting Accuracy'
	print "\nTesting for Single Decision Tree Classifier :"
	print "-------------------------------------------------"
	print "SingleDecisionTreeClassifier accuracy : "+ str(single_decision_tree_classifier_accuracy)

	return single_decision_tree_classifier_accuracy
 

if __name__ == '__main__':
	SingleDecisionTreeClassifier()
