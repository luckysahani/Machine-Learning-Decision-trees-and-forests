#!/usr/bin/python
import nltk
import re,time,os,sys
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog
import idx2numpy

def SingleDecisionTreeClassifier(pix):
	print "\nCreating HOG Dataset from MNIST Data"
	start_time = time.time()
	training_image_data_hog = [hog(img, orientations=9, pixels_per_cell=(pix,pix), cells_per_block=(3, 3))
					for img in training_image_data]
	testing_image_data_hog = [hog(img, orientations=9, pixels_per_cell=(pix, pix), cells_per_block=(3, 3))
					for img in testing_image_data]
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
# 
	print '\n# printing Accuracy'
	print "\nTesting for Single Decision Tree Classifier with pixels per cell = ("+str(pix)+','+str(pix)+') :'
	print "-------------------------------------------------"
	print "\nSingleDecisionTreeClassifier accuracy for ("+str(pix)+','+str(pix)+") : "+ str(single_decision_tree_classifier_accuracy)

	return single_decision_tree_classifier_accuracy
 

if __name__ == '__main__':
	training_image_data = idx2numpy.convert_from_file("train-images.idx3-ubyte")
	training_label_data = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
	testing_image_data = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
	testing_label_data = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")
	print "Started"
	pixel_array = [4,5,6,7,8,9]
	sum_of_accuracy = 0.0
	for pix in pixel_array:
		sum_of_accuracy +=SingleDecisionTreeClassifier(pix)
	average_accuracy = sum_of_accuracy/len(pixel_array)
	print "\nSingle Decison Tree average accuracy : "+ str(average_accuracy)
