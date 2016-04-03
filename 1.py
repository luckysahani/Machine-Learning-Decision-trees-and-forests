#!/usr/bin/python
import nltk
# nltk.download('punkt') # for tokens
# nltk.download("stopwords") # for stopwords
# nltk.download('wordnet')
import re,time,os,sys
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# word_lemmatizer = WordNetLemmatizer()
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import loadlocal_mnist
# import os
# import time
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog

def SingleDecisionTreeClassifier():
	print "Creating Dataset from MNIST Data"
	start_time = time.time()
	training_image_data, training_label_data = loadlocal_mnist(
		images_path=os.getcwd()+'/train-images.idx3-ubyte', 
		labels_path=os.getcwd()+'/train-labels.idx1-ubyte')
	testing_image_data, testing_label_data = loadlocal_mnist(
		images_path=os.getcwd()+'/t10k-images.idx3-ubyte', 
		labels_path=os.getcwd()+'/t10k-labels.idx1-ubyte')
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to make the dataset"

	print "Creating Image Dataset using Histogram of Gradients"
	start_time = time.time()
	training_image_data_hog = [hog(train_image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
					for train_image in training_image_data]
	testing_image_data_hog = [hog(test_image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
				   for test_image in testing_image_data]
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to make the HOG Images"

	print '\nTraining data'
	start_time = time.time()
	# single_decision_tree_classifier = DecisionTreeClassifier(max_features=None, max_depth=None)
	single_decision_tree_classifier = DecisionTreeClassifier()
	single_decision_tree_classifier.fit(training_image_data_hog, training_label_data)
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to train the classifier"
	print 'Training Completed'

	print '\nTesting data '
	start_time = time.time()
	match_single_decision_tree_classifier = 0
	unmatch_single_decision_tree_classifier = 0
	predicted_labels = single_decision_tree_classifier.predict(testing_image_data_hog)
	for i in range(0,len(testing_image_data_hog)):
		if( testing_label_data[i] == predicted_labels[i]):
			match_single_decision_tree_classifier = match_single_decision_tree_classifier + 1
		else:
			unmatch_single_decision_tree_classifier = unmatch_single_decision_tree_classifier + 1
	single_decision_tree_classifier_accuracy = (float) (match_single_decision_tree_classifier )/ (match_single_decision_tree_classifier + unmatch_single_decision_tree_classifier)
	# single_decision_tree_classifier_accuracy = single_decision_tree_classifier.score(images_test, labels_test)
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to test the data "

	print '\nPrinting Accuracy'
	print "\nTesting for Single Decision Tree Classifier :"
	print "-------------------------------------------------"
	print "SingleDecisionTreeClassifier accuracy : "+ str(single_decision_tree_classifier_accuracy)

	return single_decision_tree_classifier_accuracy
 

if __name__ == '__main__':
	SingleDecisionTreeClassifier()
	# sum_of_accuracy_for_a_metric = 0.0
	# metrics = ['euclidean', 'manhattan', 'minkowski']
	# for metric in metrics:
	#     sum_of_accuracy_for_a_metric = 0.0
	#     for k in range(1,5):
	#         sum_of_accuracy_for_a_metric += single_decision_tree_classifier(k, metric)
	#     print "\nMean Accuracy for metric : "+str(metric)+" is "+str(sum_of_accuracy_for_a_metric/4)
