#!/usr/bin/python
import nltk
import re,time,os,sys
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog

def Random_Forest_Classifier(number_of_trees=100):
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
	# training_image_data_hog = training_image_data
	# testing_image_data_hog = testing_image_data
	# training_image_data_hog = [hog(train_image, cells_per_block=(1, 1))
	# 				for train_image in training_image_data]
	# testing_image_data_hog = [hog(test_image, cells_per_block=(1, 1))
	# 			   for test_image in testing_image_data]
	training_image_data_hog = [hog(train_image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
					for train_image in training_image_data]
	testing_image_data_hog = [hog(test_image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
				   for test_image in testing_image_data]
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to make the HOG Images"

	print '\nTraining data'
	start_time = time.time()
	# random_forest_classifier = DecisionTreeClassifier(max_features=None, max_depth=None)
	random_forest_classifier = RandomForestClassifier(n_estimators=number_of_trees)
	random_forest_classifier.fit(training_image_data_hog, training_label_data)
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to train the classifier"
	print 'Training Completed'

	print '\nTesting data '
	start_time = time.time()
	match_random_forest_classifier = 0
	unmatch_random_forest_classifier = 0
	predicted_labels = random_forest_classifier.predict(testing_image_data_hog)
	for i in range(0,len(testing_image_data_hog)):
		if( testing_label_data[i] == predicted_labels[i]):
			match_random_forest_classifier = match_random_forest_classifier + 1
		else:
			unmatch_random_forest_classifier = unmatch_random_forest_classifier + 1
	random_forest_classifier_accuracy = (float) (match_random_forest_classifier )/ (match_random_forest_classifier + unmatch_random_forest_classifier)
	# random_forest_classifier_accuracy = random_forest_classifier.score(images_test, labels_test)
	end_time = time.time() - start_time
	print "It took "+ str(end_time) + " to test the data "

	print '\nPrinting Accuracy'
	print "\nTesting for Random Forest Classifier with Number of Trees ="+str(number_of_trees)
	print "-------------------------------------------------"
	print "Random Forest accuracy : "+ str(random_forest_classifier_accuracy)

	return random_forest_classifier_accuracy
 

if __name__ == '__main__':
	number_of_trees = [10,25,50,100,500]
	sum_of_accuracy = 0.0
	for tree_count in number_of_trees:
		sum_of_accuracy +=Random_Forest_Classifier(tree_count)
	average_accuracy = sum_of_accuracy/len(number_of_trees)
	print "Random Forest average accuracy : "+ str(average_accuracy)
