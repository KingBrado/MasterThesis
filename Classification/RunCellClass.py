import tensorflow as tf
import numpy as np
import CNN_Image_Manager as CNNIM
from CellClass_CNN import *
from helper_functions import *
#Reading in all images

file = open("Accuracies.txt",'w')
number_of_epochs = 200
number_of_images_per_batch = 16
number_of_classes = 2
regularisation_param = 0.1
learning_rate = 0.00001
keep_dropout =1.0
    
image_manager = CNNIM.CNNImageManager(number_of_images_per_batch)
[images, labels] = image_manager.read_training_batch(0)

images_shape = np.shape(images)

print(images_shape, np.shape(labels))
number_of_batches = image_manager.total_number_of_training_images // number_of_images_per_batch
total_number_of_iterations = image_manager.total_number_of_training_images // number_of_images_per_batch
print("Total of %d training images" %(number_of_batches*number_of_images_per_batch))
cnn_layers = [
        { "filter size": 11, "input depth": 3, "output depth": 8, "strides": [1, 1, 1, 1], "max pooling": True, "ksize": [1,2,2,1], "pooling strides": [1,2,2,1], "using dropout": False},
        { "filter size": 9, "input depth": 8, "output depth": 16, "strides": [1, 1, 1, 1], "max pooling": True, "ksize": [1, 2, 2, 1], "pooling strides": [1, 2, 2, 1], "using dropout": False},
        { "filter size": 7, "input depth": 16, "output depth": 32, "strides": [1, 1, 1, 1], "max pooling": True, "ksize": [1,2,2,1], "pooling strides": [1,2,2,1], "using dropout": False},
        { "filter size": 7, "input depth": 32, "output depth": 64, "strides": [1, 1, 1, 1], "max pooling": True, "ksize": [1,2,2,1], "pooling strides": [1,2,2,1], "using dropout": False},
        #{ "filter size": 7, "input depth": 64, "output depth": 128, "strides": [1, 1, 1, 1], "max pooling": True, "ksize": [1,2,2,1], "pooling strides": [1,2,2,1], "using dropout": False},
        #{ "filter size": 5, "input depth": 128, "output depth": 256, "strides": [1, 1, 1, 1], "max pooling": True, "ksize": [1,2,2,1], "pooling strides": [1,2,2,1], "using dropout": False}
        ]
fc_layers = [
            { "number of neurons": 128, "using relu": True, "using dropout":True},
            { "number of neurons": 2, "using relu": False, "using dropout":False}
            ]
[input_images, desired_labels, keep_prob, accuracy, training, last_layer, intermediate_layer] = classification_graph(images_shape[0], number_of_classes, images_shape[1], images_shape[2], cnn_layers, fc_layers, regularisation_param, learning_rate)

merged_summaries = tf.summary.merge_all()
training_writer = tf.summary.FileWriter("ClassLog"+"/training_plot")
test_writer = tf.summary.FileWriter("ClassLog"+"/test_plot")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    training_writer.add_graph(sess.graph)
    training_accuracy = 0
    for epoch in range(number_of_epochs):
        print("Epoch",epoch)
        iteration = 0
        while(iteration < number_of_batches):
            [images, labels] = image_manager.read_training_batch(iteration)
            feed_dict={input_images:images, desired_labels:labels, keep_prob:keep_dropout}
            sess.run(training, feed_dict=feed_dict)
            training_accuracy += sess.run(accuracy, feed_dict=feed_dict)
            iteration+=1
            print("Iteration",iteration,"ended (Training)")
        s = sess.run(merged_summaries, feed_dict=feed_dict)
        training_writer.add_summary(s, epoch)
        training_accuracy /= iteration
        
        [images, labels] = image_manager.read_test_images()
        test_accuracy = 0
        test_iteration = 0
        for i in range(np.size(images,axis=0)-3):
            feed_dict={input_images:images[i:i+3,:,:,:], desired_labels:labels[i:i+3,:], keep_prob:1.0}
            test_accuracy += sess.run(accuracy, feed_dict=feed_dict)
            iteration +=1
            test_iteration += 1
            print("Iteration",iteration,"ended (Testing), Accuracy", test_accuracy)
        s = sess.run(merged_summaries, feed_dict=feed_dict)
        test_writer.add_summary(s, epoch)
        test_accuracy /= test_iteration
        file.write("%d    %10.8f    %10.8f\n" %(epoch,training_accuracy, test_accuracy))
        file.flush()
file.close()
