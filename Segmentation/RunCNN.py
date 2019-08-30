import tensorflow as tf
from CellSeg_CNN import *
import numpy as np
import DataReader

#Reading the images
data_reader = DataReader.DataReader()
input_reader = data_reader.input_reader

training_images = data_reader.training_images
if(input_reader.use_data_rotation):
    rotated_images = data_reader.pi_half_rotated_images
number_of_training_images = np.size(training_images,axis=0)
image_height = np.size(training_images, axis=1)
image_width = np.size(training_images, axis=2)

test_images = data_reader.test_images
number_of_test_images = np.size(test_images,axis=0)
#Reading the ground truth classes
[training_classes, training_defined_samples] = data_reader.training_classes
if(input_reader.use_data_rotation):
    [rotated_classes, rotated_defined_mask] = data_reader.pi_half_rotated_classes_and_masks
[test_classes, test_defined_samples] = data_reader.test_classes

#Reading parameters
learning_rate = input_reader.learning_rate
regularisation_param = tf.constant(input_reader.regularisation_parameter)
n_epochs = input_reader.number_of_epochs
tensorboard_file_location = input_reader.tensorboard_location
input_patch_width = input_reader.input_patch_width
input_patch_height = input_reader.input_patch_height

computed_input_patch_width = data_reader.computed_input_patch_width
computed_input_patch_height = data_reader.computed_input_patch_height
left_padding = data_reader.left_padding
right_padding = data_reader.right_padding
upper_padding = data_reader.upper_padding
lower_padding = data_reader.lower_padding

output_patch_width = input_reader.output_patch_width
output_patch_height = input_reader.output_patch_height
number_of_classes = len(input_reader.classes_colors)
cnn_layers = input_reader.cnn_layers
max_output = np.maximum(number_of_training_images, number_of_test_images)

if input_reader.deconv_layer_is_present:
    [input_image, output_classes, defined_samples_mask, predicted_labelling, reshaped_output_classes, reshaped_defined_samples, valid_output_classes, logits, valid_logits, logit_loss, last_conv_output, accuracy, training] = train_graph(max_output, computed_input_patch_height, computed_input_patch_width,
                                                                                                        output_patch_height, output_patch_width, number_of_classes,
                                                                                                        cnn_layers,regularisation_param, learning_rate, 
                                                                                                        input_reader.deconv_layer_is_present, input_reader.deconv_layer)
else:                                                                                                        
    [input_image, output_classes, defined_samples_mask, predicted_labelling, reshaped_output_classes, reshaped_defined_samples, valid_output_classes, logits, valid_logits, logit_loss, last_conv_output, accuracy, training] = train_graph(max_output, computed_input_patch_height, computed_input_patch_width, 
                                                                                                        output_patch_height, output_patch_width, number_of_classes,
                                                                                                        cnn_layers,regularisation_param, learning_rate)

merged_summaries = tf.summary.merge_all()
training_writer = tf.summary.FileWriter(tensorboard_file_location+"/training_plot")
test_writer = tf.summary.FileWriter(tensorboard_file_location+ "/test_plot")

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    training_writer.add_graph(sess.graph)
    test_writer.add_graph(sess.graph)
    for epoch in range(n_epochs):
        train_accuracy = 0 
        test_accuracy = 0
        accuracy_counter = 0
        num_samples = 0
        n_width_patches = (image_width - left_padding - right_padding) // input_patch_width
        n_height_patches = (image_height - upper_padding - lower_padding) // input_patch_height
        for i in range(n_height_patches):
            for j in range(n_width_patches):
                counter = epoch * n_height_patches * n_width_patches + i * n_width_patches + j

                image_crop = training_images[0:number_of_training_images, input_patch_height * i: input_patch_height * i + computed_input_patch_height, input_patch_width * j:input_patch_width * j + computed_input_patch_width, :]
                labels_crop = training_classes[0:number_of_training_images, output_patch_height * i: output_patch_height*i + output_patch_height, output_patch_width * j: output_patch_width * j +output_patch_width,:]
                mask_crop = training_defined_samples[0:number_of_training_images, output_patch_height * i: output_patch_height*i + output_patch_height, output_patch_width * j: output_patch_width * j +output_patch_width]
                feed_dict={input_image: image_crop, output_classes: labels_crop, defined_samples_mask: mask_crop}
                nsamp = np.sum(mask_crop)
                num_samples += nsamp
                if (nsamp==0):
                    print("\t\t****")

                if (nsamp!=0):              
                    [local_train_accuracy, dummy] = sess.run([accuracy, training], feed_dict=feed_dict)
                    train_accuracy += local_train_accuracy
                    accuracy_counter += 1
                if(counter % input_reader.summaries_every == 0):
                    s = sess.run(merged_summaries, feed_dict=feed_dict)
                    training_writer.add_summary(s, counter)
                    
                    #Printing predicted images
                    predicted_image = sess.run(predicted_labelling, feed_dict=feed_dict)
                    print_labelled_images(predicted_image, counter, input_reader.classes_colors,'train')
                    #Printing filters in numpy format
                    numpy_weights_1 = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))[0]
                    print_filters_as_images(numpy_weights_1, counter, data_reader.mean, data_reader.std)
        
        if(input_reader.use_data_rotation):
            n_width_patches = (image_width - left_padding - right_padding) // input_patch_width
            n_height_patches = (image_height - upper_padding - lower_padding) // input_patch_height
            for i in range(n_width_patches):
                for j in range(n_height_patches):
                    counter = epoch * n_height_patches * n_width_patches + i * n_width_patches + j

                    image_crop = rotated_images[0:number_of_training_images, input_patch_width * i: input_patch_width * i + computed_input_patch_width, input_patch_height * j:input_patch_height * j + computed_input_patch_height, :]
                    labels_crop = rotated_classes[0:number_of_training_images, output_patch_width * i: output_patch_width*i + output_patch_width, output_patch_height * j: output_patch_height * j +output_patch_height,:]
                    mask_crop = rotated_defined_mask[0:number_of_training_images, output_patch_width * i: output_patch_width*i + output_patch_width, output_patch_height * j: output_patch_height * j +output_patch_height]
                    feed_dict={input_image: image_crop, output_classes: labels_crop, defined_samples_mask: mask_crop}
                    nsamp = np.sum(mask_crop)
                    num_samples += nsamp
                    if (nsamp==0):
                        print("\t\t****")

                    if (nsamp!=0):              
                        [local_train_accuracy,dummy] = sess.run([accuracy, training], feed_dict=feed_dict)
                        train_accuracy += local_train_accuracy
                        accuracy_counter += 1
        train_accuracy /= accuracy_counter
        accuracy_counter = 0
        #Testing      
        for i in range(n_height_patches):
            for j in range(n_width_patches):
                counter = epoch * n_height_patches * n_width_patches + i * n_width_patches + j
                if counter % input_reader.summaries_every == 0:
                    image_crop = test_images[:, input_patch_height * i: input_patch_height * i + computed_input_patch_height, input_patch_width * j:input_patch_width * j + computed_input_patch_width, :]
                    labels_crop = test_classes[:, output_patch_height * i: output_patch_height*i + output_patch_height, output_patch_width * j: output_patch_width * j +output_patch_width,:]
                    mask_crop = test_defined_samples[:, output_patch_height * i: output_patch_height*i + output_patch_height, output_patch_width * j: output_patch_width * j +output_patch_width]
                    feed_dict={input_image: image_crop, output_classes: labels_crop, defined_samples_mask: mask_crop}
                    s = sess.run(merged_summaries, feed_dict=feed_dict)
                    test_writer.add_summary(s,counter)
                    [predicted_image, local_test_accuracy] = sess.run([predicted_labelling, accuracy], feed_dict=feed_dict)
                    test_accuracy += local_test_accuracy
                    accuracy_counter += 1
                    print_labelled_images(predicted_image, counter, input_reader.classes_colors,'test')
        if(accuracy_counter != 0):
            test_accuracy /= accuracy_counter
            save_accuracy_to_file(epoch, train_accuracy, test_accuracy)
        print("Epoch %d finished" %(epoch))
    
    saver.save(sess, "/tmp/model_params.ckpt")
    print("\n\nTotal number of samples = %d" %(num_samples))

