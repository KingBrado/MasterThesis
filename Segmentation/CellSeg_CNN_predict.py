import tensorflow as tf
from CellSeg_CNN import *
import numpy as np
import DataReader

data_reader = DataReader.DataReader()
input_reader = data_reader.input_reader

predict_images = data_reader.test_images
number_of_images = np.size(predict_images,axis=0)
print(number_of_images)

cnn_layers = input_reader.cnn_layers

if input_reader.deconv_layer_is_present:
    [input_image, predicted_labelling] = predict_graph(3, cnn_layers, input_reader.output_patch_height, input_reader.output_patch_width, input_reader.deconv_layer_is_present, input_reader.deconv_layer)
else:
    [input_image, predicted_labelling] = predict_graph(3, cnn_layers)

saver = tf.train.Saver()
    
with tf.Session() as sess:
    saver.restore(sess, "/tmp/model_params.ckpt")
    j = np.minimum(10, number_of_images)
    while(j < number_of_images):
        print("Starting Batch number %d" % (j / 10))
        predicted_image = sess.run(predicted_labelling, feed_dict={input_image: predict_images[j-10:j,:,:,:]})
        print_labelled_images(predicted_image, j - 10, input_reader.classes_colors, 'predict')
        print("Batch number %d done" % (j / 10))
        j += 10
    predicted_image = sess.run(predicted_labelling, feed_dict={input_image: predict_images[j-10:number_of_images,:,:,:]})
    print_labelled_images(predicted_image, j - 10, input_reader.classes_colors, 'predict')