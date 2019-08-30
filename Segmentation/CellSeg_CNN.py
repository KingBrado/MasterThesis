import tensorflow as tf
import numpy as np
import cv2
import helper_functions as help 

def print_filters_as_images(numpy_weights, counter, mean, std):
    #Manually reshaping filters to have image format
    weights_shape = numpy_weights.shape;
    reshaped_filters = np.zeros((weights_shape[3],weights_shape[0],weights_shape[1],weights_shape[2]),dtype=np.float32)
    for index_i in range(weights_shape[0]):
        for index_j in range(weights_shape[1]):
            for index_k in range(weights_shape[2]):
                for index_l in range(weights_shape[3]):
                    reshaped_filters[index_l][index_i][index_j][index_k] = numpy_weights[index_i][index_j][index_k][index_l]
                                    
    reshaped_filters = reshaped_filters * std + mean
    reshaped_filters = reshaped_filters.astype(np.uint8)
    reshaped_filters_shape=np.shape(reshaped_filters)
    #Composing the images into a mosaic n_rows x 4
    n_rows = int(reshaped_filters_shape[0] // 4)
    image_of_filters = []
    for row_index in range(n_rows):
        row_of_filters = []
        for column_index in range(4):
            padded_filter = np.pad(reshaped_filters[row_index * 4 + column_index],((1,1),(1,1),(0,0)),'constant',constant_values=0)
            #Unfortunately opencv wants BGR format
            for i in range(np.shape(padded_filter)[0]):
                for j in range(np.shape(padded_filter)[1]):
                    temp = padded_filter[i][j][0]
                    padded_filter[i][j][0] = padded_filter[i][j][2]
                    padded_filter[i][j][2] = temp
            if len(row_of_filters) == 0:
                row_of_filters = padded_filter
            else:
                row_of_filters = np.append(row_of_filters, padded_filter, axis=1)
        if len(image_of_filters) == 0:
            image_of_filters = row_of_filters
        else:
            image_of_filters = np.append(image_of_filters, row_of_filters, axis=0)
    #Taking care of cases where the number of filters is not a multiple of 4
    if reshaped_filters_shape[0] % 4 != 0:
        row_of_filters = []
        for i in range(n_rows * 4,reshaped_filters_shape[0]):
            padded_filter = np.pad(reshaped_filters[i],((1,1),(1,1),(0,0)),'constant',constant_values=0)
            for i in range(np.shape(padded_filter)[0]):
                for j in range(np.shape(padded_filter)[1]):
                    temp = padded_filter[i][j][0]
                    padded_filter[i][j][0] = padded_filter[i][j][2]
                    padded_filter[i][j][2] = temp
            if len(row_of_filters) == 0:
                row_of_filters = padded_filter
            else:
                row_of_filters = np.append(row_of_filters, padded_filter, axis=1)
        row_of_filters = np.pad(row_of_filters, ((0,0),(0, np.shape(image_of_filters)[1] - np.shape(row_of_filters)[1]), (0,0)),'constant',constant_values=0)
        image_of_filters = np.append(image_of_filters, row_of_filters,axis=0)
    cv2.imwrite("OutputImages/Filters/Filters_timestep_"+str(counter)+".png",image_of_filters)

def print_labelled_images(predicted_labelling, counter, colors, name):
    labelling_shape=np.shape(predicted_labelling)
    image_to_print = np.zeros(dtype=np.uint8, shape=(labelling_shape[0],labelling_shape[1],labelling_shape[2],3))
    for i in range(labelling_shape[0]):
        for j in range(labelling_shape[1]):
            for k in range(labelling_shape[2]):
                selected_color = colors[predicted_labelling[i][j][k]]["color"]
                image_to_print[i][j][k] = [selected_color[2], selected_color[1], selected_color[0]]
        cv2.imwrite("OutputImages/PredictedLabelling/"+name+str(counter)+"_"+str(i+1)+".png",image_to_print[i])

def save_accuracy_to_file(counter, train_accuracy, test_accuracy, file_name = "Accuracy.txt"):
    if counter == 0:
        file = open("Accuracy.txt", 'w')
    else:
        file = open("Accuracy.txt", 'a')
    file.write("%d %f %f\n" %(counter, train_accuracy, test_accuracy))
    file.close()
    
def convnet_layer(input_activations, filter_size, size_in, size_out, strides, using_max_pooling = False, ksize = None, pool_strides=None, name = "conv"):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", shape=[filter_size, filter_size, size_in, size_out], initializer=tf.truncated_normal_initializer(stddev=0.2, dtype=tf.float32))
            biases = tf.get_variable("biases", shape=[size_out], initializer = tf.constant_initializer(0.1, dtype=tf.float32))
            #weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, size_in, size_out],dtype = tf.float32, stddev=0.2),  name = "weights")
            #biases = tf.Variable(tf.constant(0.1, shape=[size_out]), name = "biases")
            conv = tf.nn.conv2d(input_activations, weights, strides, padding='VALID')
            if(using_max_pooling):
                conv = tf.nn.max_pool(conv, ksize, pool_strides, padding='VALID')
            output_activations = tf.nn.relu(conv + biases)
            squared_weigths_sum = tf.reduce_sum(tf.multiply(weights,weights))
            squared_biases_sum = tf.reduce_sum(tf.multiply(biases,biases))
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("output_activations",output_activations)
            return [output_activations, tf.add(squared_weigths_sum, squared_biases_sum)]
        
def deconv_layer(input_activations, output_height, output_width, filter_size, size_in, size_out, strides, name = "deconv"):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", shape = [filter_size, filter_size, size_out, size_in], initializer=tf.truncated_normal_initializer(stddev=0.2, dtype=tf.float32))
            deconv = tf.nn.conv2d_transpose(input_activations, weights, [tf.shape(input_activations)[0], output_height, output_width, size_out], strides, padding = 'VALID')
            tf.summary.histogram("weights", weights)
            return deconv
            
#Tensorflow
def train_graph(number_of_images, input_patch_height, input_patch_width, output_patch_height, output_patch_width, number_of_classes, cnn_layers,regularisation_param, learning_rate,
          deconv_layer_is_present = False, deconvolution_layer = None, name = "graph"):
    with tf.name_scope(name):
        #Placeholders for training input image and ground truth images
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name = "input_data")
        tf.summary.image("input_crop", input_image, max_outputs=number_of_images)
        output_classes= tf.placeholder(tf.float32, shape=[None, None, None, number_of_classes], name = "expected_output")
        defined_samples_mask = tf.placeholder(tf.bool, shape = [None,None, None], name = "defined_samples")
        output_shape = tf.shape(output_classes)
        reshaped_output_classes = tf.reshape(output_classes, shape=[output_shape[0] * output_shape[1] * output_shape[2], number_of_classes])
        reshaped_defined_samples = tf.reshape(defined_samples_mask, shape = [output_shape[0] * output_shape[1] * output_shape[2]])
        valid_output_classes = tf.boolean_mask(reshaped_output_classes, reshaped_defined_samples)
        
        #Convolutions 
        totalRegTerm = tf.constant(0,dtype=tf.float32)
        conv = input_image  
        layer_index = 1
        for layer in cnn_layers:
            [conv, reg_term] = convnet_layer(conv, layer["filter size"], layer["input depth"], layer["output depth"], list(layer["strides"]), layer["max pooling"], layer["ksize"], layer["pooling strides"], "conv"+str(layer_index))
            totalRegTerm = tf.add(totalRegTerm, reg_term)
            layer_index += 1
        if deconv_layer_is_present:
            conv = deconv_layer(conv, output_patch_height, output_patch_width, deconvolution_layer["filter size"], deconvolution_layer["input depth"], deconvolution_layer["output depth"], deconvolution_layer["strides"])
        
        predicted_labelling = tf.argmax(conv,axis=3)
        logits = tf.reshape(conv, shape=[tf.shape(conv)[0] * output_patch_width * output_patch_height, number_of_classes])
        valid_logits = tf.boolean_mask(logits, reshaped_defined_samples)
        logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels =valid_output_classes, logits = valid_logits)) 
        regularisation_loss = tf.multiply(regularisation_param, totalRegTerm)
        loss = tf.add(logit_loss, regularisation_loss)
        
        matching_logits = tf.equal(tf.argmax(valid_logits, axis=1), tf.argmax(valid_output_classes,axis=1))
        accuracy = tf.reduce_mean(tf.cast(matching_logits,tf.float32))
        tf.summary.scalar("logitloss_function", logit_loss)
        tf.summary.scalar("loss_function", loss)
        tf.summary.scalar("accuracy", accuracy)
        training = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return [input_image, output_classes, defined_samples_mask, predicted_labelling, reshaped_output_classes, reshaped_defined_samples, valid_output_classes, logits, valid_logits, logit_loss, conv, accuracy, training]

def predict_graph(number_of_images, cnn_layers, output_patch_height = 0, output_patch_width = 0, deconv_layer_is_present = False, deconvolution_layer = None, name = "graph"):
    with tf.name_scope(name):
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name = "input_data")
        tf.summary.image("input_crop", input_image, max_outputs=number_of_images)
        conv = input_image  
        layer_index = 1
        for layer in cnn_layers:
            [conv, reg_term] = convnet_layer(conv, layer["filter size"], layer["input depth"], layer["output depth"], list(layer["strides"]), layer["max pooling"], layer["ksize"], layer["pooling strides"], "conv"+str(layer_index))
            layer_index += 1
        if deconv_layer_is_present:
            conv = deconv_layer(conv, output_patch_height, output_patch_width, deconvolution_layer["filter size"], deconvolution_layer["input depth"], deconvolution_layer["output depth"], deconvolution_layer["strides"])
        predicted_labelling = tf.argmax(conv,axis=3)
        
        return [input_image, predicted_labelling]