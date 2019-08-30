import tensorflow as tf

def fully_connected_layer(input_activations, number_of_weights_per_neuron, number_of_neurons, using_relu, using_droput = False, keep_prob = 1.0, name = "fully connected layer"):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            input_shape = input_activations.get_shape()
            weights = tf.get_variable("weights", shape=[number_of_weights_per_neuron, number_of_neurons], initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
            biases = tf.get_variable("biases", shape=[number_of_neurons], initializer = tf.constant_initializer(0.1, dtype=tf.float32))
            h = tf.matmul(input_activations, weights) + biases
            output = tf.nn.relu(h) if using_relu else h
            output = tf.nn.dropout(output, keep_prob) if using_droput else output
            squared_weigths_sum = tf.reduce_sum(tf.multiply(weights,weights))
            squared_biases_sum = tf.reduce_sum(tf.multiply(biases,biases))
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("output_activations",output)
            return [output, tf.add(squared_weigths_sum, squared_biases_sum)]
            
def convnet_layer(input_activations, filter_size, size_in, size_out, strides, using_max_pooling = False, ksize = None, pool_strides=None, using_droput = False, keep_prob = 1.0, name = "conv"):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", shape=[filter_size, filter_size, size_in, size_out], initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
            biases = tf.get_variable("biases", shape=[size_out], initializer = tf.constant_initializer(0.1, dtype=tf.float32))
            #weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, size_in, size_out],dtype = tf.float32, stddev=0.2),  name = "weights")
            #biases = tf.Variable(tf.constant(0.1, shape=[size_out]), name = "biases")
            conv = tf.nn.conv2d(input_activations, weights, strides, padding='VALID')
            if(using_max_pooling):
                conv = tf.nn.max_pool(conv, ksize, pool_strides, padding='VALID')
            output_activations = tf.nn.relu(conv + biases)
            output_activations = tf.nn.dropout(output_activations, keep_prob) if using_droput else output_activations
            squared_weigths_sum = tf.reduce_sum(tf.multiply(weights,weights))
            squared_biases_sum = tf.reduce_sum(tf.multiply(biases,biases))
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("output_activations",output_activations)
            return [output_activations, tf.add(squared_weigths_sum, squared_biases_sum)]
            
def classification_graph(number_of_images, number_of_classes, image_height, image_width, cnn_layers, fc_layers, regularisation_param, learning_rate, name="classification_graph"):
    input_images = tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3], name = "input_data")
    tf.summary.image("input_crop", input_images, max_outputs=number_of_images)
    desired_labels = tf.placeholder(tf.float32, shape=[None, number_of_classes], name = "desired_labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    totalRegTerm = tf.constant(0, dtype=tf.float32)
    conv = input_images  
    layer_index = 1
    for layer in cnn_layers:
        [conv, reg_term] = convnet_layer(conv, layer["filter size"], layer["input depth"], layer["output depth"], list(layer["strides"]), layer["max pooling"], layer["ksize"], layer["pooling strides"], 
                                         layer["using dropout"], keep_prob, "conv"+str(layer_index))
        totalRegTerm = tf.add(totalRegTerm, reg_term)
        layer_index += 1
    
    conv_after_cnn = conv    
    layer_index = 1
    number_of_weights_per_neuron = conv.get_shape()[1] * conv.get_shape()[2] * conv.get_shape()[3]
    conv = tf.reshape(conv, shape=[tf.shape(conv)[0], tf.shape(conv)[1] * tf.shape(conv)[2] * tf.shape(conv)[3]])

    for layer in fc_layers:
        [conv, reg_term] = fully_connected_layer(conv, number_of_weights_per_neuron, layer["number of neurons"], layer["using relu"], layer["using dropout"], keep_prob, name = "fully_connected_layer" + str(layer_index))
        totalRegTerm = tf.add(totalRegTerm, reg_term)
        layer_index += 1
        number_of_weights_per_neuron = conv.get_shape()[1]
        
    logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = desired_labels, logits = conv)) 
    regularisation_loss = tf.multiply(regularisation_param, totalRegTerm)
    loss = tf.add(logit_loss, regularisation_loss)
    matching_logits = tf.equal(tf.argmax(conv, axis=1), tf.argmax(desired_labels,axis=1))
    accuracy = tf.reduce_mean(tf.cast(matching_logits,tf.float32))
    tf.summary.scalar("logitloss_function", logit_loss)
    tf.summary.scalar("loss_function", loss)
    tf.summary.scalar("accuracy", accuracy)
    training = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return [input_images, desired_labels, keep_prob, accuracy, training, conv, conv_after_cnn]
