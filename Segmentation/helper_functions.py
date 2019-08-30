import numpy as np
import tensorflow as tf

def print_numpy_array_to_file(array, dimensions = "3D", name = "filename.txt"):

    file = open(name,'w')
    if dimensions == "2D":
        array_shape = np.shape(array)
        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                file.write("%f   " %array[i,j])
            file.write("\n")
    elif dimensions == "3D":
        array_shape= np.shape(array)
        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                for k in range(array_shape[2]):
                    file.write("%f    " %array[i,j,k])
                file.write("\n")
            file.write("\n")
    elif dimensions == "4D":
        array_shape = np.shape(array)
        for i in range(array_shape[0]):
            file.write("Image number %d\n" %(i+1))
            for j in range(array_shape[1]):
                for k in range(array_shape[2]):
                    for l in range(array_shape[3]):
                        file.write("%f    " %array[i,j,k,l])
                    file.write("\n")
                file.write("\n")
    file.close()

def print_tensorflow_tensor_to_file(tensor, feed_dict = None, dimensions = "3D", name = "filename"):
    
    file = open(name,'w')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if dimensions == "1D":
            tensor_shape = sess.run(tf.shape(tensor), feed_dict=feed_dict)
            for i in range(tensor_shape[0]):
                file.write("%f   " %sess.run(tensor, feed_dict=feed_dict)[i])
            file.write("\n")
        elif dimensions == "2D":
            tensor_shape = sess.run(tf.shape(tensor), feed_dict=feed_dict)
            for i in range(tensor_shape[0]):
                for j in range(tensor_shape[1]):
                    file.write("%f   " %sess.run(tensor, feed_dict=feed_dict)[i,j])
                file.write("\n")
        elif dimensions == "3D":
            tensor_shape = sess.run(tf.shape(tensor), feed_dict=feed_dict)
            for i in range(tensor_shape[0]):
                for j in range(tensor_shape[1]):
                    for k in range(tensor_shape[2]):
                        file.write("%f    " %sess.run(tensor, feed_dict=feed_dict)[i,j,k])
                    file.write("\n")
                file.write("\n")
        elif dimensions == "4D":
            tensor_shape = sess.run(tf.shape(tensor), feed_dict=feed_dict)
            for i in range(tensor_shape[0]):
                file.write("Image number %d\n" %(i+1))
                for j in range(tensor_shape[1]):
                    for k in range(tensor_shape[2]):
                        for l in range(tensor_shape[3]):
                            file.write("%f    " %sess.run(tensor, feed_dict=feed_dict)[i,j,k,l])
                        file.write("\n")
                    file.write("\n")
    file.close()