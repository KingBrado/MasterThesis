import os
import numpy as np
import InputReader
import helper_functions as help
import cv2
class DataReader:
    
    def __init__(self):
        try:
            self.input_reader = InputReader.InputReader("Input.json")
        except:
            raise IOError("Could not open Input.json")
        

        training_files = os.listdir(self.input_reader.training_samples_path)
        test_files = os.listdir(self.input_reader.test_samples_path)
        training_pair_files =  []
        test_pair_files = []
        for file in training_files:
            if 'labeling' in file:
                pair = [file[:-13]+'.png' , file]
                training_pair_files.append(pair)
        if not self.input_reader.predicting:
            for file in test_files:
                if 'labeling' in file:
                    pair = [file[:-13]+'.png' , file]
                    test_pair_files.append(pair)
        else:
            for file in test_files:
                pair = [file, 'Dummy']
                test_pair_files.append(pair)
        self.calculate_padding()
        self.training_images = self.stack_training_images(training_pair_files)
        self.training_classes = self.stack_training_classes(training_pair_files)
        if(self.input_reader.use_data_rotation): 
            self.image_augmentation()
        self.test_images = self.stack_test_images(test_pair_files)
        if not self.input_reader.predicting:
            self.test_classes = self.stack_test_classes(test_pair_files)
    
    def prepare_image(self,image_file):
        image = cv2.imread(image_file)
        image = image.astype(np.float64)
        image = np.expand_dims(image,0)
        return image

    def prepare_classes(self, ground_truth):
        image = cv2.imread(ground_truth)
        classes = np.zeros((np.size(image,0),np.size(image,1),len(self.input_reader.classes_colors)))
        defined_pixels = np.full((np.size(image,0),np.size(image,1)),True)
        undefined_class_color = np.array(self.input_reader.undefined_class_color)
        #OpenCV reads images in BGR mode, but we use RGB
        for i in range(np.size(image,0)):
            for j in range(np.size(image,1)):
                compare = image[i,j,:] == [undefined_class_color[2], undefined_class_color[1], undefined_class_color[0]]
                if(compare[0] & compare[1] & compare[2]):
                    defined_pixels[i,j] = False
                else:
                    for color in self.input_reader.classes_colors:
                        color_array = np.array(color["color"])
                        compare = image[i,j,:] == [color_array[2], color_array[1], color_array[0]]
                        if(compare[0] & compare[1] & compare[2]):
                            label = color["label"]
                            classes[i,j,label - 1] = 1
                            break
        return [classes.astype(np.float64), defined_pixels]
    
    def calculate_padding(self):
        self.computed_input_patch_height = self.input_reader.output_patch_height
        self.computed_input_patch_width = self.input_reader.output_patch_width
        if self.input_reader.deconv_layer_is_present:
            self.computed_input_patch_width = int((self.computed_input_patch_width - self.input_reader.deconv_layer["filter size"]) / self.input_reader.deconv_layer["strides"][2] + 1)
            self.computed_input_patch_height = int((self.computed_input_patch_height - self.input_reader.deconv_layer["filter size"]) / self.input_reader.deconv_layer["strides"][1] + 1)
        for cnn_layer in reversed(self.input_reader.cnn_layers):
            if cnn_layer["max pooling"]:
                self.computed_input_patch_width = (self.computed_input_patch_width - 1)*cnn_layer["pooling strides"][2] + cnn_layer["ksize"][2]
                self.computed_input_patch_height = (self.computed_input_patch_height - 1)*cnn_layer["pooling strides"][1] + cnn_layer["ksize"][1]
            self.computed_input_patch_width = (self.computed_input_patch_width - 1)*cnn_layer["strides"][2] + cnn_layer["filter size"]
            self.computed_input_patch_height = (self.computed_input_patch_height - 1)*cnn_layer["strides"][1] + cnn_layer["filter size"]
            
    
    def stack_training_images(self,training_pair_files):
        images = self.prepare_image(self.input_reader.training_samples_path + training_pair_files[0][0])
        for pair in training_pair_files[1:]:
            images = np.append(images, self.prepare_image(self.input_reader.training_samples_path + pair[0]), axis=0)
        self.mean = np.mean(images, axis=(0,1,2))
        self.std = np.std(images, axis=(0,1,2))
        images = (images - self.mean) / self.std
        print("Original Image size (shape):", np.shape(images))
        self.left_padding = int((self.computed_input_patch_width - self.input_reader.input_patch_width) // 2)
        self.right_padding = int((self.computed_input_patch_width - self.input_reader.input_patch_width - self.left_padding))
        self.upper_padding = int((self.computed_input_patch_height - self.input_reader.input_patch_height) // 2)
        self.lower_padding = int((self.computed_input_patch_height - self.input_reader.input_patch_height - self.upper_padding))
        images = np.pad(images, ((0,0), (self.upper_padding, self.lower_padding), (self.left_padding, self.right_padding), (0,0)), 'constant', constant_values=0)
        print("Size after padding:", np.shape(images))
        return images

    def stack_training_classes(self, training_pair_files):
        [classes, defined_class] = self.prepare_classes(self.input_reader.training_samples_path + training_pair_files[0][1])
        classes = np.expand_dims(classes,axis=0)
        defined_class = np.expand_dims(defined_class,axis=0)
        for pair in training_pair_files[1:]:
            [ground_truth, black_truth] = self.prepare_classes(self.input_reader.training_samples_path + pair[1])
            ground_truth = np.expand_dims(ground_truth, axis=0)
            black_truth = np.expand_dims(black_truth, axis=0)
            classes = np.append(classes,ground_truth,axis=0)
            defined_class = np.append(defined_class,black_truth,axis=0)
        return [classes,defined_class]

    def image_augmentation(self):
        pi_rotated_images = np.rot90(self.training_images, 2, (1,2))
        self.training_images = np.append(self.training_images, pi_rotated_images, axis=0)
        self.pi_half_rotated_images = np.rot90(self.training_images, 1, (1,2))
        
        pi_rotated_classes = np.rot90(self.training_classes[0], 2, (1,2))
        self.training_classes[0] = np.append(self.training_classes[0], pi_rotated_classes, axis=0)
        pi_half_rotated_classes = np.rot90(self.training_classes[0], 1, (1,2))
        
        pi_rotated_defined_mask = np.rot90(self.training_classes[1], 2, (1,2))
        self.training_classes[1] = np.append(self.training_classes[1], pi_rotated_defined_mask, axis=0)
        pi_half_rotated_defined_mask = np.rot90(self.training_classes[1], 1, (1,2))
        self.pi_half_rotated_classes_and_masks = [pi_half_rotated_classes, pi_half_rotated_defined_mask]
        
    def stack_test_images(self,test_pair_files):
        images = self.prepare_image(self.input_reader.test_samples_path + test_pair_files[0][0])
        for pair in test_pair_files[1:]:
            images = np.append(images, self.prepare_image(self.input_reader.test_samples_path + pair[0]), axis=0)
        images = (images - self.mean) / self.std
        images = np.pad(images, ((0,0), (self.upper_padding, self.lower_padding), (self.left_padding, self.right_padding), (0,0)), 'constant', constant_values=0)
        return images

    def stack_test_classes(self, test_pair_files):
        [classes, defined_class] = self.prepare_classes(self.input_reader.test_samples_path + test_pair_files[0][1])
        classes = np.expand_dims(classes,axis=0)
        defined_class = np.expand_dims(defined_class,axis=0)
        for pair in test_pair_files[1:]:
            [ground_truth, black_truth] = self.prepare_classes(self.input_reader.test_samples_path + pair[1])
            ground_truth = np.expand_dims(ground_truth, axis=0)
            black_truth = np.expand_dims(black_truth, axis=0)
            classes = np.append(classes,ground_truth,axis=0)
            defined_class = np.append(defined_class,black_truth,axis=0)
        return [classes,defined_class]

