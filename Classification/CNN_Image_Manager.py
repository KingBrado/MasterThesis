import numpy as np
import os
import cv2

class CNNImageManager():
    
    def __init__(self, images_per_batch):
        if not images_per_batch % 2 == 0:
            print("The number of images per batch must be a multiple of 2")
            exit(-1)
        self.path_to_images = "Smaller_Set/Original_Images/"
        self.number_of_images_per_class = images_per_batch // 2
        self.benign_training_files = os.listdir(self.path_to_images + "Benign/training/")
        self.malignant_training_files = os.listdir(self.path_to_images + "Malignant/training/")
        self.total_number_of_training_images = len(self.benign_training_files) * 2
        
        self.benign_test_files = os.listdir(self.path_to_images + "Benign/test/")
        self.malignant_test_files = os.listdir(self.path_to_images + "Malignant/test/")
        self.total_number_of_test_images = len(self.malignant_test_files) * 2
        
        # self.path_to_images = "CNN_Images/Original_Images/"
        # self.number_of_images_per_class = images_per_batch // 8
        # self.adenosis_files = os.listdir(self.path_to_images + "adenosis/")
        # self.ductal_carcinoma_files = os.listdir(self.path_to_images + "ductal_carcinoma/")
        # self.fibroadenoma_files = os.listdir(self.path_to_images + "fibroadenoma/")
        # self.lobular_carcinoma_files = os.listdir(self.path_to_images + "lobular_carcinoma/")
        # self.mucinous_carcinoma_files = os.listdir(self.path_to_images + "mucinous_carcinoma/")
        # self.papillary_carcinoma_files = os.listdir(self.path_to_images + "papillary_carcinoma/")
        # self.phyllodes_tumor_files = os.listdir(self.path_to_images + "phyllodes_tumor/")
        # self.tubular_adenoma_files = os.listdir(self.path_to_images + "tubular_adenoma/")
        # self.total_number_of_images = len(self.adenosis_files) * 8
        
        # self.path_to_images = "CNN_Images/Images_from_Segmentations/"
        # self.number_of_images_per_class = images_per_batch // 8
        # self.adenosis_files = os.listdir(self.path_to_images + "adenosis_labels/")
        # self.ductal_carcinoma_files = os.listdir(self.path_to_images + "ductal_carcinoma_labels/")
        # self.fibroadenoma_files = os.listdir(self.path_to_images + "fibroadenoma_labels/")
        # self.lobular_carcinoma_files = os.listdir(self.path_to_images + "lobular_carcinoma_labels/")
        # self.mucinous_carcinoma_files = os.listdir(self.path_to_images + "mucinous_carcinoma_labels/")
        # self.papillary_carcinoma_files = os.listdir(self.path_to_images + "papillary_carcinoma_labels/")
        # self.phyllodes_tumor_files = os.listdir(self.path_to_images + "phyllodes_tumor_labels/")
        # self.tubular_adenoma_files = os.listdir(self.path_to_images + "tubular_adenoma_labels/")
        # self.total_number_of_images = len(self.adenosis_files) * 8
                
        
    def prepare_image(self, image_file):
        image = cv2.imread(image_file)
        #image = cv2.medianBlur(image,3)
        #label_image = np.zeros((np.size(image,0), np.size(image, 1), 1))
        
        ##Comment lines 42 - 56 in case of original images classification
        #for w in range(np.size(image,0)):
            #for h in range(np.size(image,1)):
                #if np.array_equal(image[w,h],[229, 0, 255]):
                    #label_image[w,h] = 3
                #elif np.array_equal(image[w,h], [255, 0, 185]):
                    #label_image[w,h] = 1
                #elif np.array_equal(image[w,h], [93,0,85]):
                    #label_image[w,h] = -1
                #elif np.array_equal(image[w,h],[255,255,255]):
                    #label_image[w,h] = -3
                #elif np.array_equal(image[w,h],[229, 0, 185]):
                    #label_image[w,h] = 1
                #elif np.array_equal(image[w,h],[255, 0,255]):
                    #label_image[w,h] = 3
                #else:
                    #print("Unrecognised colour in label image (%d,%d) = [%d,%d,%d]" %(w,h,image[w,h][0],image[w,h][1],image[w,h][2]))
                    #àcv2.imwrite("ErrorImage.png",image)
                    #exit(-1)
        
        #image = label_image
        image = image.astype(np.float64)
        return image

    def read_training_batch(self, index):
        
        #Defining the 8 cancer classes
        # adenosis = 0
        # ductal_carcinoma = 1
        # fibroadenoma = 2
        # lobular_carcinoma = 3
        # mucinous_carcinoma = 4
        # papillary_carcinoma = 5
        # phyllodes_tumor = 6
        # tubular_adenoma = 7
        
        benign = 0
        malignant = 1
        
        all_images = []
        init_index = index * self.number_of_images_per_class
        end_index = np.minimum((index + 1) * self.number_of_images_per_class, len(self.benign_training_files))
        labels = np.zeros(((end_index - init_index) * 2, 2), np.float32)
       
        current_index = 0
        
        for file in self.benign_training_files[init_index:end_index]:
            image = self.prepare_image(self.path_to_images + "Benign/training/" + file)
            all_images.append(image)
            labels[current_index][benign] = 1
            current_index +=1
        
        for file in self.malignant_training_files[init_index:end_index]:
            image = self.prepare_image(self.path_to_images + "Malignant/training/" + file)
            all_images.append(image)
            labels[current_index][malignant] = 1
            current_index +=1    
            
        # for file in self.adenosis_files[init_index:end_index]:
            # image = self.prepare_image(self.path_to_images + "adenosis/" + file)
            # all_images.append(image)
            # labels[current_index][adenosis] = 1
            # current_index+=1
            
        # for file in self.ductal_carcinoma_files[init_index:end_index]:
            # image = self.prepare_image(self.path_to_images + "ductal_carcinoma/" + file)
            # all_images.append(image)
            # labels[current_index][ductal_carcinoma] = 1
            # current_index+=1
            
        # for file in self.fibroadenoma_files[init_index:end_index]:
            # image = self.prepare_image(self.path_to_images + "fibroadenoma/" + file)
            # all_images.append(image)
            # labels[current_index][fibroadenoma] = 1
            # current_index+=1
            
        # for file in self.lobular_carcinoma_files[init_index:end_index]:
            # image = self.prepare_image(self.path_to_images + "lobular_carcinoma/" + file)
            # all_images.append(image)
            # labels[current_index][lobular_carcinoma] = 1
            # current_index+=1
            
        # for file in self.mucinous_carcinoma_files[init_index:end_index]:
            # image = self.prepare_image(self.path_to_images + "mucinous_carcinoma/" + file)
            # all_images.append(image)
            # labels[current_index][mucinous_carcinoma] = 1
            # current_index+=1
            
        # for file in self.papillary_carcinoma_files[init_index:end_index]:
            # image = self.prepare_image(self.path_to_images + "papillary_carcinoma/" + file)
            # all_images.append(image)
            # labels[current_index][papillary_carcinoma] = 1
            # current_index+=1
            
        # for file in self.phyllodes_tumor_files[init_index:end_index]:
            # image = self.prepare_image(self.path_to_images + "phyllodes_tumor/" + file)
            # all_images.append(image)
            # labels[current_index][phyllodes_tumor] = 1
            # current_index+=1

        # for file in self.tubular_adenoma_files[init_index:end_index]:
            # image = self.prepare_image(self.path_to_images + "tubular_adenoma/" + file)
            # all_images.append(image)
            # labels[current_index][tubular_adenoma] = 1
            # current_index+=1
        
        all_images = np.array(all_images)
        self.mean = np.mean(all_images,axis=(0,1,2))
        self.std = np.std(all_images, axis=(0,1,2))
        all_images = (all_images - self.mean) / self.std
        #all_images = (all_images) / 255.0 * 2 - 1.0
        return [all_images, labels]
        
    def read_test_images(self):
        benign = 0
        malignant = 1
        all_images = []
        labels = np.zeros((self.total_number_of_test_images, 2), np.float32)
        current_index = 0
        for file in self.benign_test_files:
            image = self.prepare_image(self.path_to_images + "Benign/test/" + file)
            all_images.append(image)
            labels[current_index][benign] = 1
            current_index +=1
        
        for file in self.malignant_test_files:
            image = self.prepare_image(self.path_to_images + "Malignant/test/" + file)
            all_images.append(image)
            labels[current_index][malignant] = 1
            current_index +=1    
            
        all_images = np.array(all_images)
        all_images = (all_images - self.mean) / self.std
        
        return [all_images, labels]
