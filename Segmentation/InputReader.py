import json

class InputReader:

    def __init__(self, input_file_name):
        with open(input_file_name,'r') as input_file:
            self.data = json.load(input_file)
        self.initialise_parameters()
    
    def initialise_parameters(self):
        self.learning_rate = float(self.data["learning rate"])
        self.regularisation_parameter = float(self.data["regularisation parameter"])
        self.number_of_epochs = int(self.data["number of epochs"])
        self.training_samples_path  = self.data["training samples path"]
        self.test_samples_path = self.data["test samples path"]
        self.undefined_class_color = list(self.data["undefined class color"])
        self.classes_colors = list(self.data["classes colors"])
        self.tensorboard_location = self.data["tensorboard location"]
        self.input_patch_width = int(self.data["input patch width"])
        self.input_patch_height = int(self.data["input patch height"])
        self.output_patch_width = int(self.data["output patch width"])
        self.output_patch_height = int(self.data["output patch height"])
        self.cnn_layers = list(self.data["CNN layers"])
        self.summaries_every = int(self.data["summaries printing interval"])
        if "Deconv layer" in self.data:
            self.deconv_layer_is_present = True
            self.deconv_layer = dict(self.data["Deconv layer"])
        else:
            self.deconv_layer_is_present = False
        self.use_data_rotation = self.data["use data rotation"]
        self.predicting = self.data["predicting"]