from caffe2.python import workspace
import numpy as np

class CaffeClassifier(object): 
    
    def __init__(self):
       
        with open('init_net.pb') as f:
            init_net = f.read()
        with open('predict_net.pb') as f:
            predict_net = f.read()

        self.model = workspace.Predictor(init_net, predict_net)

 
    def predict(self,X,feature_names):
        return self.model.run({'data': X})