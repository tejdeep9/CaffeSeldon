from caffe2.python import workspace
import numpy as np
import base64
import json

class CaffeClassifier(object): 
    
    def __init__(self):
       
        with open('/microservice/init_net.pb') as f:
            init_net = f.read()
        with open('/microservice/predict_net.pb') as f:
            predict_net = f.read()

        self.model = workspace.Predictor(init_net, predict_net)

 
    def predict(self,X,feature_names):
        data_type = X[0]
        imagestring = X[1]
        shape = X[2]
        print data_type
        print encoded
        print shape
        
        encoded = imagestring.encode('utf-8')
        decoded = base64.b64decode(encoded)
        img = np.frombuffer(decoded, dtype = array_data_type).reshape(array_shape)
        
        return self.model.run({'data': img})