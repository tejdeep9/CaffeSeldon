from caffe2.python import workspace
import numpy as np
import base64
import json

class CaffeClassifier(object): 
    
    def __init__(self):
       
        with open('init_net.pb', 'rb') as f:
            init_net = f.read()
        with open('predict_net.pb', 'rb') as f:
            predict_net = f.read()

        self.model = workspace.Predictor(init_net, predict_net)

 
    def predict(self,X,feature_names):       

        data = X.item(0).decode('utf-8')
        elem = json.loads(data);
        data_type = elem[0]
        imagestring = elem[1]
        shape = elem[2]
        print data_type
        #print imagestring
        print shape

        encoded = imagestring.encode('utf-8')
        decoded = base64.b64decode(encoded)
        img = np.frombuffer(decoded, dtype = np.float32).reshape(shape)
        
        result = self.model.run([img])
        np.shape(result)
        print type(result)
        print np.shape(result)
        return result