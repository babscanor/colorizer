import pretrainedmodels as ptm 
import torch
from torch.nn import Softmax

model_name = 'inceptionresnetv2'
model = ptm.__dict__[model_name](num_classes=1000, pretrained='imagenet')

class Fusion(object):
    def __init__(self, batch_size):
        self.model = model
        self.batch_size = batch_size
    
    def predict(self,t):
        return self.model.forward(t)

    def fusion(self,t):
        t = self.predict(t)
        t = t.reshape(self.batch_size,1000,1)
        t = t.repeat(1,1,32*32)
        t = t.reshape(self.batch_size, 1000,32,32)
        return t

