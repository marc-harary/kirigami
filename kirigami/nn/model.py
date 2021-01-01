import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, layer_dict):
        super(Model, self).__init__()
        for i, layer in enumerate(layer_dict):
            layer_type = layer.pop("layer_type")
            try:
                layer_class = getattr(nn, layer_type)
            except:
                raise AttributeError("Invalid layer type")
            try:
                layer_obj = layer_class(**layer) 
            except:
                raise TypeError("Layer does not have attribute")
            setattr(self, 'layer'+i, layer_obj)
    
    def forward(self, ipt):
        ret = ipt
        i = 0
        while hasattr(self, "layer"+i):
            layer = getattr(self, "layer"+i)
            ret = layer(ret)
            i += 1
        return ret  
