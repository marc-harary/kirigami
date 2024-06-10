# hubconf.py

dependencies = ['torch', 'pytorch_lightning', 'torchmetrics']

import os
from kirigami import KirigamiModule

def kirigami(pretrained=True, checkpoint_path='weights/main.ckpt'):
    hub_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(hub_dir, checkpoint_path)
    model = KirigamiModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model
