# hubconf.py

dependencies = ['torch']

from kirigami import KirigamiModule

def kirigami(pretrained=True, checkpoint_path='weights/main.ckpt'):
    model = KirigamiModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model
