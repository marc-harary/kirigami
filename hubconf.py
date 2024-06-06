# hubconf.py

dependencies = ['torch']

from kirigami import KirigamiModule

def kirigami(pretrained=True, checkpoint_path='weights/main.ckpt'):
    hub_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the checkpoint file
    checkpoint_path = os.path.join(hub_dir, checkpoint_path)
    model = KirigamiModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model
