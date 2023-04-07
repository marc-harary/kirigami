from pytorch_lightning.cli import LightningCLI
import kirigami
from kirigami import KirigamiModule, DataModule


def main():
    cli = LightningCLI(KirigamiModule, DataModule, save_config_callback=None)


if __name__ == "__main__":
    main()
