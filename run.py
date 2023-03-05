from pytorch_lightning.cli import LightningCLI
import kirigami

def main():
    cli = LightningCLI(kirigami.KirigamiModule, kirigami.DataModule)

if __name__ == "__main__":
    main()
