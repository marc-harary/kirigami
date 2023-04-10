from pytorch_lightning.cli import LightningCLI
import kirigami
from kirigami import KirigamiModule, DataModule


def main():
    cli = LightningCLI(
        KirigamiModule,
        DataModule,
        parser_kwargs={
            "predict": {"default_config_files": ["configs/predict.yaml"]},
            "test": {"default_config_files": ["configs/test.yaml"]},
        },
    )


if __name__ == "__main__":
    main()
