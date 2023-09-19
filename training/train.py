from mlproject import util
from lightning.pytorch import Trainer, seed_everything


def setup_training(config, prepare_data=False):
    
    datamodule, model = util.setup_data_and_model_from_config(config)
    if prepare_data:
        datamodule.prepare_data()

    if config['deterministic']:
        seed_everything(42, workers=True)
    trainer = Trainer(max_epochs=config['max_epochs'], 
                      val_check_interval=config['val_check_interval'], 
                      deterministic=config['deterministic'], 
                      default_root_dir=config['log_dir'])

    return datamodule, model, trainer


def main():
    """ Runs training. Call: python training/train.py runs/config.yaml"""
    
    config = util._get_config_dict()
    datamodule, model, trainer = setup_training(config)
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()