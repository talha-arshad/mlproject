from mlproject import util

def main():
    """ Runs training. Call: python training/train.py runs/config.yaml"""
    config = util._get_config()
    print(config)

    datamodule, model = util.setup_data_and_model_from_args(config['model_class'], config['data_class'], config['model_class_config'], config['data_class_config']) 
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    print('train_dataset: \n', datamodule.train_dataset)
    print('\nModel:\n', model)

if __name__ == '__main__':
    main()