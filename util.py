import yaml
import importlib
from argparse import ArgumentParser
from mlproject.models import LitModel
import torch

from typing import Callable
DATA_CLASS_MODULE = "mlproject.data"
MODEL_CLASS_MODULE = "mlproject.models"

def _setup_parser_get_args():
    """ set up a parser to ingest and return the path to config.yaml """
    parser = ArgumentParser()
    parser.add_argument("config_yaml", help="Provide a config.yaml file for your run")
    return parser.parse_args()

def load_config_from_yaml(config_file: str, safe_load=False) -> dict:
    """ Extracts a dict from a yaml file
        Parameters:
            config_file (str): Path to a yaml file
        Returns:
            config (dict): dict of parameters from config file
    """
    with open(config_file) as file:
        if safe_load:
            config = yaml.safe_load(file)
        else:
            config = yaml.full_load(file)
    return config

def _get_config_dict():
    """ When a script is called with a config.yaml as an argument, extract the args dict and return it """
    args = _setup_parser_get_args()
    yaml_file = args.config_yaml
    config_dict = load_config_from_yaml(yaml_file)
    return config_dict


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def setup_optimizer(config: dict, model: torch.nn.Module) -> tuple:
    optimizer_class = getattr(torch.optim, config['optimizer'])
    optimizer = optimizer_class(model.parameters(), lr=config["lr"])

    if config['scheduler'] is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, config['scheduler'])
        scheduler = scheduler_class(optimizer=optimizer, **config['scheduler_config'])
    else:
        scheduler = None

    return optimizer, scheduler


def setup_data_and_model_from_config(config: dict) -> tuple:
    """ Given the configs dict sets up lit data module and lit model and return them """
    data_class = import_class(f"{DATA_CLASS_MODULE}.{config['data_class']}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{config['model_class']}")

    data = data_class(**config['data_class_config'])
    
    torch_model = model_class(**config['model_class_config']) 
    loss_fn = getattr(torch.nn.functional, config['loss_fn'])
    optimizer, scheduler = setup_optimizer(config, torch_model)
    lit_model = LitModel(model=torch_model, 
                         optimizer=optimizer, 
                         scheduler=scheduler, 
                         loss_fn=loss_fn)

    return data, lit_model

def main():
    pass

if __name__ == '__main__':
    main()

