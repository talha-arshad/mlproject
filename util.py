import yaml
import importlib
from argparse import ArgumentParser

DATA_CLASS_MODULE = "mlproject.data"
MODEL_CLASS_MODULE = "mlproject.models"

def _setup_parser_get_args():
    """ set up a parser to ingest and return the path to config.yaml """
    parser = ArgumentParser()
    parser.add_argument("config_yaml", help="Provide a config.yaml file for your run")
    return parser.parse_args()

def load_config_from_yaml(config_file: str) -> dict:
    """ Extracts a dict from a yaml file
        Parameters:
            config_file (str): Path to a yaml file
        Returns:
            config (dict): dict of parameters from config file
    """
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

def _get_config():
    """ When a script is called with a config.yaml as an argument, extract the args dict and return it """
    args = _setup_parser_get_args()
    yaml_file = args.config_yaml
    config = load_config_from_yaml(yaml_file)
    return config


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def setup_data_and_model_from_args(model_class: str, data_class: str, model_class_config: dict, data_class_config: dict) -> tuple:
    """ Given data and model class and configs set up and return them """
    data_class = import_class(f"{DATA_CLASS_MODULE}.{data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{model_class}")

    data = data_class(**data_class_config)
    model = model_class(**model_class_config)

    return data, model

def main():
    pass

if __name__ == '__main__':
    main()

