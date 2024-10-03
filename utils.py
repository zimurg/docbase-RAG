# utils.py

import os, sys, json


def get_data_path(data_dir="Data"):
    wd = os.getcwd()
    data_path = os.path.join(wd, data_dir)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The directory {data_path} does not exist.")
        sys.exit(1)

    # TODO:Considerar agregar permisos de acceso.
    return data_path


def load_config(config_path="config.json"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    except Exception as e:
        print(f"Error al cargar el archivo {config_path}: {e}")
        sys.exit(1)
