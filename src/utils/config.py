import importlib.util


def loadconfig(filepath, name=""):
    spec = importlib.util.spec_from_file_location("configs", filepath)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.CustomConfig(name)
