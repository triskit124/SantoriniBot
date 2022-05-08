import configparser
import os


def read_config(filepath):
    if not os.path.exists(filepath):
        raise ("No config file in path {}".format(filepath))  # complain if file doesn't exist

    config = configparser.ConfigParser()
    config.read(filepath)
    return config


def save_config(filepath, config):
    with open(filepath, 'w') as configfile:
        return config.write(configfile)
