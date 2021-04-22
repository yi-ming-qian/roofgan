from config.config_gan import RoofGANConfig


def get_config(name):
    if name == 'pqnet':
        return PQNetConfig
    elif name == 'gan':
        return RoofGANConfig
    else:
        raise ValueError("Got name: {}".format(name))
