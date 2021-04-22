from networks.networks_gan import Generator, Discriminator


def get_network(name, config):
    if name == 'G':
        net = Generator(config.n_dim)
    elif name == 'D':
        net = Discriminator()
    else:
        raise ValueError
    return net


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
