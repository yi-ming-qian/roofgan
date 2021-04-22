from agent.agent_gan import WGANAgant


def get_agent(config):
    if config.module == 'house':
    	return HouseAgent(config)
    elif config.module == 'lvae':
    	return  LvaeAgent(config)
    else:
        raise ValueError

