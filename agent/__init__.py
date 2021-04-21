from agent.agent_lgan import WGANAgant


def get_agent(config):
    if config.module == 'house' or config.module == 'houseplus':
    	return HouseAgent(config)
    elif config.module == 'lvae':
    	return  LvaeAgent(config)
    else:
        raise ValueError

