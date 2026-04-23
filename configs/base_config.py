import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.hidden_dims = (256, 256, 256)
    config.discount = 0.99
    config.num_qs = 2
    config.tau = 0.005
    config.critic_layer_norm = False
    config.temp_lr = 3e-4
    config.critic_weight_decay = config_dict.placeholder(float)

    return config
