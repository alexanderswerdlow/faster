from configs import sac_config


def get_config():
    config = sac_config.get_config()


    config.model_cls = "IDQLLearner"


    config.num_qs = 2
    config.num_min_qs = 1
    config.critic_layer_norm=True
    config.expectile = 0.8

    config.N = 32
    config.train_N = 32
    config.actor_drop = 0.0
    config.d_actor_drop = 0.0
    config.actor_lr = 3e-4
    config.batch_split = 1
    config.ddim_eta = 0.0
    config.deterministic_ddim_eta0 = False
    config.T = 10

    return config
