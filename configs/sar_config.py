from configs import sac_config


def get_config():
    config = sac_config.get_config()


    config.model_cls = "SAREXPOLearner"


    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm=True

    config.N = 32
    config.train_N = 32
    config.sar_N = 8
    config.target_entropy = None
    config.ne_samples = 0
    config.ne_samples_train = 0
    config.adjust_target_entropy = False
    config.soft_sampling_dist_backup = False
    config.soft_sampling_dist = False
    config.soft_sampling_beta = 1.0
    config.r_action_scale = 1.0
    config.actor_drop = 0.0
    config.d_actor_drop = 0.0
    config.actor_lr = 3e-4
    config.batch_split = 1
    config.T = 10

    return config
