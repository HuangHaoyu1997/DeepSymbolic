class config:
    seed = 123
    lr = 1e-3
    rollout_episode = 10
    num_steps = 1000
    num_episodes = 100000
    ckpt_freq = 10
    
    # for CMA-ES
    sigma_init = 0.1
    pop_size = 100
