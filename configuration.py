class config:
    seed = 123
    lr = 1e-3
    rollout_episode = 10
    num_steps = 1000
    num_episodes = 100000
    ckpt_freq = 10
    num_parallel = 80
    zero_coef = 1.0 # 鼓励采用尽可能多的None操作
    # for CMA-ES
    sigma_init = 3.5
    pop_size = 30
