class config:
    seed = 123
    lr = 1e-3
    rollout_episode = 5
    num_steps = 200
    num_episodes = 100000
    ckpt_freq = 2
    num_parallel = 40
    zero_coef = 1.0 # 鼓励采用尽可能多的None操作
    std_coef = 0.0
    num_mat = 4
    # for CMA-ES
    sigma_init = 1.5
    pop_size = 30
