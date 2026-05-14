import simulation as exp

import ray
    

ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    log_to_driver=False,
    #num_gpus=1,
    memory = 48 * 1024**3,          # 63 GB for workers
    object_store_memory=20 * 1024**3  # 27 GB for object store
)



settings = exp.ExperimentSettings(
    phase1_iters = 500,
    phase2_iters = 1500,
    phase3a_iters = 500,
    phase3b_iters = 1500,
    save_results=True,
    travel_enabled_phase3a=False,
    travel_enabled_phase3b=False,
    restrict_trade_to_region = True,

    fixed_tax_planner_id=None, #("p_top", "p_bottom"),
    fixed_tax_bracket_rates_top=(0.01, 0.095, 0.15, 0.23, 0.32, 0.395, 0.42), #lux-old
    #fixed_tax_bracket_rates_top=(0.09, 0.123, 0.18, 0.24, 0.30, 0.36, 0.405),
    fixed_tax_bracket_rates_bottom=(0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37), #US
    experiment_extra_tag="final_base",
    layout_file = "stacked_51x25_symetric_original.txt",

    travel_cost_coin_phase3a = 10,
    travel_cost_labor_phase3a = 10,

    travel_cost_coin_phase3b = 10,
    travel_cost_labor_phase3b = 10,

# same as AI economist original 
    # num_workers=15,
    # num_envs_per_worker=2,
    # num_gpus=1, 
    # num_cpus_per_worker= 1,
    # rollout_fragment_length=200,
    # train_batch_size=6000,      # 15 workers * 2 envs * 200 steps
    # sgd_minibatch_size=1500,
    # num_sgd_iter=4,    


# alternative 
    num_workers=4, #8, #parallel rollout processes
    num_envs_per_worker=2, #environments internally (vectorized environments).
    num_gpus=1,
    num_cpus_per_worker= 1,
    rollout_fragment_length=200, #steps each environment produces before sending data back - gives one planner decision per fragment
    train_batch_size=1600, #3200,      # total number of transitions collected before one training update: 8 workers * 2 envs * 200 steps
    sgd_minibatch_size=800, #how the training batch is split during optimization:
    #gradient updates on each mini-batch, where number of minibatches per iteration: train_batch_size/sgd_minibatch_size
    num_sgd_iter=4, #How many times PPO reuses the same batch.
)


print(ray.available_resources())

results = exp.run_experiment(settings)