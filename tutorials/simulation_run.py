import simulation as exp

import ray
    

ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    log_to_driver=False,
    num_cpus=26, 
    #num_gpus=1,
    memory=63 * 1024**3,          # 63 GB for workers
    object_store_memory=27 * 1024**3  # 27 GB for object store
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
    experiment_extra_tag = "Original",


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
    num_workers=8, #parallel rollout processes
    num_envs_per_worker=2, #environments internally (vectorized environments).
    num_gpus=1,
    num_cpus_per_worker= 1,
    rollout_fragment_length=200, #steps each environment produces before sending data back - gives one planner decision per fragment
    train_batch_size=3200,      # total number of transitions collected before one training update: 8 workers * 2 envs * 200 steps
    sgd_minibatch_size=800, #how the training batch is split during optimization:
    #gradient updates on each mini-batch, where number of minibatches per iteration: train_batch_size/sgd_minibatch_size
    num_sgd_iter=4, #How many times PPO reuses the same batch.
)


print(ray.available_resources())

results = exp.run_experiment(settings)