import simulation as exp

import ray
ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False)

settings = exp.ExperimentSettings(
    phase1_iters = 5,
    phase2_iters = 12,
    phase3a_iters = 5,
    phase3b_iters = 12,
    save_results=True,
    travel_enabled_phase3a=False,
    travel_enabled_phase3b=False,
    restrict_trade_to_region = True,
    experiment_extra_tag = "Original_with_travel",
)

results = exp.run_experiment(settings)