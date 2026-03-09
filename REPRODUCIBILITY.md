{
  "profile_name": "extended",
  "seeds": [
    101,
    202,
    303,
    404,
    505
  ],
  "episodes_per_experiment": 64,
  "steps_per_episode": 512,
  "mc_rollouts": 1000,
  "chunk_size": 25,
  "K_values": [
    3,
    4
  ],
  "selected_k4": true,
  "run_extended_sweeps": true,
  "R_brier_max": 0.05,
  "omega_min_factor": 0.005,
  "T_C_max": 50,
  "k_calib": 1.0,
  "sigma_dither": 0.08,
  "epsilon_control": 0.5,
  "missing_fraction_target": 0.516,
  "mode1_base_rate": 0.16,
  "observer_mode_accuracy_approx": 0.55,
  "w3_sweep_values": [
    0.05,
    0.1,
    0.2,
    0.3,
    0.5
  ]
}