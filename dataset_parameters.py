kitti_parameters = {
        'rigidity_weight_cnn': 0.5,
        'rigidity_sigma_direction': 0.75,
        'rigidity_sigma_structure': 0.25,
        'variational_lambda_consistency': 0.0,
        'variational_lambda_1storder': 0.1,
        'variational_lambda_2ndorder': 20,
        'alignment_ransac_threshold': 1.0,
        'feature_sampling': 10,
        'occlusion_threshold': 2.0,
    }

sintel_parameters = {
    'alignment_ransac_threshold': 3.0,
    'occlusion_threshold': 1.5,
    'rigidity_weight_cnn': 0.1, # Can also be 0.01
    'rigidity_sigma_direction': 0.75,
    'rigidity_sigma_structure': 4.0, # Before: 2.0
    'variational_lambda_consistency': 0.0,
    'variational_lambda_1storder': 0.75,
    'variational_lambda_2ndorder': 2,
}
