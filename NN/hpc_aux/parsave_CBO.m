function parsave_CBO(filename, vstar_app, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, dt, N, alpha, lambda, gamma, learning_rate, anisotropic, sigma, particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_V_update, V0mean, V0std)

    save(filename, 'vstar_app', 'performance_tracking', 'image_size', 'NN_architecture', 'NNtype', 'architecture', 'neurons', 'd', 'epochs', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma', 'particle_reduction', 'parameter_cooling', 'batch_size_N', 'batch_size_E', 'full_or_partial_V_update', 'V0mean', 'V0std')

end

