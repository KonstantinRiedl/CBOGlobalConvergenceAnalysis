function parsave_CBOmemory(filename, xstar_app, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, dt, N, memory, kappa, alpha, beta, lambda1, lambda2, gamma, learning_rate, anisotropic1, sigma1, anisotropic2, sigma2, particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_XY_update, X0mean, X0std)

    save(filename, 'xstar_app', 'performance_tracking', 'image_size', 'NN_architecture', 'NNtype', 'architecture', 'neurons', 'd', 'epochs', 'dt', 'N', 'memory', 'kappa', 'alpha', 'beta', 'lambda1', 'lambda2', 'gamma', 'learning_rate', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'particle_reduction', 'parameter_cooling', 'batch_size_N', 'batch_size_E', 'full_or_partial_XY_update', 'X0mean', 'X0std')

end

