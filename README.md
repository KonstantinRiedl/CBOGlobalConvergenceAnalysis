# EnergyBasedCBOAnalysis
Numerical illustration of a novel analysis framework for consensus-based optimization (CBO) and numerical experiments demonstrating the practicability of the method.

CBO is a multi-agent metaheuristic derivative-free optimization method capable of globally minimizing nonconvex and nonsmooth functions in high dimensions. It is based on stochastic swarm intelligence, and inspired by consensus dynamics and opinion formation.

The underlying dynamics is flexible enough to incorporate different mechanisms widely used in evolutionary computation and machine learning, such as memory effects and gradient information.

Version 4.0

Date 23.12.2022

------

## R e f e r e n c e s

### Consensus-Based Optimization Methods Converge Globally

https://arxiv.org/abs/2103.15130

and

### Convergence of Anisotropic Consensus-Based Optimization in Mean-Field Law

https://arxiv.org/abs/2111.08136, https://link.springer.com/chapter/10.1007/978-3-031-02462-7_46

by

- Massimo &nbsp; F o r n a s i e r &nbsp; (Technical University of Munich & Munich Center for Machine Learning & Munich Data Science Institute), 
- Timo &nbsp; K l o c k &nbsp; (deeptech consulting & formerly Simula Research Laboratory),
- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich & Munich Center for Machine Learning)

and

### Leveraging Memory Effects and Gradient Information in Consensus-Based Optimization: On Global Convergence in Mean-Field Law

https://arxiv.org/abs/2211.12184

by

- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich & Munich Center for Machine Learning)

------

## D e s c r i p t i o n

MATLAB implementation, which illustrates CBO and the intuition behind our novel global convergence analysis approach, and tests the method on a complicated high-dimensional and well understood benchmark problem in the machine learning literature.

For the reader's convenience we describe the folder structure in what follows:

BenchmarkFunctions
* objective_function.m: objective function generator
* ObjectiveFunctionPlot1/2d.m: plotting routine for objective function

EnergyBasedCBOAnalysis
* analyses: convergence and parameter analyses of CBO
    * CBONumericalExample.m: testing script
    * CBOIntuition_averageparticle.m: Intuition behind our global convergence analysis: CBO always performs a gradient descent of the squared Euclidean distance to the global minimizer
    * CBOParameters_PhaseTransition.m: Phase transition diagrams for parameter analysis
    * DecayComparison_V_anisotropicandisotropic_differentd.m: Comparison of the decay behavior of isotropic and anisotropic CBO in different dimensions
    * DecayComparison_VandVar_an_isotropic_differentd.m: Comparison of the decay behavior of the functional V and the variance Var for isotropic or anisotropic CBO  in different dimensions
    * DecayComparison_VandVar_differentV0.m: Comparison of the decay behavior of the functional V and the variance Var for different initial conditions for isotropic or anisotropic CBO
* CBO: code of CBO optimizer
    * compute_valpha.m: computation of consensus point
    * CBO_update: one CBO step
    * CBO.m: CBO optimizer
    * CBOmachinelearning.m: CBO optimizer for machine learning applications
* visualizations: visualization of the CBO dynamics
    * CBODynamicsIllustration.m: Illustration of the CBO dynamics
    * CBOIllustrative.m: Illustration of the CBO at work
    * CBOMemorygradientDecayComparison_VandVar.m: Illustration of the different decay behavior of the functional V and the variance Var

EnergyBasedCBOmemorygradientAnalysis
* analyses: convergence and parameter analyses of CBOMemoryGradient
    * CBOmemorygradientNumericalExample.m: testing script
    * CBOMemorygradientParameters_PhaseTransition.m: Phase transition diagrams for parameter analysis
    * CBOMemorygradientDecayComparison_parameters.m: Comparison of the decay behavior of CBOMemoryGradient for different parameters
* CBOmemorygradient: code of CBOmemorygradient optimizer
    * compute_yalpha.m: computation of consensus point
    * S_beta.m: function to compare current with in-time best position
    * CBOmemorygradient_update.m: one CBOmemorygradient step
    * CBOmemorygradient.m: CBOmemorygradient optimizer
    * CBOmemorygradientmachinelearning.m: CBOmemorygradient optimizer for machine learning applications
* visualizations: visualization of the CBOmemorygradient dynamics
    * CBOmemorygradientDynamicsIllustration.m: Illustration of the CBOmemorygradient dynamics
    * CBOmemorygradientIllustrative.m: Illustration of the CBOmemorygradient at work

Example_NN: machine learning experiments with CBO and CBOMemoryGradient as optimization methods for training
* architecture
    * NN.m: forward pass of NN
    * eval_accuracy.m: evaluate training or test accuracy
    * comp_performance.m: compute and display loss and training or test accuracy
* data: data and function to load data
* Scripts_for_CBO
    * MNISTClassificationCBO.m: script for training the NN with CBO for MNIST data set
* Scripts_for_CBOmemorygradient
    * MNISTClassificationCBOmemorygradient.m: script for training the NN with CBOMemoryGradient for MNIST data set

Example_CompressedSensing: compressed sensing experiments with CBO and CBOMemoryGradient as optimizers
* objective_function
    * objective_function_compressed_sensing.m: evaluation of the objective function associated with compessed sensing
* CBOmemorygradientCompressedSensing.m: script solving a CS problem with CBO and CBOMemoryGradient
* CBOmemorygradientCompressedSensing_PhaseTransition.m: Phase transition diagrams for compressed sensing

------

## C i t a t i o n s

```bibtex
@article{CBOConvergenceFornasierKlockRiedl,
      title = {Consensus-Based Optimization Methods Converge Globally},
     author = {Massimo Fornasier and Timo Klock and Konstantin Riedl},
       year = {2021},
    journal = {arXiv preprint arXiv:2103.15130},
}
@article{CBOAnisotropicFornasierKlockRiedl,
      title = {Convergence of Anisotropic Consensus-Based Optimization in Mean-Field Law},
     author = {Massimo Fornasier and Timo Klock and Konstantin Riedl},
       year = {2021},
    journal = {arXiv preprint arXiv:2111.08136},
}
@article{CBOMemoryGradientRiedl,
      title = {Leveraging Memory Effects and Gradient Information in Consensus-Based Optimization: On Global Convergence in Mean-Field Law},
     author = {Konstantin Riedl},
       year = {2022},
    journal = {arXiv preprint arXiv:2211.12184},
}
```
