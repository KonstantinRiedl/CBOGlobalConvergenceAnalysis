# EnergyBasedCBOAnalysis
Numerical illustration of a novel analysis framework for consensus-based optimization (CBO) and numerical experiments demonstrating the practicability of the method.

CBO is a multi-agent metaheuristic derivative-free optimization method capable of globally minimizing nonconvex and nonsmooth functions in high dimensions. It is based on stochastic swarm intelligence, and inspired by consensus dynamics and opinion formation.

Version 3.0

Date 16.11.2021

------

## R e f e r e n c e s

### Consensus-Based Optimization Methods Converge Globally in Mean-Field Law

https://arxiv.org/abs/2103.15130

and

### Convergence of Anisotropic Consensus-Based Optimization in Mean-Field Law

https://arxiv.org/abs/2111.XXXXX

by

- Massimo &nbsp; F o r n a s i e r &nbsp; (Technical University of Munich & Munich Data Science Institute), 
- Timo &nbsp; K l o c k &nbsp; (Simula Research Laboratory & University of San Diego),
- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich)

------

## D e s c r i p t i o n

MATLAB implementation, which illustrates CBO and the intuition behind our novel global convergence analysis approach, and tests the method on a complicated high-dimensional and well understood benchmark problem in the machine learning literature.

For the reader's convenience we describe the folder structure in what follows:

BenchmarkFunctions
* objective_function.m: objective function generator
* ObjectiveFunctionPlot1/2d.m: plotting routine for objective function

EnergyBasedCBOAnalysis
* CBO: code of CBO optimizer
    * compute_valpha.m: computation of consensus point
    * CBO_update: one CBO step
    * CBO.m: CBO optimizer
    * CBOmachinelearning.m: CBO optimizer for machine learning applications
* visualizations: visualization of the CBO dynamics
    * CBODynamicsIllustration.m: Illustration of the CBO dynamics
    * CBOIllustrative.m: Illustration of the CBO at work
    * DecayComparison_VandVar.m: Illustration of the different decay behavior of the functional V and the variance Var
* analyses: convergence and parameter analyses of CBO
    * CBOIntuition_averageparticle.m: Intuition behind our global convergence analysis: __CBO always performs a gradient descent of the squared Euclidean distance to the global minimizer__
    * CBOParameters_PhaseTransition.m: Phase transition diagrams for parameter analysis
    * DecayComparison_V_anisotropicandisotropic_differentd.m: Comparison of the decay behavior of isotropic and anisotropic CBO in different dimensions
    * DecayComparison_VandVar_an_isotropic_differentd.m: Comparison of the decay behavior of the functional V and the variance Var for isotropic or anisotropic CBO  in different dimensions
    * DecayComparison_VandVar_differentV0.m: Comparison of the decay behavior of the functional V and the variance Var for different initial conditions for isotropic or anisotropic CBO

NN: machine learning experiments with CBO as optimization method for training
* architecture
    * NN.m: forward pass of NN
    * eval_accuracy.m: evaluate training or test accuracy
    * comp_performance.m: compute and display loss and training or test accuracy
* data: data (not included) and function to load data
* Scripts_for_CBO
    * MNISTClassificationCBO.m: script training the NN for MNIST with CBO

------

## C i t a t i o n s

```bibtex
@article{CBOConvergenceFornasierKlockRiedl,
      title = {Consensus-Based Optimization Methods Converge Globally in Mean-Field Law},
     author = {Massimo Fornasier and Timo Klock and Konstantin Riedl},
       year = {2021},
    journal = {arXiv preprint arXiv:2103.15130},
}
@article{CBOAnisotropicFornasierKlockRiedl,
      title = {Convergence of Anisotropic Consensus-Based Optimization in Mean-Field Law},
     author = {Massimo Fornasier and Timo Klock and Konstantin Riedl},
       year = {2021},
    journal = {arXiv preprint arXiv:2111.XXXXX},
}
```
