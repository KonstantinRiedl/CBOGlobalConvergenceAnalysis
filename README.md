# CBOGlobalConvergenceAnalysis
Numerical illustration of a novel analysis framework for consensus-based optimization (CBO) and numerical experiments demonstrating the practicability of the method.

CBO is a multi-agent metaheuristic derivative-free optimization method capable of globally minimizing nonconvex and nonsmooth functions in high dimensions. It is based on stochastic swarm intelligence, and inspired by consensus dynamics and opinion formation.

The underlying dynamics is flexible enough to incorporate different mechanisms widely used in evolutionary computation and machine learning, such as memory effects and gradient information.

Version 6.0

Date 13.11.2023

------

## R e f e r e n c e s

### Consensus-Based Optimization Methods Converge Globally

https://arxiv.org/abs/2103.15130

by

- Massimo &nbsp; F o r n a s i e r &nbsp; (Technical University of Munich & Munich Center for Machine Learning & Munich Data Science Institute), 
- Timo &nbsp; K l o c k &nbsp; (deeptech consulting & formerly Simula Research Laboratory),
- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich & Munich Center for Machine Learning)

and

### Convergence of Anisotropic Consensus-Based Optimization in Mean-Field Law

https://arxiv.org/abs/2111.08136, https://link.springer.com/chapter/10.1007/978-3-031-02462-7_46

by

- Massimo &nbsp; F o r n a s i e r &nbsp; (Technical University of Munich & Munich Center for Machine Learning & Munich Data Science Institute), 
- Timo &nbsp; K l o c k &nbsp; (deeptech consulting & formerly Simula Research Laboratory),
- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich & Munich Center for Machine Learning)

and

### Leveraging Memory Effects and Gradient Information in Consensus-Based Optimization: On Global Convergence in Mean-Field Law

https://arxiv.org/abs/2211.12184, https://doi.org/10.1017/S0956792523000293

by

- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich & Munich Center for Machine Learning)

and

### Gradient is All You Need?

https://arxiv.org/abs/2306.09778

by

- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich & Munich Center for Machine Learning),
- Timo &nbsp; K l o c k &nbsp; (deeptech consulting & formerly Simula Research Laboratory),
- Carina &nbsp; G e l d h a u s e r &nbsp; (Technical University of Munich & Munich Center for Machine Learning),
- Massimo &nbsp; F o r n a s i e r &nbsp; (Technical University of Munich & Munich Center for Machine Learning & Munich Data Science Institute)

and

### Consensus-Based Optimization with Truncated Noise

https://arxiv.org/abs/2310.16610

by

- Massimo &nbsp; F o r n a s i e r &nbsp; (Technical University of Munich & Munich Center for Machine Learning & Munich Data Science Institute), 
- Peter &nbsp; R i c h t á r i k &nbsp; (King Abdullah University of Science and Technology & KAUST AI Initiative & SDAIA-KAUST Center of Excellence in Data Science and Artificial Intelligence), 
- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich & Munich Center for Machine Learning),
- Lukang &nbsp; S u n &nbsp; (King Abdullah University of Science and Technology & KAUST AI Initiative),

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

CBOstochasticGD
* algorithms: code of CH scheme and GD/Langevin scheme
    * CH: CH scheme
    * GDLangevin: GD/Langevin scheme
* analyses: analyses of CBO, CH and GD/Langevin scheme
    * CBOSchemeIntuition: Intuition that CBO behaves like a stochastic relaxation of gradient descent
    * CHSchemeIntuition: Intuition that CH behaves gradient-like
    * GDLangevinIntuition: Visualization of GD and Langevin dynamics

EnergyBasedCBOtruncateddiffusionAnalysis
* analyses: convergence and parameter analyses of CBOtruncateddiffusion
    * CBOtruncateddiffusionDecayComparison_V_differentM.m: Comparison of the decay behavior of CBO with and without truncated diffusion for different parameters
    * CBOtruncateddiffusionParameters_PhaseTransition.m: Phase transition diagrams for parameter analysis
* CBOtruncateddiffusion: code of CBOtruncateddiffusion optimizer
    * CBOtruncateddiffusion.m: CBOtruncateddiffusion optimizer
    * CBOtruncateddiffusion_update.m: one CBOtruncateddiffusion step

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
@inproceedings{CBOAnisotropicFornasierKlockRiedl,
         title = {Convergence of anisotropic consensus-based optimization in mean-field law},
        author = {Fornasier, Massimo and Klock, Timo and Riedl, Konstantin},
     booktitle = {Applications of Evolutionary Computation: 25th European Conference, EvoApplications 2022, Held as Part of EvoStar 2022, Madrid, Spain, April 20--22, 2022, Proceedings},
         pages = {738--754},
          year = {2022},
  organization = {Springer}
}
@article{CBOMemoryGradientRiedl,
         title = {Leveraging Memory Effects and Gradient Information in Consensus-Based Optimization: On Global Convergence in Mean-Field Law},
        author = {Konstantin Riedl},
          year = {2022},
       journal = {arXiv preprint arXiv:2211.12184},
}
@article{GradientIsAllYouNeedRiedlKlockGeldhauserFornasier,
         title = {Gradient is All You Need?},
        author = {Konstantin Riedl and Timo Klock and Carina Geldhauser and Massimo Fornasier},
          year = {2023},
       journal = {arXiv preprint arXiv:2306.09778},
}
@article{CBOTruncatedNoiseFornasierRichtarikRiedlSun,
         title = {Consensus-Based Optimization with Truncated Noise},
        author = {Massimo Fornasier and Peter Richt{\'a}rik and Konstantin Riedl and Lukang Sun},
          year = {2023},
       journal = {arXiv preprint arXiv:2310.16610},
}
```
