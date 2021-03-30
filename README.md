# CEBasedCBOAnalysis
Numerical illustration of a convex envelope based analysis framework for consensus based optimization

Version 1.0

Date 30.03.2021

------

## R e f e r e n c e

### Consensus-based optimization methods converge globally in mean-field law

by

- Massimo &nbsp; F o r n a s i e r &nbsp; (Technical University of Munich & Munich Data Science Institute), 
- Timo &nbsp; K l o c k &nbsp; (Simula Research Laboratory & University of San Diego),
- Konstantin &nbsp; R i e d l &nbsp; (Technical University of Munich)

https://arxiv.org/abs/2103.15130

------

## D e s c r i p t i o n

MATLAB implementation illustrating CBO and the intuition behind our convex
envelope based global convergence analysis approach.

For the reader's convenience we group the MATLAB scripts into five different
categories:

Visualization of the objective functions
- ObjectiveFunctionPlot1d.m
- ObjectiveFunctionPlot2d.m

CBO illustration
- CBOIllustrative.m
- DecayComparison_JandVar.m

CBO intuition
- CBOIntuition_averageparticle.m

Comparison of our functional J with variance of particles
- DecayComparison_JandVar.m
- DecayComparison_JandVar_differentV0.m

Further material
- AlternativeDynamics.m
- DdeltaEstimation.m
