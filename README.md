# EnergyBasedCBOAnalysis
Numerical illustration of a novel analysis framework for consensus based optimization

Version 2.0

Date 04.04.2021

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

MATLAB implementation illustrating CBO and the intuition behind our novel
global convergence analysis approach.

For the reader's convenience we group the MATLAB scripts into five different
categories:

Visualization of the objective functions
- ObjectiveFunctionPlot1d.m
- ObjectiveFunctionPlot2d.m

CBO illustration
- CBOIllustrative.m
- DecayComparison_VandVar.m

CBO intuition
- CBOIntuition_averageparticle.m

Comparison of our functional V with variance of particles Var
- DecayComparison_VandVar.m
- DecayComparison_VandVar_differentV0.m

Further material
- AlternativeDynamics.m
- DdeltaEstimation.m
