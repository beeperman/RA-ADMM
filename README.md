# Robust Accelerated Augmented Lagrangian Method and Alternating Direction Method of Multipliers

Authors: Fangda Gu\*, Jingqi Li\*, Ziye Ma\*

This is the code repo for EE227C final project.

In the project we introduced two robust methods on the dual:

- [RA-ALM] Robust Accelerated Augmented Laguangian Method
- [RA-ADMM] Robust Accelerated Alternating Direction Method of Multipliers

These methods are runnable using the following scripts:

- [RA-ALM] : `alms/raalm.py` One can find the `plot.py` under the same folder which generates the plots.
- [RA-ADMM] : `RA_ADMM.m` In addition, the Lyapunov candidate function plot can be generated from `RA_ADMM_Lyapunov.m`.



There are several paper for reference:
A Robust Accelerated Optimization Algorithm for Strongly Convex Functions

<https://arxiv.org/pdf/1710.04753.pdf>

Fast Alternating Direction Optimization Methods

<ftp://ftp.math.ucla.edu/pub/camreport/cam12-35.pdf>

ADMM and Accelerated ADMM as Continuous Dynamical Systems

<http://proceedings.mlr.press/v80/franca18a/franca18a.pdf>

Accelerated Alternating Direction Method of Multipliers

<https://dl.acm.org/doi/pdf/10.1145/2783258.2783400>

