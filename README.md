This repository includes a Julia implementation of an inexact regularized L-shaped method discussed in the paper "[On the Convergence of L-shaped Algorithms for Two-Stage Stochastic Programming](https://arxiv.org/abs/2309.01244)". The two folders correspond to two experiments reported in the paper. We use the proposed inexact regularized L-shaped method for solving 7 two-stage stochastic programming problems: 20term, Gbd, LandS, SH10, SH31, SSN, and Storm.

One can use   *dev '{$PathToFolder}/first_experiment'*   OR   *dev '{$PathToFolder}/second_experiment'*   to switch between two experiments. 

To use our code, please cite the following paper: 
```
@article{birge2023convergence,
  title={On the Convergence of L-shaped Algorithms for Two-Stage Stochastic Programming},
  author={Birge, John R and Lu, Haihao and Zhou, Baoyu},
  journal={arXiv preprint arXiv:2309.01244},
  year={2023}
}
```
