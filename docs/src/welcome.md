# Welcome to BladeAD

![alt text](/src/images/lsdolab.png "Title displayed")

`BladeAD` is a package for **low-fidelity rotor-aerodynamic analysis and design**. 
It implements three rotor-aerodynamic models:
1. Blade element momentum (BEM) theory
2. Pitt--Peters dynamic inflow model
3. Peters--He dynamic inflow model

For the dynamic inflow models, the current version only considers steady operating conditions.
`BladeAD` is implemented using a newly developed algebraic modeling language called the [Computational 
System Design Language](https://csdl-alpha.readthedocs.io/en/latest/) (`CSDL`).
The critical feature of `CSDL` is that it **automates sensitivity analysis** using a graph-based modeling approach.
This allows the user to make use fo the state-of-the art **gradient-based** optimization algorithms to optimize rotor designs. 

# Cite us
```none
@article{ruh2023fast,
  title={Fast and robust computation of optimal rotor designs using blade element momentum theory},
  author={Ruh, Marius L and Hwang, John T},
  journal={AIAA Journal},
  volume={61},
  number={9},
  pages={4096--4111},
  year={2023},
  publisher={American Institute of Aeronautics and Astronautics}
}
```

<!-- Remove/add custom pages from/to toc as per your package's requirement -->

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/tutorials
<!-- src/custom_1
src/custom_2 -->
src/examples
src/api
```
