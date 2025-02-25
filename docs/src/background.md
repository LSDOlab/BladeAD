---
title: Background
---

In this section, we give a high-level overview of the theory behind the three models that are implemented in this software package. 
For a more detailed discussion we refer the reader to appropriate papers. 

## Blade element momentum (BEM) theory
Blade element momentum theory is a low-fidelity physics-based model that combines momentum theory with 2-D airfoil analysis to predict the performance of a propeller. 

###  Momentum theory
The idea behind momentum theory is expressed visually in the figure below. 

```{figure} /src/images/Streamtube4.svg
:align: center
:alt: momentum_theory
```

Thrust is computed by condsidering the change in (linear) momentum as the flow is accelerated through a so-called "stream tube".
The momentum equation in integral form simplifies to

$$
T = \int 2\pi\rho u_x\left(u_x^{\prime} - V_x \right)r\,dr.
$$

The unkowns are $u_x$ and $u_x^{\prime}$, which are the axial velocities at the disk and the wake.
One of the fundamental results of momentum theory is that, neglecting the effect of rotational induced velocities, the axial velocity at the disk $u_x$ is the mean of the free stream velocity $V_x$ and the axial velocity in the wake $u_x^{\prime}$, such that

$$
u_x = \frac{1}{2}\left(V_x + u_x^{\prime}\right).
$$

Solving for $u_x^{\prime}$, we can substitute the expression into the integral equation and differentiate to obtain an expression for infinitesimal thrust $dT$ as 

$$
dT = 4\pi\rho u_x\left(u_x - V_x\right)r\,dr =  4\pi\rho V_x^2\left(1+a_x\right)a_xr\,dr, 
$$

where the non-dimensional induction factor $a_x$ is related to the axial velocity at the rotor disk as

$$
u_x = V_x\left(1 + a_x \right).
$$

Going from the integral to the differential formulation introduces error as interactions between infinitesimal blade elements is neglected. 


By considering the angular momentum balance, a similar expression can be derived for the infinitesimal torque of a rotor. 
The expression is

$$
dQ = 2 \pi\rho u_x u_{\theta} r^2 dr = 4\pi\rho V_xV_{\theta}\left(1+a_x\right)a_{\theta}r^2\,dr,
$$
where $u_{\theta}$ is the induced velocity in the tangential direction and $V_{\theta}$ is the rotational speed (i.e., $\Omega r$).
The tangential induction factor is given as $a_{\theta} = \frac{u_{\theta}}{2V_{\theta}}$.

The equations for $dT$ and $dQ$ allow us to write the aerodynamic efficiency of a rotor **element** in terms of the induced velocities.
By definition, efficiency is the ratio of the output power
to the input power, leading to

$$
\eta = \frac{V_x\,dT}{\Omega\,dQ} = \frac{4 V_x\left(u_x - V_x\right)}{u_{\theta}V_{\theta}}.
$$

### Blade element theory
The equations for $dT$ and $dQ$ derived from momentum theory are in terms of two unknowns, the axial velocity at the disk $u_x$ and the tangential-indcued velocity $u_{\theta}$.
Therefore, we cannot directly compute thrust and torque. 
To close the system, we introduce blade element theory. 
The idea is shown in the figure below.

```{figure} /src/images/Blade_element_inflow.svg
:align: center
:alt: momentum_theory
```

In blade element theory, the rotor blade is discretized in the spanwise direction and each element is treated as a 2-D airfoil. 
The sectional thrust and torque can be written in terms of the sectional lift and drag, such that

$$
    dT = dL\cos\phi - dD\sin\phi,
$$
$$
    \frac{dQ}{r} = dL\sin\phi + dD\cos\phi.
$$

Writing $dL$ and $dD$ in terms of the lift and drag coefficients, the expressions for sectional thrust and torque become

$$
    dT = \frac{1}{2}C_x\rho W^2 B c  dr,
$$
$$
    \frac{dQ}{r} =\frac{1}{2}C_{\theta}\rho W^2 B c dr,
$$

where, $c$ is the airfoil chord length and  $C_x$, $C_{\theta}$, and $W$ are given by 

$$
\begin{equation} \label{Cx Ctheta matrix eqn}
    \begin{bmatrix}
        C_x \\
        C_{\theta} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
        \cos\phi & - \sin\phi \\
        \sin\phi & \cos\phi \\
    \end{bmatrix}
    \begin{bmatrix}
        C_l\\
        C_d\\
    \end{bmatrix}, \quad \textrm{and} \quad
    W = \sqrt{u_x^2 + \left(V_{\theta}- \frac{1}{2}u_{\theta}\right)^2}.
\end{equation}
$$

It is important to note that **$\mathbf{C_l}$ and $\mathbf{C_d}$ are functions of angle of attack, Reynolds number, and Mach number.**

$$
\begin{equation}\label{Cl Cd}
    C_l = f\left(\alpha,Re, Ma\right)\quad\text{and} \quad C_d = g\left(\alpha,Re, Ma\right).
\end{equation}
$$

It is common to use a surrogate model to predict the lift and drag coefficients. 
Often `XFOIL` is a good option to generate a comprehensive data set for training. 
`BladeAD` provides built-in capability to train a 2-D ($\alpha$ and $Re$) machine learning surrogate model based on `XFOIL` data that can be accessed via [airfoil tools](http://www.airfoiltools.com).
However, **for a quick start, a 1-D airfoil model (linear lift, quadratic drag)** can be used as a simple substituted.
`BladeAD` uses the Viterna method (see e.g., {cite:p}`mahmuddin2017airfoil`) to extrapolate the airfoil polar to $\pm90^{\circ}$ degrees. 
This is particularly important since it is not uncommon for sections of the rotor blade to be stalled depending on the operating condition.

### Combining blade element momentum (BEM) theory
Blade element theory gives us a second set of equations to solve for thrust and torque. 
By equating the expressions for sectional thrust $dT$ and torque $dQ$ derived from momentum and blade element theory, we can write the induced velocities as

$$
\begin{equation}\label{BEM utheta}
    u_{\theta} = \frac{2 \sigma C_{\theta} V_{\theta}}{4\sin\phi\cos\phi + \sigma C_{\theta}}\quad \text{and} \quad  u_x = V_x + \frac{\sigma C_x V_{\theta}}{4\sin\phi\cos\phi + \sigma C_{\theta}},
\end{equation}
$$

where $C_x$ and $C_{\theta}$ are given above and $\sigma$ is the sectional blade solidity, defined as $\sigma = Bc/(2\pi r)$ for a rotor with $B$ number of blades. 
The induced velocities $u_x$ and $u_{\theta}$ can be related the inflow geometry shown in the figure above by

$$
\tan\phi = \frac{u_x}{V_{\theta}-\frac{u_{\theta}}{2}}.
$$

This allows us to derive the following residual equation in terms of the inflow angle $\phi$.

$$
\begin{equation}\label{residual function}
    R\left(\phi\right) = \frac{J}{\pi}\left(\sigma C_{\theta} + 4\sin\phi\cos\phi\right) + \left(\sigma C_x - 4\sin^2\phi\right).
\end{equation}
$$

To solve for $\phi$, we need to know how the chord length $c$ as well as the local twist angle $\theta$ vary along the blade. 
This is because $C_l$ and $C_d$ depend on the angle of attack $\alpha$, which we can express as $\alpha = \theta - \phi$. 
In addition, we also require knowledge of the lift and drag polar as explained previously. 
Assuming that we know these pieces of information, we can now solve for $\phi$ via a bracketed search algorithm. 
It can be shown (see e.g.,  {cite:p}`ning2014simple,ruh2021robust`) that a bracketed search algorithm is guaranteed to converge for all reasonable operating conditions.
After solving for $\phi$, we use the equations for $u_x$ and $u_{\theta}$ to find the induced velocities, which allows us to use either the blade element or the momentum equations to compute thrust and torque. 

## Bibliography

```{bibliography} references.bib
```