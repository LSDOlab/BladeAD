---
title: Pitt–Peters dynamic inflow model
---

In dynamic inflow models, the goal is to derive an expression for the axial-induced velocity at the rotor disk. This is done by relating a set of inflow variables (states) to the rotor aerodynamic loads via a set of first-order differential equations. 

**Dynamic inflow models are applicable to edgewise AND axial flow conditions**. However, for axial (or mostly axial) flow, we recommend BEM theory due to its simplicity, robustness and reasonable accuracy.

In the simplest formulation, there are three flow states, one constant ($\lambda_{0}$) and two linear ($\lambda_{c},\lambda_{s}$), which represent the perturbations in the wake-induced downwash. The normalized induced velocity ($\lambda=\frac{u_x^i}{\Omega R}$) is approximated by a first order Fourier expansion, in which the coefficients are the flow states, such that

$$
\lambda(\overline{r}, \psi) = \lambda_{0} + \overline{r}\lambda_{c}\cos{\psi} + \overline{r}\lambda_{s}\sin{\psi},
$$

where $\overline{r}$ is the normalized radius and $\psi$ is the azimuth angle in the rotor plane. The response of the inflow states to aerodynamic perturbations is described a set of linear, first order differential equations as

$$
LM \begin{bmatrix}
\dot{\lambda}_{0} \\ 
\dot{\lambda}_{c} \\
\dot{\lambda}_{s}
\end{bmatrix} 
+ \begin{bmatrix}
\lambda_{0} \\ 
\lambda_{c} \\
\lambda_{s}
\end{bmatrix} 
= L \begin{bmatrix}
C_T \\ 
-C_{M_{y}} \\
C_{M_{x}}
\end{bmatrix},
$$

where the aerodynamic loading coefficient are in terms of the integrated thrust and moment coefficients, which are given by

$$
C_T = \frac{T}{\rho A \left(\Omega R\right)^2}
$$

$$
C_{M_{x}} = \frac{M_x}{\rho A \left(\Omega R\right)^2 R}
$$

$$
C_{M_{y}} = \frac{M_y}{\rho A \left(\Omega R\right)^2 R},
$$

where $M_x$ is the rotor hub roll moment, $M_y$ is the rotor hub pitching moment, and $A$ is the rotor area. **It is important to note that to compute the aerodynamic loads, we require an airfoil model just like in BEM theory.**

Pitt and Peters {cite:p}`pitt1980theoretical` developed expression for the stability matrix $L$ and the mass matrix $M$, which are given as 

$$
L = \frac{1}{V_{\textrm{eff}}}\begin{bmatrix}
\frac{1}{2} & \frac{-15\pi}{64}\sqrt{\frac{1-\cos\chi}{1+\cos\chi}} & 0\\
\frac{15\pi}{64}\sqrt{\frac{1-\cos\chi}{1+\cos\chi}} & \frac{4\cos\chi}{1 + \cos\chi}& 0 \\
0 & 0 & \frac{4}{1 + \cos\chi}
\end{bmatrix},
$$

and

$$
M = \begin{bmatrix}
\frac{128}{75\pi} & 0 & 0 \\
0 & \frac{64}{45\pi} & 0 \\
0  & 0 & \frac{64}{45\pi} \\
\end{bmatrix},
$$

where $\chi$ is the wake skew angle and $V_{\text{eff}}$ is the effective velocity given as 

$$
V_{\text{eff}} = \frac{\mu^2 + \lambda\left(\lambda + \lambda_i\right)}{\sqrt{\mu^2 + \lambda^2}}.
$$

In the expression for the effective velocity, $\mu$ is the normalized in-plane component of inflow velocity, i.e., $\mu=\frac{V_{\infty}\cos i}{\Omega R}$. This quantity is also know as the *rotor advance ratio* where $i$ is the rotor disk incidence angle or angle of attack (positive for forward tilt). The other unknown the expression for $V_{\text{eff}}$ is $\lambda_i$, which is the mean (axial) induced velocity and is related to the thrust coefficient via momentum theory as

$$
\lambda_i = \frac{C_T}{2\sqrt{\mu^2 + \left(\lambda_i+\mu_z\right)^2}},
$$
where $\mu_z=\frac{V_{\infty}\sin i}{\Omega R}$ is the axial (i.e., normal) component of the rotor inflow and is also referred to as *axial velocity ratio*.
Solving for $\lambda_i$ is typically done via a Newton-Raphson algorithm. 

**Lastly, we note that in the current version of `BladeAD`, we solve for steady state and neglect transient effects**

For a more detailed derivation of the dynamic inflow model in general, we refer the reader to a comprehensive text book on rotorcraft aero-mechanics by Johnson {cite:p}`johnson2013rotorcraft`, specifically (Ch. 11). Chen {cite:p}`chen1989survey` provides a comprehensive, high-level summary of various nonlinear inflow models that have been developed.


## Bibliography

```{bibliography} ../references_pp.bib
```
