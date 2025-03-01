# NASA Inflow Measurements

For these validation studies, we consider experimental data collected by NASA. The collected data are measuremunts of the downwash of a test rotor at multiple span-wise and azimuthally distributed locations for different operating conditions. The experiments were conducted by Elliott et al. {cite:p}`elliott1988inflowV1,elliott1988inflowV2,hoad1989inflow` and consider various operating conditions. For this validation study, we chose three operating conditions characterized by advance ratios of $\mu=0.15,\,0.23,\,0.35$.

The rotor under consideration has the following high-level specifications:
- number of blades: 4
- radius: $R=0.86\,m$
- blade chord: $c=0.066\,m$ and is constant along the span
- blade twist: linearly varying from root to tip by $8^{\circ}$. The twist is off-set such that at $r/R=0.75$, the blade twist with respect to the rotor plane is $0^{\circ}$
- airfoil: NACA 0012

**For airfoil model, we use a 2-D machine learning model, trained based on `XFOIL` data, available on [airfoil tools](http://www.airfoiltools.com/airfoil/details?airfoil=n0012-il).

We summarize the key parameters of the experimental setup in the table below for the three considered cases. Not all parameters (e.g., coning angle) are considered in the numerical experiments, i.e., not all parameters are accounted for in the computational model. The following parameters are considered in the analysis:
- $\alpha$ (deg): rotor disk tilt angle
- $\theta_0$ (deg): collective pitch at $r/R=0.75$
- $\theta_{1c} (deg)$: lateral cyclic pitch (cosine coefficient of first harmonic)
- $\theta_{1s} (deg)$: longitudinal cyclic pitch (sin coefficient of first harmonic)
- $\xi_0$: mean lag angle (zero-th harmonic of blade lag motion)


| Test Case  | $C_T$ | $V_{\infty}$ (m/s) | RPM | $\alpha$ | $\theta_0$ | $\theta_{1c}$ | $\theta_{1s}$ | $\xi_0$ |
|------------|--------|----------------|------|---------------|-----------------|-----------------|-----------------|--------------|
| 1 | 0.0064 | 28.50 | 2113 | -3.00 | 9.37 | -1.11 | 3.23 | 0.95 |
| 2 | 0.0064 | 43.86 | 2113 | -3.04 | 8.16 | -1.52 | 4.13 | 0.90 |
| 3 | 0.0064 | 66.75 | 2113 | -5.70 | 9.20 | -0.30 | 6.80 | 1.30 |


## Results

We use the Pitt-Peters models and the Peters--He model with different number of flow states with the above specifications and operating conditions.

### Test Case 1

```{figure} /src/images/nasa_vnv_2_ph.svg
:align: center
:alt: vnv_nasa_1_ph
:figwidth: 100 %
```

```{figure} /src/images/nasa_vnv_2_pp.svg
:align: center
:alt: vnv_nasa_1_pp
```

### Test Case 2

```{figure} /src/images/nasa_vnv_1_ph.svg
:align: center
:alt: vnv_nasa_1_ph
:figwidth: 100 %
```

```{figure} /src/images/nasa_vnv_1_pp.svg
:align: center
:alt: vnv_nasa_1_pp
```

### Test Case 3

```{figure} /src/images/nasa_vnv_3_ph.svg
:align: center
:alt: vnv_nasa_3_ph
:figwidth: 100 %
```

```{figure} /src/images/nasa_vnv_3_pp.svg
:align: center
:alt: vnv_nasa_3_pp
```


## Bibliography

```{bibliography} ../nasa_vnv_references.bib
```