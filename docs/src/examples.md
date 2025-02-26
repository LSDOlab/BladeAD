# Examples

We provide several examples of how to use the various solvers (BEM, Pitt--Peters, Peters--He). 
The API is shared among the solvers such that each model is evaluated via a `solver_model.evaluate()` call.
We start with simple examples that only consider the forward analysis and end with more complex examples that consider design optimization of the rotor.

```{toctree}
:maxdepth: 1
:caption: List of examples
:titlesonly:
:numbered:
:includehidden:

examples/ex_run_bem
examples/ex_run_pitt_peters
examples/ex_run_peters_he
examples/ex_run_bem_optimization
```