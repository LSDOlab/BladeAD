# Examples

We provide several examples of how to use the various solvers (BEM, Pitt--Peters, Peters--He). 
The API is shared among the solvers such that each model is evaluated via a `solver_model.evaluate()` call.
We start with simple examples that only consider the forward analysis and end with more complex examples that consider design optimization of the rotor.

```{toctree}
:maxdepth: 2
:caption: List of examples
:titlesonly:
:numbered:
:includehidden:

_temp/examples/ex_run_bem
_temp/examples/ex_run_pitt_peters
_temp/examples/ex_run_peters_he
_temp/examples/ex_run_bem_optimization
_temp/examples/ex_vnv_nasa_reports
```