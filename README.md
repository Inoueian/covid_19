# covid_19
This repo contains Jupyter notebooks that use the pymc3 package in Python to perform Bayesian modeling of the number of infections and deaths from the novel coronavirus, particularly on data from Chicago. Different data sets can imported to perform the same analysis.

There are 5 models, from simple to complicated. The first 2 models are not in any way realistic, as they do not account for changes in behavior over time, but they should help the understanding of the more realistic models:
1. [Exponential model](../master/notebooks/exponential_model.ipynb) assumes that the number of infections increases as a simple exponential function.
2. [Generation interval model](../master/notebooks/generation_interval_model.ipynb) assumes that the basic reproduction number, `R_0`, is applicable for all time.
3. [Change point model](../master/notebooks/change_point_model.ipynb) assumes that the reproduction number, `R`, changed from the initial value, `R_0`, to a smaller value, by some ratio, `R_ratio`, between 0 and 1 at some point in time.
4. [2 change points model](../master/notebooks/two_change_points_model.ipynb) assumes that the reproduction number changed twice, once sometime in March, then the second time on March 22, when the shelter-in-place order fully came into effect (this is for Chicago).
5. [2 fixed change points model](../master/notebooks/two_fixed_change_points_model.ipynb) assumes that the reproduction number changed twice, once on March 22, then on May 1, when masks were required when social distancing was not feasible. This is semi-regularly being updated as of June 12.

Sources for the parameters used in the models are given in these notebooks. Results are accurate only to the extent that the assumptions are valid.

The models use the number of deaths, rather than the number of confirmed cases, as the data. This is because the number of deaths due to COVID-19 is a much less noisy measure. The number of confirmed cases depends greatly on the number of tests given, how the test takers are chosen, etc., which are hard to model accurately. The number of confirmed cases does have the advantage that it responds faster to changes in the number of new infections.

Functions used in these notebooks are defined in [covid_func.py](../master/covid_func.py).
