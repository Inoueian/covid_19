# covid_19
This repo contains Jupyter notebooks that use the pymc3 package in Python to perform Bayesian modeling of the number of infections and deaths from the novel coronavirus, particularly on data from Chicago. Different data sets can imported to perform the same analysis.

There are 3 models, from simple to complicated. Only the most complicated model is in any way realistic (it takes into account that people's behavior after the beginning of the simulation), but the simpler models should help the understanding of the change point model:
1. [Exponential model](../master/notebooks/exponential_model.ipynb) assumes that the number of infections increases as a simple exponential function.
2. [Generation interval model](../master/notebooks/generation_interval_model.ipynb) assumes that the basic reproduction number, `R_0`, is applicable for all time.
3. [Change point model](../master/notebooks/change_point_model.ipynb) assumes that the reproduction number, `R`, changed from the initial value, `R_0`, to a smaller value, by some ratio, `R_ratio`, between 0 and 1　at some point in time.

Sources for the parameters used in the models are given in these notebooks. Although the change point model embodies much of my current knowledge about the situation, I cannot guarantee the accuracy of its assumptions with great confidence.

The models use the number of deaths, rather than the number of confirmed cases, as the data. This is because the number of deaths due to COVID-19 is a much less noisy measure. The number of confirmed cases depends greatly on the number of tests given, how the test takers are chosen, etc., which are hard to model accurately. The number of confirmed cases does have the advantage that it responds faster to changes in the number of new infections.

Functions used in these notebooks are defined in [covid_func.py](../master/covid_func.py).
