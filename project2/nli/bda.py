"""Tutorial 2's Bayesian regression of correctness on input features, for repeated fits.

Same model as part2/analysis.py: logit = global_mean + sys_shift[s] + w[s] . f, Laplace(0, 1)
priors on shifts and weights, Normal(0, 2) on the global mean. The only change is the
likelihood: each system was trained 15 times (5 folds x 3 repetitions), so for every item we
observe k correct predictions out of 15. A Binomial(15, p) on k is the product of the 15
Bernoulli terms the tutorial would write for those fits, which makes inference 15x cheaper.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def model(num_systems, sys_id, features, trials, correct=None, s_std=1.0, w_std=1.0):
    d = features.shape[1]
    global_mean = numpyro.sample("global_mean", dist.Normal(0.0, 2.0))
    with numpyro.plate("systems", num_systems):
        sys_shift = numpyro.sample("sys_shift", dist.Laplace(0.0, s_std))
        w = numpyro.sample("w", dist.Laplace(0.0, w_std).expand([d]).to_event(1))
    logit = global_mean + sys_shift[sys_id] + jnp.sum(w[sys_id] * features, -1)
    numpyro.sample("correct", dist.Binomial(total_count=trials, logits=logit), obs=correct)


def fit(sys_id, features, correct, trials, num_systems, seed=0, num_warmup=1000, num_samples=1000, num_chains=4):
    """Features are standardised first, as in the tutorial, so weights are per standard deviation.

    BDA_CHAINS=parallel runs the chains at once on CPU cores (same model, samples and seed); it
    needs numpyro.set_host_device_count before jax starts (see errors.py)."""
    x = np.asarray(features, dtype=float)
    mean, scale = x.mean(0), np.sqrt(x.var(0) + 1e-5)
    x = (x - mean) / scale
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method=os.environ.get("BDA_CHAINS", "sequential"), progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), num_systems=num_systems, sys_id=jnp.asarray(sys_id, dtype=jnp.int32),
             features=jnp.asarray(x), trials=jnp.asarray(trials), correct=jnp.asarray(correct))
    return mcmc
