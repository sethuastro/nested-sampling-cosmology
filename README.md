# Nested sampling: from scratch to supernova cosmology

I'm building this project to understand nested sampling properly, from the basic algorithm up to a real cosmological model comparison. Stage 1 is a nested sampler written in NumPy and tested on Gaussian problems where the evidence is known exactly. Stage 2 runs the same tests with BlackJAX Nested Slice Sampling and looks at how it scales with dimension. Next is a ΛCDM against wCDM comparison with the Pantheon+ supernovae.

I produced every result here on a CPU. Stage 2 uses GPU-capable code (JAX), but I have not run it on a GPU.

## Stage 1: a nested sampler from scratch

Files: `nested_sampler.py`, `gaussian_test.py`, `stage1_plots.py`, `stage1_calibration.py`

I started with a basic rejection-sampling nested sampler and went through it line by line against the maths below. Then I extended it in two ways: it now returns the likelihoods and prior masses of the dead points, and it has a second error estimate (`simulate_logz`). I also added the plots and the calibration test described in this section.

### The idea

The Bayesian evidence for a model is

$$Z = \int \mathcal{L}(\theta)\,\pi(\theta)\,d\theta .$$

Define the prior mass enclosed by the likelihood contour $\mathcal{L} = \lambda$:

$$X(\lambda) = \int_{\mathcal{L}(\theta) > \lambda} \pi(\theta)\,d\theta .$$

$X$ falls from 1 to 0 as $\lambda$ rises, so the multi-dimensional integral becomes a one-dimensional one:

$$Z = \int_0^1 \mathcal{L}(X)\,dX .$$

### The algorithm

1. Draw $N$ live points from the prior.
2. Find the live point with the lowest likelihood, $\mathcal{L}_i$.
3. Shrink the prior mass: $X_i = t_i X_{i-1}$, where $t_i$ is the largest of $N$ uniform numbers, so $t_i \sim \mathrm{Beta}(N, 1)$ and $\langle \ln t_i \rangle = -1/N$. This gives $X_i \approx e^{-i/N}$.
4. Add its contribution, $Z \mathrel{+}= \mathcal{L}_i\,w_i$ with $w_i = X_{i-1} - X_i$.
5. Replace that point with a new prior draw that has $\mathcal{L} > \mathcal{L}_i$.
6. Stop when the live points can add at most `dlogz` to $\ln Z$, then add the remaining live points, each with weight $X_{\rm final}/N$.

The posterior weight of each point is $p_i = \mathcal{L}_i w_i / Z$. The information is

$$H = \sum_i p_i \ln\frac{\mathcal{L}_i}{Z},$$

and the statistical uncertainty on the log-evidence is $\sigma(\ln Z) \approx \sqrt{H/N}$.

Everything is computed in log space to avoid underflow.

### The test

Likelihood: a normalised $d$-dimensional Gaussian with mean $\mu$ and width $\sigma$. Prior: uniform on $[-a, a]^d$. Then

$$Z = (2a)^{-d}\prod_{k=1}^{d}\left[\Phi\!\left(\frac{a-\mu_k}{\sigma}\right) - \Phi\!\left(\frac{-a-\mu_k}{\sigma}\right)\right],$$

which is very close to $(2a)^{-d}$ when the Gaussian sits well inside the box.

Results with $a = 5$, $\sigma = 1$, $\mu = 0$ and $N = 400$ live points (`python gaussian_test.py`):

| $d$ | $\ln Z$ (sampler) | $\ln Z$ (analytic) | Difference | Likelihood calls |
|---|---|---|---|---|
| 1 | −2.318 ± 0.048 | −2.303 | −0.3σ | 159,651 |
| 2 | −4.619 ± 0.067 | −4.605 | −0.2σ | 710,518 |
| 3 | −6.930 ± 0.082 | −6.908 | −0.3σ | 2,334,175 |

### The likelihood as a function of prior mass

`python stage1_plots.py`

![L against X](figures/stage1_likelihood_vs_prior_mass.png)

Left: the curve $\mathcal{L}(X)$ that the sampler traced out, against the exact curve. For this problem the contour is a sphere of radius $r$, so $X = V_d r^d/(2a)^d$ and $\mathcal{L}(X) = \mathcal{L}_{\max}\exp[-r(X)^2/2\sigma^2]$. The small offsets between the points and the line are real, not plotting errors. The sampler assumes $X_i = e^{-i/N}$, but the true $X_i$ wanders around that value. This is where the error on $\ln Z$ comes from.

Right: since $dX = X\,d\ln X$, the quantity $\mathcal{L}X$ is the evidence contributed per unit $\ln X$. Its peak is where the posterior sits. $H$ sets how far the sampler has to compress to get there, which is why the run length grows with $N H$. This plot is what made the role of $H$ click for me.

### Is the error bar right?

A single run that lands 0.3σ from the truth says very little. So `python stage1_calibration.py` repeats the run with many seeds and compares the scatter between runs with the estimated error.

![Calibration](figures/stage1_calibration.png)

| $d$ | Runs | Mean difference from analytic | Scatter between runs | Mean estimated error $\sqrt{H/N}$ |
|---|---|---|---|---|
| 1 | 100 | −0.003 ± 0.005 | 0.053 | 0.047 |
| 2 | 50 | +0.009 ± 0.009 | 0.061 | 0.066 |
| 3 | 20 | +0.008 ± 0.018 | 0.080 | 0.081 |

The mean difference is consistent with zero, and the scatter matches the estimated error to within the precision that this many runs allows.

`simulate_logz` in `nested_sampler.py` gives a second error estimate. Instead of using the mean shrinkage, it draws every $t_i$ from $\mathrm{Beta}(n_i, 1)$ many times and looks at the spread of the resulting $\ln Z$. In 2D it gives 0.067, the same as $\sqrt{H/N}$.

### Limitation

New points are found by rejection sampling from the whole prior. A prior draw lands inside the contour with probability $X$, so each new point costs about $1/X$ likelihood calls, and a whole run costs about $N/X_{\rm final}$. That is 2.3 million calls in 3D, and for this problem it grows by a factor of 4 with each extra dimension. This limitation is the reason for Stage 2.

## Stage 2: BlackJAX Nested Slice Sampling

Files: `blackjax_runner.py`, `stage2_blackjax_gaussian.py`

`blackjax_runner.py` is my run loop around `blackjax.nss`. I based it on the Handley group's BlackJAX nested sampling workshop. It keeps the outer loop, the stopping rule and the bookkeeping in one place, so Stages 2 and 3 can share them.

### Why slice sampling replaces rejection

The outer loop is unchanged: remove the worst live points, add $\mathcal{L}_i w_i$, replace them. Only the replacement step is different.

- **Rejection** draws from the whole prior and hopes to land inside the contour. The cost per point is $1/X$, which grows exponentially as the run goes on.
- **Slice sampling** starts from a live point that is already inside the contour and moves it. It picks a random direction, finds the stretch of that line that lies inside the contour, and jumps to a uniformly chosen point on it. After a few such moves the point has forgotten where it started. The cost per point is roughly the same at any $X$.

Nested Slice Sampling also removes and replaces many points per step. The replacements are independent of each other, so they can be computed in parallel. That is what makes the method suited to GPUs.

### Results

Same Gaussian problem, $N = 400$, 20 points replaced per step, $\max(5, 2d)$ slice moves per new point (`python stage2_blackjax_gaussian.py`):

| $d$ | $\ln Z$ (analytic) | $\ln Z$ (BlackJAX) | Difference | BlackJAX calls | Rejection calls |
|---|---|---|---|---|---|
| 1 | −2.303 | −2.301 ± 0.047 | +0.0σ | 80,639 | 159,651 |
| 2 | −4.605 | −4.755 ± 0.073 | −2.1σ | 95,982 | 710,518 |
| 3 | −6.908 | −6.988 ± 0.085 | −0.9σ | 130,488 | 2,334,175 |
| 5 | −11.513 | −11.546 ± 0.111 | −0.3σ | 270,039 | ~4 × 10⁷ (predicted) |
| 10 | −23.026 | −23.369 ± 0.154 | −2.2σ | 802,409 | ~4 × 10¹⁰ (predicted) |
| 20 | −46.052 | −46.048 ± 0.213 | +0.0σ | 2,491,452 | ~4 × 10¹⁶ (predicted) |

I measured the rejection calls for $d \le 3$ with the Stage 1 sampler. For $d \ge 5$ I predicted them from $N/X_{\rm stop}$ and did not run them.

![Cost against dimension](figures/stage2_cost_vs_dimension.png)

Two of these six single runs are about 2σ low. I didn't want to wave that away, so the script repeats the run 20 times in 2D and in 10D to check whether it is chance or a bias:

| $d$ | Runs | Mean difference from analytic | Scatter between runs | Mean estimated error |
|---|---|---|---|---|
| 2 | 20 | +0.017 ± 0.017 | 0.078 | 0.069 |
| 10 | 20 | −0.013 ± 0.028 | 0.124 | 0.152 |

The mean difference is consistent with zero in both cases, so there is no sign of a bias at this precision. The two 2σ runs were chance.

The BlackJAX error on $\ln Z$ is estimated by simulating the shrinkage factors, the same idea as `simulate_logz` in Stage 1.

## Usage

```bash
pip install -r requirements.txt

# Stage 1
python gaussian_test.py
python stage1_plots.py
python stage1_calibration.py          # a few minutes

# Stage 2
python stage2_blackjax_gaussian.py    # a few minutes

pytest
```

## References

- J. Skilling, *Nested sampling for general Bayesian computation*, Bayesian Analysis 1, 833 (2006).
- D. Yallup, N. Kroupa, W. Handley, *Nested Slice Sampling: Vectorized Nested Sampling for GPU-Accelerated Inference*, arXiv:2601.23252.
- BlackJAX nested sampling workshop, github.com/handley-lab/workshop-blackjax-nested-sampling. The run loop in `blackjax_runner.py` follows it.
