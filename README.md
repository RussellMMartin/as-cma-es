# AS-CMA
Adaptive Sampling CMA-ES (AS-CMA) is a supplement to the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm for optimizing in the presence of noise. 

## Features
When optimizing in the presence of noise, it is often unclear how long each candidate should be evaluated due to a speed-accuracy tradeoff: short evaluations enable faster iteration but are more subject to noise, while long evaluations are slower but more accurate. AS-CMA is a way of systematically identifying how long each candidate should be measured to avoid cost estimates that are too imprecise (which misleads the optimizer) or too precise (which wastes time). We have designed AS-CMA-ES so it can slot into existing optimization code without disrupting the existing framework. Pseudocode for the `AS-CMA-ES` library is as follows: 

```bash
# 1. Initialize AS-CMA once with the constructor
as_cma = AS_CMA_ES(E_t, beta, y_hat, N, verbosity)

while True: 
    # 2. Using CMA-ES, get this generation's candidates to evaluate
    candidates = cma.ask()

    # 3. Using AS-CMA, determine how long each candidate should be evaluated
    sample_times = as_cma.ask_all_sample_times(candidates)

    # 4. Evaluate each candidate for the specified amount of time
    costs = evaluate(candidates, sample_times)

    # 5. After a generation, update AS-CMA and CMA-ES state variables
    as_cma.tell_generation_results(candidates, costs)
    cma.tell_generation_results(candidates, costs)
```

## Usage
To use AS-CMA, we recommend first examining the demo file `.\matlab\test_AS-CMA-ES.m` or `.\python\test_AS-CMA-ES.py`, depending on which language you prefer. There, you can see a barebones implementation of the AS-CMA-ES libary deployed in an optimization loop. For further documentation, see the class definition itself (either `.\matlab\AS-CMA-ES.m` or `.\python\AS-CMA-ES.py`).

To deploy AS-CMA in your own CMA-ES optimization problem, you will need the following information that would not normally be used for CMA-ES:
* $\mathcal{E}(t)$: The sample error (units of percent) vs. sample time (any units). This can be empirically determined, and does not need to be interpolated (AS-CMA does that for you).
* $\hat{y}_{max}$ and $\hat{y}_{min}$: Estimates of the maximum and minimum cost for the first generation. These estimates do not need high accuracy. 
* $\beta$: The signal-to-noise ratio that AS-CMA will try to maintain throughout optimization for every candidate and its nearest neighbor. We recommend using $\beta = 1.3$.

To determine $\mathcal{E}(t)$ empirically, we recommend conducting a handful (approximately 3 to 12) generations of CMA-ES with sample time equal to the maximum desired condition time $t_{max}$. $\mathcal{E}(t)$ can be found for each sample window size $t_i$ by estimating each condition's cost from its time-series data from $0$ to $t_i$, then identifying this estimate's error by comparing to the cost found with a window of size $t_{max}$. The errors for each time window $\epsilon_i$ can then be found by taking the standard deviation of all the errors at each time window (the mean of these errors should center at approximately 0).

# Contact
For questions, suggestions, or issues, please contact me at [rumartin@stanford.edu](mailto:rumartin@stanford.edu). 

# Citation
Keep an eye out for our manuscript:
> Russell M. Martin and Steven H. Collins. Expediting CMA-ES optimization in the presence of noise. Submitted to: *IEEE Transactions on Cybernetics*. 

This work uses the CMA-ES optimization algorithm, described in [Hansen 2016](https://arxiv.org/abs/1604.00772). We use the [`pycma`](https://github.com/CMA-ES/pycma?tab=readme-ov-file) package for CMA-ES in Python and an adaptation of the CMA-ES Matlab algorithm given in [Hansen 2016](https://arxiv.org/abs/1604.00772). 
