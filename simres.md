## Results of simulations of a simple unimodal distribution and the synthetic multi-modal distribution in the CSGLD paper
### 1. Simulation 1: a simple unimodal distribution with known energy density
In this simulation, I used a simple bivariate normal distribution where the two elements are uncorrelated stand normal random variables to show that the density 
estimates for the energy function produced by CCSGLD are not too worse than the histogram estimates given by CSGLD in terms of fitting the discretized ground truth. This particular distribution was considered for a 
reason that the analytic form of the energy density can be derived apart from just the discrete version of ground truth (which is also the highest endeavor for the 
histogram estimates achived by CSGLD). That is, with $\pi(\mathbf{x}) = \frac{1}{2\pi}\exp(-\frac{x(1)^2 + x(2)^2}{2})$ and $\mathbf{x} = (x(1), x(2))$, the energy 
function
$$U(\mathbf{x}) = \log(2\pi) + \frac{1}{2}(x(1)^2 + x(2)^2)$$
follows a shifted gamma distribution with a probability density function of the form
$$f_U(u) = \frac{2^{1/2}}{\Gamma(1/2)}(u - \log 2\pi)^{1/2 - 1}\exp{-\frac{u-\log 2\pi}{2}},  u \in (\log 2\pi, \infty).$$
The stochastic gradient noise was assumed following a Laplace distribution with a scale parameter of $0.32/\sqrt{2}$. 
A Cauchy kernel was considered in CCSGLD together with a minibatch size of $100$. 
For the hyperparameters, I set a lattice with $245$ grid points for the kernel density estimation in CCSGLD and correspondingly $244$ energy subregions in CSGLD. 
For both methods, the algorithm was run for $8\times 10^5$ iterations with the first $15,000$ as the warm-up. 
Following [Deng et al. (2020)](https://proceedings.neurips.cc/paper/2020/hash/b5b8c484824d8a06f4f3d570bc420313-Abstract.html), I fixed the temperature $\tau$ at $1.0$, $\zeta$ at $0.75$, the learning rate $\varepsilon$ at $0.001$, and set the step size for stochastic approximation 
as $\omega_t = \frac{10}{t^{0.8}+100}$. The bandwidth of kernel density estimation in CCSGLD was set in a non-increasing pattern that also 
accounted for a rule of thumb in deconvolution kernel density estimation, that is, 
$h^{(t)} = \sqrt{\min\left(\delta_t, 2(\hat{\sigma}^{(t)})^2\log n\right)}$, where $\delta_t = \frac{\rho\kappa}{\max(\kappa, t)}$ 
with $\rho = 20$ and $\kappa = 1,000$.  Figures below shows the sampling history and the contour plot of the true sampling distribution and those of samples produced by CSGLD/CCSGLD. 

**(Sampling history)**

<!--![simpleU](https://github.com/roxiesun/CCSGLD/blob/main/images/csgld_contour_0227SimpleULaplacestepsize10.gif)-->
<img src="/images/csgld_contour_0227SimpleULaplacestepsize10.gif" width="90%" height="90%"/>

**(Contour)**

<!--![simpleUContour](https://github.com/roxiesun/CCSGLD/blob/main/images/0227Contour_SimpleULaplaceStep10.png)-->
<img src="/images//0227Contour_SimpleULaplaceStep10.png" width="90%" height="90%"/>


It seems that CCSGLD performs a bit worse in sampling from the target distribution or in estimating the energy density. But what I am trying to confirm through this simulation is that the discretized version of ground truth (the black curves in (c) and (d) of the first figure above ), which is the ground truth for the histogram density estimates of CSGLD, actually deviates from the true energy density (the blue curves in the first figure above). Hence for a more complex target distribution where the density function of  $U(\mathbf{x})$ has no analytic form, what we can plot is just the discretized ground truth, and I consider it reasonble if the energy density estimates of CCSGLD across iterations cannot fit the black curves better than the histogram estimates of CSGLD.

I also tried another measure of the sampling performace suggested by [Nemeth and Fearnhead (2021)](https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1847120). We can measure the accuracy of the algorithms by estimating the expectation of a test function $\varphi(\mathbf{x}) = x(1) + x(2)$, where $E_\pi[\varphi(\mathbf{x})] = \int \pi(\mathbf{x})\varphi(\mathbf{x})d\mathbf{x} = 0$ for this simple case. I took a look at the bias and RMSE of the estimators $\frac{1}{T}\displaystyle\sum\limits_{t=1}^{T}\varphi(\mathbf{x}^{(t)})$ for both approaches. Based on a single replication, the bias for CSGLD and CCSGLD is $-0.024$ and $-0.016$, respectively, while the RMSE is $0.932$ and $0.970$, respectively. This measure was also considered in the following simulation.


### 2. Simulation 2: the synthetic multi-modal distribution in [Deng et al. (2020)](https://proceedings.neurips.cc/paper/2020/hash/b5b8c484824d8a06f4f3d570bc420313-Abstract.html)
Now we aim at simulating from the distribution $\pi(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$ with
$$U(\mathbf{x}) = \sum_{i=1}^2\frac{x(i)^2 - 10\cos(1.2\pi x(i))}{3}$$
and $\mathbf{x} = (x(1),x(2))$. The distribution contains nine important modes where the center one has the largest probability mass and the four on the corners have the smallest. I considered two setups to deal with the stochastic gradient noise by assuming (i) $\epsilon_k \overset{i.i.d}{\sim} N(0, 0.32^2)$ or (ii) $\epsilon_k \overset{i.i.d}{\sim} Laplace(0.32/\sqrt{2})$. A second-order kernel whose characteristic function has a compact and symmetric support was adopted for the former case while a Cauchy kernel was considered for the latter. The energy lattice/subregions and hyperparameters were set as in Simulation 1. I ran five replications under each setting.

**(Setting i)**

The figure below gives the sampling history and density estimates across iterations based on the fifth replication under setting (i) with super smooth normal stochastic gradient noise.

<!--![Normal_1](https://github.com/roxiesun/CCSGLD/blob/main/images/Rep5_Nor2ndKStepsize10Sigma032n100.gif)-->
<img src="/images/Rep5_Nor2ndKStepsize10Sigma032n100.gif" width="90%" height="90%"/>

The figure below gives the contour plot of the true target sampling distribution and those of samples produced by CSGLD/CCSGLD based on the fifth replication under setting (i) with super smooth normal stochastic gradient noise.

<!--![Normal_1_contour](https://github.com/roxiesun/CCSGLD/blob/main/images/Rep5_Contour_Nor2ndKStep10Sig032.png)-->
<img src="/images/Rep5_Contour_Nor2ndKStep10Sig032.png" width="90%" height="90%"/>

Under setting (i), the average bias of estimating the expected test function $E_\pi\varphi$ for CSGLD and CCSGLD is $0.087$ and $0.137$, respectively; and the average RMSE are $1.578$ and $1.570$, respectively. It appears that CCSGLD does not perform better, possibly because only $4 times 10^5$ was run for this setting due to time cost.

**(Setting ii)**

The figure below gives the sampling history and density estimates across iterations based on the first replication under setting (ii) with ordinal smooth Laplace stochastic gradient noise.

<!--![Laplace_1](https://github.com/roxiesun/CCSGLD/blob/main/images/Rep1_LapCauStepsize10Sigma032n100.gif)-->
<img src="/images/Rep1_LapCauStepsize10Sigma032n100.gif" width="90%" height="90%"/>

The figure below gives the contour plot of the true target sampling distribution and those of samples produced by CSGLD/CCSGLD based on the first replication under setting (ii) with ordinal smooth Laplace stochastic gradient noise.

<!--![Laplace_1_contour](https://github.com/roxiesun/CCSGLD/blob/main/images/Rep1_Contour_LapCauStep10Sig032n100.png)-->
<img src="/images/Rep1_Contour_LapCauStep10Sig032n100.png" width="90%" height="90%"/>

Under setting (ii), the average bias of estimating the expected test function $E_\pi\varphi$ for CSGLD and CCSGLD is $0.129$ and $0.092$, respectively; and the average RMSE are $1.564$ and $1.519$, respectively. In that sense, CCSGLD performs slightly better than CSGLD under this setting. But a problem is that CCSGLD is more time-consuming than CSGLD: running a single replication of $8\times 10^5$ iterations takes around 2.5 hours.
