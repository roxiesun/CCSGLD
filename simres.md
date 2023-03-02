## Results of simulations of a simple unimodal distribution and the synthetic multi-modal distribution in the CSGLD paper
### 1. Simulation 1: a simple unimodal distribution with known energy density
In this simulation, I used a simple bivariate normal distribution where the two elements are uncorrelated stand normal random variables to show that the density 
estimates for the energy function produced by CCSGLD are not worse than the histogram estimates given by CSGLD. This particular distribution was considered for a 
reason that the analytic form of the energy density can be derived apart from just a discrete version of ground truth (which is also the highest endeavor for the 
histogram estimates achived by CSGLD). That is, with $\pi(\mathbf{x}) = \frac{1}{2\pi}\exp(-\frac{x(1)^2 + x(2)^2}{2})$ and $\mathbf{x} = (x(1), x(2))$, the energy 
function
$$U(\mathbf{x}) = \log(2\pi) + \frac{1}{2}(x(1)^2 + x(2)^2)$$
follows a shifted gamma distribution with a probability density function of the form
$$f_U(u) = \frac{2^{1/2}}{\Gamma(1/2)}(u - \log 2\pi)^{1/2 - 1}\exp{-\frac{u-\log 2\pi}{2}},  u \in (\log 2\pi, \infty).$$
The stochastic gradient noise was assumed following a Laplace distribution with a scale parameter of $0.32/\sqrt{2}$. 
A Cauchy kernel was considered in CCSGLD together with a minibatch size of 100. 
For the hyperparameters, I set a lattice with 245 grid points for the kernel density estimation in CCSGLD and correspondingly 244 energy subregions in CSGLD. 
For both methods, the algorithm was run for $8\times 10^5$ iterations with the first $15,000$ as the warm-up. 
Following thr CSGLD paper, I fixed the temperature $\tau$ at 1.0, the learning rate $\varepsilon$ at $0.001$, and set the step size for stochastic approximation 
as $\omega_t = \frac{10}{t^{0.8}+100}$. The bandwidth of kernel density estimation in CCSGLD was set in a non-increasing pattern that also 
accounted for a rule of thumb in deconvolution kernel density estimation, that is, 
$h^{(t)} = \sqrt{\min\left(\delta_t, 2(\hat{\sigma}^{(t)})^2\log n\right)}$, where $\delta_t = \frac{\rho\kappa}{\max(\kappa, t)}$ 
with $\rho = 20$ and $\kappa = 1,000$.  Figures below shows the sampling history and the contour plot of the true sampling distribution and those of samples produced by
CSGLD/CCSGLD.


<!--![ccmc6e5g245](https://github.com/roxiesun/ccmc/blob/main/images/ccmc6e5g245new.gif)-->
<img src="/images/ccmc6e5g245new.gif" width="75%" height="75%"/>


Similar results were obtained if the lattice size is increased to 489 (Figure below). 
I'm not sure if this wiggling problem will always occur as the iteration goes up and the bandwidth gets increasingly small.

<!--![ccmc6e5g489](https://github.com/roxiesun/ccmc/blob/main/images/ccmc6e5g489new.gif)-->
<img src="/images/ccmc6e5g489new.gif" width="75%" height="75%"/>

Another notable problem is that, although the histogram of the $x_2$ samples seems close to uniform, samples near the boundaries were drawn with slightly higher frequency even if the desired sampling distribution $\mathbf{\pi}$ was set as uniform. 


### 2. CMC for estimating the histogram of the marginal density of $x_2$
I also tried the CMC algorithm to get a histogram estimate for the marginal density of $x_2$. Figure (a) below shows the true histogram of this marginal density together with the CMC estimates across $10^6$ iterations while Figure (b) is the CMC estimates only. The number of subregions was set to 244 according to the lattice used by ccmc, and parameters like $\delta_t, M, \rho,$ and $\kappa$ were all set the same. I added an additional assumption $\sum_i\widehat{g_i}^{(str)} = 1$ to control the size of $\widehat{g_i}^{(itr)}$.

It seems that  $\widehat{g_i}^{(itr)} \propto \int_{E_i}\psi(\mathbf{x})d\mathbf{x}$ holds when $itr$ gets large, and the histogram of $x_2$ samples are closed to uniform as shown in (c) below.


<!--![cmc1e6m244](https://github.com/roxiesun/ccmc/blob/main/images/cmc1e6m244.gif)-->
<img src="/images/cmc1e6m244.gif" width="70%" height="70%"/>


If the additional contraint is set as $\sum_i\widehat{g_i}^{(itr)} = 4$, then the estimated histogram in (a) gets much closer to the truth at around $3\times10^5$ iterations but then exceeds the true bars in subregions around $x_2 = 25$ as shown in the Figure below. To be honest, how to set such a constraint still remains a question to me.


<!--![cmc1e6m244c4](https://github.com/roxiesun/ccmc/blob/main/images/cmc1e6m244c4.gif)-->
<img src="/images/cmc1e6m244c4.gif" width="70%" height="70%"/>


### 3.SAMC for estimating the histogram of the marginal density of $x_2$
I'm not sure if I'm doing this right, but it seems to me that the SAMC and CMC algorithm differs only in the number of samples $\mathbf{x}_k^{(t)}$ $(k = 1,\dots, M)$ drawn in the sampling step (i.e., $M = 10$ or $1$) in estimating histogram of the marginal distribution. The figure below gives the true vs. estimated histogram, and the histogram of the $x_2$ samples obtained by SAMC.

The estimated $\widehat{g_i}^{(t)}$ seems less proportional to the true histogram than those obtained from CMC as shown in (a), while histogram of the $x_2$ samples is still close to uniform as shown in (c). 


<!--![samc1e6m244](https://github.com/roxiesun/ccmc/blob/main/images/samc1e6m244.gif)-->


<img src="/images/samc1e6m244.gif" width="70%" height="70%"/>
