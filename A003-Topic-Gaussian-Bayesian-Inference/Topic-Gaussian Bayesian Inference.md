#Machine_Learning #Gaussian_Process #Bayesian_Inference 

---
- Author: ApeironLi
- Version: 2023.7.25-1.0
---
## 1. Problem Definition
- <u>Dataset</u>:
Input Dataset (Deterministic Constant) $X=[\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_N]^T \in \mathbb{R}^{N\times M}$,
Output Dataset (Deterministic Constant) $Y=[\mathbf{y}_1,\mathbf{y}_2,...,\mathbf{y}_N]^T \in \mathbb{R}^{N\times O}$.
- <u>Task</u>:
Test Input (Deterministic Variable)  $\mathbf{x}\in\mathbb{R}^{1\times M}$,
Test Output (Deterministic/Stochastic Variable)  $\mathbf{y}\in\mathbb{R}^{1\times O}$.
- <u>Feature Function</u>:
$$\phi_\mathbf{x}=\phi(\mathbf{x})\in \mathbb{R}^F,\phi_X=\phi(X)\in \mathbb{R}^{N\cdot F},$$
$F$ - Feature amount, can be infinity!
e.g. $\phi(\mathbf{x})=[1,\mathbf{x},\mathbf{x}^2,\mathbf{x}^3]^T$
$\phi(x)=[\mathbf{x},\mathbf{x},\cdots,\mathbf{x}],||\phi(\mathbf{x})||=\infty$ (for Gaussian Process)
- <u>Weight</u>: Stochastic Parameter
$$
W=[\mathbf{w}_1,\mathbf{w}_2,...,\mathbf{w}_F]^T\in\mathbb{R}^{F\times O}.
$$
- <u>Latent Function and Input Output Realtionship</u>:
$$
\begin{align*}
f_\mathbf{x}&=\phi_\mathbf{x}^T\cdot\mathbf{w}
\\
\mathbf{y}&=\phi_\mathbf{x}^T\cdot \mathbf{w}+\lambda\\
&\lambda \sim \mathcal{N}(\lambda;0,\Lambda)
\end{align*}
$$
---
## 2. Vallina Gaussian Inference (Gaussian Regression)
---
Bayesian Inference:
$$
\begin{align*}
p(\mathbf{w}|X,Y)&=\frac{p(Y|\mathbf{w},X)p(\mathbf{w}|X)}{p(Y|X)}\\
&=\frac{p(Y|\mathbf{w},X)p(\mathbf{w})}{p(Y)}\\
&=\frac{p(Y|\mathbf{w},X)p(\mathbf{w})}{\int p(Y|\mathbf{w},X)p(\mathbf{w})d\mathbf{w}}\\

p(f_X|X,Y)&=\frac{p(Y|f_X,X)p(f_X|X)}{p(Y|X)}\\
&=\frac{p(Y|f_X,X)p(f_X)}{p(Y)}\\
&=\frac{p(Y|f_X,X)p(f_X)}{\int p(Y|f_X,X)p(f_X)df_X}
\end{align*}
$$
Note:
- Why not $p(\mathbf{w}|X,Y)=\frac{p(X,Y|\mathbf{w})p(\mathbf{w})}{p(X,Y)}$?
- Why $p(\mathbf{w}|X)=p(\mathbf{w})$?
- Why $p(Y|X)=p(Y)$?
- Why $p(Y)=\int p(Y|\mathbf{w},X)p(\mathbf{w})d\mathbf{w}$, and what is it?
---
- <u>Prior</u>:
$$
\begin{align*}
p(\mathbf{w})&=\mathcal{N}(\mathbf{w};\mathbf{\mu},\Sigma)\\
p(f_X)&=\mathcal{N}(f_X;\phi_X^T\mathbf{\mu},\phi_X^T\Sigma\phi_X)
\end{align*}
$$
- <u>Likelihood</u>:
$$
\begin{align*}
&Y|\mathbf{w}=(\phi_X^T\mathbf{w}+\lambda)\sim\mathcal{N}(Y;\phi_X^T\mathbf{w},\Lambda)\\
&=>p(Y|\mathbf{w})=\mathcal{N}(\phi_X^T\mathbf{w};Y,\Lambda)\\
&p(Y|f_X)=\mathcal{N}(f_X;Y,\Lambda)
\end{align*}
$$
- <u>Evidence</u>:
$$
p(Y)=\mathcal{N}(Y;\phi_X^T{\mu},\Lambda+\phi_X^T\Sigma \phi_X)
$$
- <u>Posterior</u>:
$$
\begin{align*}
p(\mathbf{w}|X,Y)=\mathcal{N}(\mathbf{w};\mu+&\Sigma\phi_X(\phi_X^T\Sigma\phi_X+\Lambda)^{-1}(Y-\phi_X^T\mu),\\
&\Sigma-\Sigma\phi_X(\phi_X^T\Sigma\phi_X+\Lambda)^{-1}\phi_X^T\Sigma)\\
p(f_X|X,Y)=\mathcal{N}(f_X;\phi_X^T\mu+&\phi_X^T\Sigma\phi_X(\phi_X^T\Sigma\phi_X+\Lambda)^{-1}(Y-\phi_X^T\mu),\\&\phi_X^T\Sigma\phi_X-\phi_X^T\Sigma\phi_X(\phi_X^T\Sigma\phi_X+\Lambda)^{-1}\phi_X^T\Sigma\phi_X)
\end{align*}
$$
- <u>Predictive Posterior</u>:
$$
\begin{align*}
&\because f_X=\phi_X^T\mathbf{w},f_\mathbf{x}=\phi_\mathbf{x}^T\mathbf{w}
\\&\therefore f_\mathbf{x}=\phi_\mathbf{x}^T(\phi_\mathbf{X}^T)^{-1}f_X
\end{align*}
$$
Form.1.
$$
\begin{align*}
p(f_\mathbf{x}|X,Y,\mathbf{x})
&=p(f_\mathbf{x}|f_X,Y,\mathbf{x})\\
&=\mathcal{N}(f_\mathbf{x};\phi_\mathbf{x}^T\mu+\phi_\mathbf{x}^T\Sigma\phi_X(\phi_X^T\Sigma\phi_X+\Lambda)^{-1}(Y-\phi_X^T\mu),\\
&\phi_\mathbf{x}^T\Sigma\phi_\mathbf{x}-\phi_\mathbf{x}^T\Sigma\phi_X(\phi_X^T\Sigma\phi_X+\Lambda)^{-1}\phi_X^T\Sigma\phi_\mathbf{x})\\
\end{align*}
$$
Form.2.
$$
\begin{align*}
p(f_\mathbf{x}|f_X,\mathbf{x})&=\mathcal{N}(f_\mathbf{x};\phi_\mathbf{x}^T\mu+\phi_\mathbf{x}^T\Sigma\phi_X(\phi_X^T\Sigma\phi_X+\Lambda)^{-1}(f_X-\phi_X^T\mu),\\
&\phi_\mathbf{x}^T\Sigma\phi_\mathbf{x}-\phi_\mathbf{x}^T\Sigma\phi_X(\phi_X^T\Sigma\phi_X+\Lambda)^{-1}\phi_X^T\Sigma\phi_\mathbf{x}+\Lambda)\\
\end{align*}
$$

## 3. (Gaussian) Hierarchical Bayesian Inference
---
Automatically Learn Feature Functions!
- Feature Function and Latent Function:
$$
\begin{align*}
\phi_\mathbf{x}&=\phi(\mathbf{x};\theta)\\
f_\mathbf{x}&=\phi^T(\mathbf{x};\theta)\cdot \mathbf{w}
\end{align*}
$$
- Learning Objective:
$$
p(\mathbf{w},\theta|X,Y)
$$
- Vanilla Bayesian Inference, fails!
$$
\begin{align*}
p(\mathbf{w},\theta|X,Y)&=\frac{p(Y|\mathbf{w},\theta,X)p(\mathbf{w},\theta|X)}{p(Y|X)}\\
&=\frac{p(Y|\mathbf{w},\theta,X)p(\mathbf{w},\theta)}{p(Y)}\\
p(Y|\mathbf{w},X,\theta)&=\mathcal{N}(\mathbf{w};{\phi_X^{\theta}}^T\mu,\Lambda)
\end{align*}
$$
However, the likelihood becomes a non-linear function of parameter $\theta$, linear projection rule can't be used! We have to acquire <u>Type-2 Maximum Likelihood</u>.

---
- Type-2 Maximum Likelihood: Maximize $p(Y|X,\theta)$ to estimate $\theta$, then do full bayesian estimate to $W$.
$$
\begin{align*}
\hat{\theta} &= \underset{\theta}{\mathrm{argmax}}p(Y|X,\theta)\\
&= \underset{\theta}{\mathrm{argmax}}\mathcal{N}(Y;{\phi_X^{\theta}}^T\mu,{\phi_X^{\theta}}^T\Sigma{\phi_X^{\theta}}+\Lambda)\\ 
&= \underset{\theta}{\mathrm{argmin}}-log\mathcal{N}(Y;{\phi_X^{\theta}}^T\mu,{\phi_X^{\theta}}^T\Sigma{\phi_X^{\theta}}+\Lambda)\\
&=\underset{\theta}{\mathrm{argmin}}\frac{1}{2}((Y-{\phi_X^{\theta}}^T\mu)^T({\phi_X^{\theta}}^T\Sigma{\phi_X^{\theta}}+\Lambda)^{-1}(Y-{\phi_X^{\theta}}^T\mu)+log|{\phi_X^{\theta}}^T\Sigma{\phi_X^{\theta}}+\Lambda|)+\frac{N}{2}log2\pi
\end{align*}
$$
---
## 4. Gaussian Process
Infinite amount of features:
- Feature Function:
$$
\phi_\mathbf{x}=[\mathbf{x},\mathbf{x},...,\mathbf{x}]^T \in \mathbb{R}^{\infty\times M}
$$
- Mean Function:
$$
\begin{align*}
m_\mathbf{x}=\phi_\mathbf{x}^T\mu\in\mathbb{R}^{1\times O}\\
m_X=\phi_X^T\mu\in\mathbb{R}^{N\times O}
\end{align*}
$$
- Kernel:
$$
\begin{align*}
K_{\mathbf{x}X}=\phi_\mathbf{x}^T\Sigma\phi_X\in\mathbb{R}^{N\times M\times M}\\
K_{X\mathbf{x}}=\phi_X^T\Sigma\phi_\mathbf{x}\in\mathbb{R}^{M\times M\times N}\\
K_{\mathbf{x}\mathbf{x}}=\phi_\mathbf{x}^T\Sigma\phi_\mathbf{x}\in\mathbb{R}^{M\times M}\\
K_{XX}=\phi_X^T\Sigma\phi_X\in\mathbb{R}^{M\times M}
\end{align*}$$
---
e.g. <u>Gaussian Feature and Gaussian Kernel</u>
- Feature Function:
$$\phi_l(\mathbf{x})=\mathrm{exp}(-\frac{(\mathbf{x}-c_l)^2}{2\lambda^2})$$
$l\in[0,F],F\to\infty$ means the $l-th$ element of feature vector $\phi_l(\mathbf{x})$.
$$
K_{\mathbf{x}\mathbf{x}}^{ij}=\phi(x_i)^T\Sigma\phi(x_j)
$$
$$
\begin{align*}
\phi(x_i)^T\Sigma\phi(x_j)&=\frac{\sigma^2(c_{max}-c_{min})}{F}\Sigma^{F\to\infty}_{l=1}\mathrm{exp}(-\frac{(x_i-c_l)^2}{2\lambda^2})\mathrm{exp}(-\frac{(x_j-c_l)^2}{2\lambda^2}) \\
&=\frac{\sigma^2(c_{max}-c_{min})}{F}\mathrm{exp}(-\frac{(x_i-x_j)^2}{4\lambda^2})\Sigma^{F\to\infty}_{l=1}\mathrm{exp}(-\frac{(c_l-\frac{1}{2}(x_i+x_j))^2}{\lambda^2})
\end{align*}
$$
$x_i$ and $x_j$ are the $i-th$ and $j-th$ element of input vector $x\in \mathbb{R}^M$.
Now increase $F$ so # of features in $\delta c$ approaches $\frac{F\delta c}{c_{max}-c_{min}}$. (Regularly increase feature numbers in each small interval.) 
We turn a Infinite summable series to a Riemann Integral with closed analytic form:
$$\begin{align*}
\phi(x_i)^T\Sigma\phi(x_j)&\to\sigma^2\mathrm{exp}(-\frac{(x_i-x_j)^2}{4\lambda^2})\int_{c_{min}}^{c_{max}}\mathrm{exp}(-\frac{(c-\frac{1}{2}(x_i+x_j))^2}{\lambda^2})dc\\
&=\sqrt{2\pi}\lambda\sigma^2\mathrm{exp}(-\frac{(x_i-x_j)^2}{4\lambda^2})
\end{align*}
$$
---
 - <u>Prior</u>:
$$
\begin{align*}
p(f_X)&=\mathcal{N}(f_X;m_X,K_{XX})
\end{align*}
$$
- <u>Likelihood</u>:
$$
\begin{align*}
p(Y|f_X)=\mathcal{N}(f_X;Y,\Lambda)
\end{align*}
$$
- <u>Evidence</u>:
$$
p(Y)=\mathcal{N}(Y;m_X,\Lambda+K_{XX})
$$
- <u>Posterior</u>:
$$
\begin{align*}
p(f_X|X,Y)=\mathcal{N}(f_X;m_X+&K_{XX}(K_{XX}+\Lambda)^{-1}(Y-m_X),\\&K_{XX}-K_{XX}(K_{XX}+\Lambda)^{-1}K_{XX})
\end{align*}
$$
- <u>Predictive Posterior</u>:
$$
\begin{align*}
&\because f_X=\phi_X^T\mathbf{w},f_\mathbf{x}=\phi_\mathbf{x}^T\mathbf{w}
\\&\therefore f_\mathbf{x}=\phi_\mathbf{x}^T(\phi_\mathbf{X}^T)^{-1}f_X
\end{align*}
$$
Form.1.
$$
\begin{align*}
p(f_\mathbf{x}|X,Y,\mathbf{x})
&=p(f_\mathbf{x}|f_X,Y,\mathbf{x})\\
&=\mathcal{N}(f_\mathbf{x};m_\mathbf{x}+K_{\mathbf{x}X}(K_{XX}+\Lambda)^{-1}(Y-m_X),\\
&K_{\mathbf{x}\mathbf{x}}-K_{\mathbf{x}X}(K_{XX}+\Lambda)^{-1}K_{X\mathbf{x}})\\
\end{align*}
$$
Form.2.
$$
\begin{align*}
p(f_\mathbf{x}|X,\mathbf{x})
&=\mathcal{N}(f_\mathbf{x};m_\mathbf{x}+K_{\mathbf{x}X}(K_{XX}+\Lambda)^{-1}(f_X-m_X),\\
&K_{\mathbf{x}\mathbf{x}}-K_{\mathbf{x}X}(K_{XX}+\Lambda)^{-1}K_{X\mathbf{x}})\\
\end{align*}
$$
---
## 5. Gaussian Process Classification
- <u>Link Function and Latent Function</u>:
$$
\begin{align*}
f_\mathbf{x}&=\phi_\mathbf{x}^T\cdot\mathbf{w}
\\
\mathbf{y}&=\sigma(\phi_\mathbf{x}^T\cdot \mathbf{w}+\lambda)\in [0,1]^{N\times O} \\
&\lambda \sim \mathcal{N}(\lambda;0,\Lambda)
\end{align*}
$$
- Vanilla Bayesian Inference, computationally intractable!
$$
\begin{align*}
p(f_X|X,Y)&=\frac{p(Y|f_X)p(f_X)}{p(Y|X)}\\&
=\frac{\Pi_{i=1}^{n}\sigma(\mathcal{N}(f_{X_i};Y_i,\Lambda))\mathcal{N}(f_X;m_{X},K_{XX})}{p(Y)}
\end{align*}
$$
---
- <u>Laplace Approximation</u>, consider a complex distribution $p(\theta)$:
1. Find a (local) maximum of $\mathrm{log}p(\theta)$:
$$
\hat{\theta}=\mathrm{argmax}\mathrm{log}p(\theta),i.e. \nabla\mathrm{log}p(\hat{\theta})=0
$$
2. Calculate the second order derivative of $\mathrm{log}p(\theta)$: For convenience, we can perform second order Taylor expansion around $\theta=\hat{\theta}+\delta$ in log space:
$$
\mathrm{log}p(\delta)=\mathrm{log}p(\hat{\theta})+\frac{1}{2}\delta^T(\nabla\nabla^T\mathrm{logp}(\hat{\theta}))\delta+\mathcal{O}(\delta^3)
$$
3. The Laplace Approximation:
$$
q(\theta)=\mathcal{N}(\theta,\hat{\theta},-\Phi^{-1})
,\Phi=\nabla\nabla^T\mathrm{logp}(\hat{\theta})$$
when $\delta \to 0$, $p(\theta) \to q(\theta)$.

---
- <u>Laplace Approximation of GP Classification</u>:
1. Do Laplace Approximation to the intractable gaussian posterior:
$$
\begin{align*}
\hat{f_X}&=\mathrm{argmax}\log p(f_X|X,Y)\\
q(f_X|Y,X)&=\mathcal{N}(f_X;\hat{f},-(\nabla\nabla^T\mathrm{log}p(f_X|Y)|_{f_X=\hat{f}})^{-1})\\
&=\mathcal{N}(f_X;\hat{f},\hat{\Sigma})
\end{align*}
$$
2. Calculate Predictive Posterior:
$$
\begin{align*}
q(f_\mathbf{x}|X,Y,\mathbf{x})&=\int p(f_\mathbf{x}|f_X,Y,\mathbf{x})q(f_X|X,Y,\mathbf{x})df_{X}\\
&\approx\int p(f_\mathbf{x}|f_X,\mathbf{x})q(f_X|X,Y)df_{X}\\
&=\int\mathcal{N}(f_x;m_x+k_{xX}K_{XX}^{-1}(f_X-m_X),k_{xx}-k_{xX}K_{XX}^{-1}k_{Xx})q(f_{X}|X,Y)df_{X}\\
&=\mathcal{N}(f_x;m_x+k_{xX}K_{XX}^{-1}(\hat{f}-m_X),k_{xx}-k_{xX}K_{XX}^{-1}k_{Xx}+k_{xX}K_{XX}^{-1}\hat{\Sigma}K_{XX}^{-1}k_{Xx})
\end{align*}
$$
Special Note: Why $p(f_\mathbf{x}|f_X,\mathbf{x})$ instead of $p(f_\mathbf{x}|f_X,Y,\mathbf{x})$?
How to take advantage from both Laplace approximation and the linear algebra conclusion from bayesian inference?

