#Machine_Learning #Gaussian_Process #Bayesian_Inference 

---
- Author: ApeironLi
- Version: 2023.7.24-1.0
---
Note: All $\int$ in this article refers to $\int_{-\infty}^{\infty}$.

### 0. Subtle Difference between Concepts
- Stochastic (random) and Deterministic.
- Parameter, Variable and Constant.

### 1. Gaussian PDF: 
$$
\mathcal{N}(\mathbf{x};\mathbf{\mu},\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\mathrm{exp}(-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^T\Sigma^{-1}(\mathbf{x}-\mathbf{\mu})),\mathbf{x},\mathbf{\mu}\in\mathbb{R}^n,\Sigma\in\mathbb{R}^n.
$$
-  $\Sigma$ must be symmetric positive definite.
- Symmetry in $\mathbf{x}$ and $\mathbf{\mu}$: $\mathcal{N}(\mathbf{x};\mathbf{\mu},\Sigma)=\mathcal{N}(\mathbf{\mu};\mathbf{x},\Sigma)$.
- Exponential of a quadratic polynomial:
$$
\begin{align*}
\mathcal{N}(\mathbf{x};\mathbf{\mu},\Sigma)&=\mathrm{exp}(a+\mathbf{\eta}^T\mathbf{x}-\frac{1}{2}\mathbf{x}^T\Lambda\mathbf{x})\\
&=\mathrm{exp}(a+\mathbf{\eta}^T\mathbf{x}-\frac{1}{2}\mathrm{tr}(\mathbf{x}\mathbf{x}^T\Lambda))
\end{align*}
$$
with $\Lambda=\Sigma^{-1}$ (precision), $\mathbf{\eta}=\Lambda^T\mathbf{\mu}$.
### 2. Gaussian Calculation Rule:
- Product of Gaussians are Gaussians:
$$
\begin{align*}
\mathcal{N}(\mathbf{x};\mathbf{a},A)\mathcal{N}(\mathbf{x};\mathbf{b},B)=\mathcal{N}(\mathbf{x};\mathbf{c},C)Z\\
C=(A^{-1}+B^{-1})^{-1}\\
\mathbf{c}=C(A^{-1}\mathbf{a}+B^{-1}\mathbf{b})\\
Z=\mathcal{N}(\mathbf{a};\mathbf{b},A+B)
\end{align*}
$$
- Linear Projections of Gaussians are Gaussians:
$$
\begin{align*}
p(\mathbf{x})&=\mathcal{N}(\mathbf{x};\mathbf{\mu},\Sigma)\\
p(A\mathbf{x})&=\mathcal{N}(\mathbf{x};A\mathbf{\mu},A\Sigma A^T)
\end{align*}
$$
- Noisy Linear Projections of Gaussians are Gaussians:
$$
\begin{align*}
\mathrm{for.} p(x)&=\mathcal{N}(x;m,V),p(z|x)=\mathcal{N}(z;Ax,B)\\
p(z)&=\int p(z|x)p(x)dx=\mathcal{N}(z;Am,AVA^T+B)
\end{align*}
$$
- Marginals of Gaussians are Gaussians:
$$
\int_{-\infty}^{\infty}\mathcal{N}
\left[
\left(
\begin{matrix}
\mathbf{x}\\\mathbf{y}
\end{matrix}
\right)
;
\left(
\begin{matrix}
\mathbf{\mu}_\mathbf{x}\\
\mathbf{\mu}_\mathbf{y}
\end{matrix}
\right)
,
\left(
\begin{matrix}
\Sigma_\mathbf{xx} & \Sigma_\mathbf{xy}\\
\Sigma_\mathbf{yx} & \Sigma_\mathbf{yy}
\end{matrix}
\right)
\right]
d\mathbf{y}
=
\mathcal{N}(
\begin{bmatrix}
1\\0
\end{bmatrix}
\begin{bmatrix}
\mathbf{x}\\\mathbf{y}
\end{bmatrix}
)
=\mathcal{N}(\mathbf{x};\mathbf{\mu}_\mathbf{x},\Sigma_\mathbf{xx})
$$
### 3. Bayesian Inference for Gaussian PDF:

- <u>For a Linear Parametric Model with Gaussian Noise</u>:
$$
\begin{align*}
\mathbf{y}=\phi_\mathbf{x}^T\mathbf{w}+\lambda
\\
\lambda\sim\mathcal{N}(\lambda;0,\Lambda)
\end{align*}
$$
$\phi_\mathbf{x}$ (feature function) is the deterministic variables;
$\mathbf{w}$ is the stochastic parameter;
$\lambda$ is the stochastic variable;
$\mathbf{y}$ can be both deterministic and stochastic variables.

- <u>Prior</u>:
$$
p(\mathbf{w})=\mathcal{N}(\mathbf{w};\mathbf{\mu},\Sigma)
$$
- <u>Likelihood</u>:
$$
\mathbf{y}|\mathbf{w}=(\phi_\mathbf{x}^T\mathbf{w}+\lambda)\sim\mathcal{N}(\mathbf{y};\phi_\mathbf{x}^T\mathbf{w},\Lambda)
$$
In this situation (Conditioned on $\mathbf{w}$), $\mathbf{w}$ is deterministic and can occur in the mean and covariance matrix of Gaussian PDF.
- <u>Evidence</u>:
$$
\begin{align*}
p(\mathbf{y})&=\int p(\mathbf{w})p(\mathbf{y}|\mathbf{w})d\mathbf{w}\\
&=\int\mathcal{N}(\mathbf{w};\mathbf{\mu},\Sigma)\mathcal{N}(\mathbf{y};\phi_\mathbf{x}^T\mathbf{w},\Lambda)d\mathbf{w}\\
&=\int \mathcal{N}(\mathbf{y};\phi_\mathbf{x}^T{\mu},\Lambda+\phi_\mathbf{x}^T\Sigma \phi_\mathbf{x})
\mathcal{N}(\mathbf{w};.)d\mathbf{w}\\
&=\mathcal{N}(\mathbf{y};\phi_\mathbf{x}^T{\mu},\Lambda+\phi_\mathbf{x}^T\Sigma \phi_\mathbf{x}) \int \mathcal{N}(\mathbf{w};.)d\mathbf{w}\\
&=\mathcal{N}(\mathbf{y};\phi_\mathbf{x}^T{\mu},\Lambda+\phi_\mathbf{x}^T\Sigma \phi_\mathbf{x})
\end{align*}
$$
- <u>Posterior</u>:
$$
\begin{align*}
&p(\mathbf{w}|\mathbf{y})
=\frac{p(\mathbf{w})p(\mathbf{y}|\mathbf{w})}{p(\mathbf{y})}\\
&\mathrm{Numerator}=\mathcal{N}(\mathbf{w};\mathbf{\mu},\Sigma)\mathcal{N}(\mathbf{y};\phi_\mathbf{x}^T\mathbf{w},\Lambda)\\
&=\mathcal{N}(\mathbf{w};\mathbf{\mu},\Sigma)\mathcal{N}(\phi_\mathbf{x}^T\mathbf{w};\mathbf{y},\Lambda)\\
&=\mathcal{N}(\mathbf{w};\mathbf{\mu},\Sigma)\mathcal{N}(\mathbf{w};\phi_\mathbf{x}^{T,-1}\mathbf{y},(\phi_\mathbf{x}^T\Lambda^{-1}\phi_\mathbf{x})^{-1})\\
&=Z\cdot\mathcal{N}(\mathbf{w};(\Sigma^{-1} +\phi_\mathbf{x}\Lambda^{-1}\phi_\mathbf{x}^T)^{-1}(\phi_\mathbf{x}^T\Lambda^{-1}\mathbf{y}+\Sigma^{-1}\mathbf{\mu}),
(\Sigma^{-1} +\phi_\mathbf{x}\Lambda^{-1}\phi_\mathbf{x}^T)^{-1})\\
&Z=\mathcal{N}(\phi_\mathbf{x}^{T,-1}\mathbf{y};\mathbf{\mu},\Sigma+(\phi_\mathbf{x}^T\Lambda\phi_\mathbf{x})^{-1})\\
&\mathrm{Denominator}=\mathcal{N}(\mathbf{y};\phi_\mathbf{x}^T{\mu},\Lambda+\phi_\mathbf{x}^T\Sigma \phi_\mathbf{x})=Z
\end{align*}
$$
By applying <u>Woodbury Matrix Identity</u>, we can acquire the second form of Gaussian posterior:
$$
p(\mathbf{w}|\mathbf{y})=\mathcal{N}(\mathbf{w};\mathbf{\mu}+\Sigma \phi_\mathbf{x}(\phi_\mathbf{x}^T\Sigma \phi_\mathbf{x}+\Lambda)^{-1}(\mathbf{y}-\phi_\mathbf{x}^T\mathbf{\mu}),\Sigma-\Sigma \phi_\mathbf{x}(\phi_\mathbf{x}^T\Sigma \phi_\mathbf{x}+\Lambda)^{-1}\phi_\mathbf{x}^T\Sigma)
$$
