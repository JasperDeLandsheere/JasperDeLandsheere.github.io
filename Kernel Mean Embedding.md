+++
title = "Kernel Mean Embedding"
hascode = true
date = Date(2022, 04, 21)
rss = "In this post, we show how Kernel Mean Embedding works through some toy examples."
+++
@def tags = ["syntax", "code"]

# KERNEL MEAN EMBEDDING

*Welcome to my first-ever blog post! No better way to start off than with an attempt to explain Kernel Mean Embedding. This post should be accessible to as many people from different sorts of backgrounds as possible. Don't let the theory scare you as this will become much clearer with the toy examples. These examples are written in Julia, which many of you won't be familiar with. However, Julia's code is very easy to interpret! The outline of the theory is based on the review article of Kernel Mean Embedding by Muandet et al. Finally, the Pluto notebooks can be found on my Github and all of my sources are referenced at the bottom of this page.*

\toc

## Kernel Methods

Consider some data which can be evaluated by the classical machine learning algorithms perceptron, support vector machine and principal component analysis. These algorithms consider the data, $x,x' \in \mathcal X$, with $\mathcal X$ a non empty set, through their inner product $\scal{x, x'}$. Another way to view this inner product is as a linear similarity measure between the elements of $\mathcal X$. But what if the data is more complex and cannot be linearly evaluated?

Let's consider this arbitrary classification problem, in which two data circles cannot be linearly separated. 

![Data in Input Space](/assets/Data%20in%20Input%20Space.png) 

Here the kernel methods come into play! Kernel methods replace the inner product $\scal{x, x'}$ as a similarity measure with a non-linear one. Consider the non-linear transformation:

$$
\phi:\mathcal X\to \mathcal F \\
x \mapsto \phi (x)
$$

from $\mathcal X$ to the high-dimensional feature space $\mathcal F$. In this new feature space the inner product can be evaluated:
$$
k(x,x') = \scal{\phi(x), \phi(x')}_{\mathcal{F}},
$$

where $\scal{\cdot, \cdot}_{\mathcal{F}}$ is the inner product of $\mathcal F$, $\phi$ is the feature map, and $k$ is the kernel function which defines a non-linear similarity measure between $x$ and $x'$. The above mentioned algorithms can then be used in this new space, by swapping out $\scal{x, x'}$ with $\scal{\phi(x), \phi(x')}_{\mathcal{F}}$. So, the algorithm itself does not change, rather the space in which it operates! 

To illustrate this, let's perform the following feature mapping of the concentric circles: $\phi:(x_{1}, x_{2})\mapsto ({x_{1}}^2,\sqrt{2}x_{1}x_{2}, {x_{2}}^2)$. In this new higher dimensional feature space we can find a linear model that defines a decision function which separates the two circles easily. We can calculate the kernel function, for this example:

$$
\scal{\phi(x), \phi(x')}_{\mathbb{R_{3}}} = {x_{1}}^2{x_{1}'}^2 + 2x_{1}x_{2}x_{1}'x_{2}' + {x_{2}}^2{x_{2}'}^2\\
= {(x_{1}x_{1}' + x_{2}x_{2}')}^2\\
= {\scal{x, x'}_{\mathbb{R_{2}}}}^2.
$$
So our new similarity measure is the square of the inner product in $\mathcal X$.

So the core idea of kernel methods is taking data which lives in an input space where it's not easy to perform machine learning and transform this data to a higher dimensional space where linear models can be used.

![Data in Feature Space](/assets/Data%20in%20Feature%20Space.png)

As seen above in the example, to evaluate equation 2 we need to work in 2 steps: explicitly constructing the feature maps $\phi(x)$ and then evaluating the inner product $\scal{\phi(x), \phi(x')}_{\mathcal{F}}$. This can become a problem when $\phi(x)$ defines a transformation to a high-dimensional featue space. **The kernel trick** offers a solution to this by evaluating $\scal{\phi(x), \phi(x')}_{\mathcal{F}}$ without explicitly constructing the feature maps. For the example: we can just consider $k(x,x') = {\scal{x, x'}}^2$. Illustration of the kernel trick:

![Visual representation of the kernel trick](/assets/Kernel%20Trick%20Visualisation.PNG) 

Now let's discuss an important requirement needed in order for the kernel trick to do its magic: $k$ has to be positive definite. What does this mean? First we have to define the $n \times n$-Gram or kernel matrix $K_{ij} := k(x_{i},x_{j})$, the collection of all pairwise inner products within the set of data vectors $x$. 

A symmetric function $k$ is positive definite on $\mathcal X$ if the Gram matrix is positive definite, that is,
$$
\sum{i=1}^n\sum{j=1}^j c_{i}c_{j}k(x_{i},x_{j}) \geq 0, \forall x_{i} \in \mathcal X 
$$

Equation 4 holds for any $n \in \mathbb N$, all finite sequences of points $x_{1},...,x_{n}$ in $\mathcal X$ and all choices of $n$ real-valued coefficients $c_{1},...,c_{n} \in \mathbb R$. It's the positive definiteness of the kernel function that guarantees the existence of a dot product space $\mathcal F$ and a feature map $\phi:\mathcal X\to \mathcal F$ such that $k(x,x') = \scal{\phi(x), \phi(x')}_{\mathcal{F}}$ without the need for the explicit construction of $\phi$.

Another important property of a positive definite kernel is that it induces a space of functions from $\mathcal X$ to $\mathbb R$ called a Reproducing Kernel Hilbert Space RKHS $\mathcal H$, which is why the p.d. kernel is also called a reproducing kernel. 

Two important properties define an RKHS: 
First, for any $x \in \mathcal X$, the function $k(x,\cdot):y\mapsto k(x,y)$ is an element of $\mathcal H$. So, when $k$ is used, the feature space $\mathcal F$ is the associated RKHS $\mathcal H$;
$$
    k:\mathcal X \to \mathcal H \subset {\mathbb R}^{\mathcal X}\\
    x \mapsto k(x,\cdot)
$$
Where $\mathbb R^{\mathcal X}$ denotes the vector space of functions from $\mathcal X$ to $\mathbb R$, we call $k$ the canonical feature map.

Second, a function $k : \mathcal X \times \mathcal X \to \mathbb R$ is called a reproducing kernel of $\mathcal H$ if $k(\cdot,x) \in \mathcal H$ for all $x \in \mathcal X$ and the reproducing property
$$
    f(x) = \scal{f,k(\cdot,x)}_{\mathcal H}
$$
holds for all $f \in \mathcal H$ and all $x \in \mathcal X$. Important for us: $k(x,x')=\scal{k(x,\cdot),k(x',\cdot)}_{\mathcal H}$.

Aronszajn (1950): *“There is a one-to-one correspondance between the reproducing kernel $k$ and the RKHS $\mathcal H$”.*


## Kernel Mean Embedding of Marginal Distributions

So how can we extend this mapping to marginal distributions? We simply take the mapping $µ$ which defines the representer in $\mathcal H$ of any distribution $\mathbb P$. The distribution $\mathbb P$ is transformed into an element, the mean embedding ${µ}_{\mathbb P}$ in an RKHS matching the positive definite kernel $k$. The element is the expected value in the RKHS and since $\mathbb P$ is a probability density distribution, it can be written as an integral! 

$$
µ: \mathcal P \to \mathcal H\\
\mathbb P \mapsto \int k(\cdot,x)\, \mathrm{d\mathbb P}x
$$

Visual representation of this mean embedding:

![Embedding of marginal distributions](/assets/Embedding%20of%20Marginal%20Distributions.PNG) 

Now how much information can this mean representation capture about the distribution $\mathbb P$? This depends on the used kernel! This can range from only the first moment of $\mathbb P$ to all information of $\mathbb P$. 

$k(x,x')=\scal{x,x'}$ : the first moment of $\mathbb P$
$k(x,x')={(\scal{x,x'}+1)}^{p}$ (polynomial kernels): moments of $\mathbb P$ up to order $p \in \mathbb N$
$k(x,x')$ is universal/characteristic: all information of $\mathbb P$

A kernel $k$ is characteristic if the map

$$
\mathbb P \mapsto µ_{\mathbb P}
$$

is injective, i.e., ${\|µ_{\mathbb P} - µ_{\mathbb Q} \|}_{\mathcal H} = 0$ if and only if $\mathbb P = \mathbb Q$. This injectivity of the map $\mathbb P \mapsto µ_{\mathbb P}$ ensures that the RKHS embedding is suitibale for regression problems (each element in the feature space corresponds to one unique distribution in the original space)!

An example of a characteristic kernel, **also the kernel used in all toy examples**, is the Gaussian kernel, which is part of the class of kernels called radial basis functions (RBFs):

$$
    k^{RBF}(x,x') = exp(-\frac{\|x-x'\|^{2}}{2{\sigma}^{2}})
$$

with $\sigma > 0$ the bandwith parameter. The Gram matrix of the Gaussian kernel becomes a matrix of ones for $\sigma \to \infty$ and an indentity matrix for $\sigma \to 0$. Which means for the former that all input is the same, and for the latter that all input is completely unique. This RBF kernel is a stationary kernel, which means that it can be described as a function of the difference of its inputs. The RBF kernel is also a universal kernel which means it can represent any smooth function with a high degree of accuracy , assuming chosen the right bandwith parameter. It must be noted that all universal kernels are characteristic, but characteristic kernels may not be universal.

In practive however the access to the true distribution of $\mathbb P$ is often lacking, and only an i.i.d. sample $x_{1},...,x_{n}$ from $\mathbb P$ is avaible. We can estimate $µ_{\mathbb P}$ by

$$
    µ_{\mathbb P} := \frac{1}{n}\displaystyle\sum_{i=1}^{n}k(x_{i}, \cdot)
$$

MMD,, recovering information

### Toy example 1: Inference


## Kernel Mean Embedding of Conditional Distributions


### Toy example 1: Regression


### Toy example 2: Kernel PCA


## References

