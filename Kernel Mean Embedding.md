+++
title = "Kernel Mean Embedding"
hascode = true
date = Date(2022, 04, 21)
rss = "In this post, we show how Kernel Mean Embedding works through some toy examples."
+++
@def tags = ["syntax", "code"]

# KERNEL MEAN EMBEDDING 

*This post about Kernel Mean Embedding is an excerpt from my thesis on using Kernel Mean Embedding to predict Pharmaceutical granules' size, changed slightly to fit a more broader audience. This post should be accessible to as many people from different sorts of backgrounds as possible. Don't let the theory scare you as this will become much clearer with the toy examples. These examples are written in Julia, which many of you won't be familiar with. However, Julia's code is very easy to interpret! The outline of the theory is based on the review article of Kernel Mean Embedding by Muandet et al. Finally, the Pluto notebooks can be found on my Github and all of my sources are referenced at the bottom of this page. I also hope to upload my thesis' version of this part somewhere in July. Enjoy!*

\toc

## Kernel Methods
Inner products serve as a powerful tool in many established machine learning algorithms, such as principal component analysis (PCA)[^Pearson] [^Hotelling], perceptron [^MinskyPapert] [^Rosenblatt], and support vector machine (SVM) [^Cortes]. These algorithms consider the data, e.g., $\mathbf{x},\mathbf{x}' \in \mathcal X$, with $\mathcal X$ a non empty set, through their inner product $\scal{\mathbf{x}, \mathbf{x}'}$, which can be interpreted as a similarity measure between $\mathbf{x}$ and $\mathbf{x}'$. But real-life data is often complex and the class of linear functions induced by the inner products might prove to be insufficient. The aim of kernel methods is to handle complex data which can't be linearly evaluated, by replacing $\scal{\mathbf{x}, \mathbf{x}'}$ with some other (non-linear) similarity measure. 

Naturally, an extension of $\scal{\mathbf{x}, \mathbf{x}'}$ can be made by explicitly applying a non-linear transformation:

$$
\phi:\mathcal X\to \mathcal F,\\
x \mapsto \phi (x),
$$

from $\mathcal X$ to the high-dimensional feature space $\mathcal F$. In this new feature space the inner product can be evaluated:

$$
k(\mathbf{x},\mathbf{x}') := \scal{\phi(\mathbf{x}), \phi(\mathbf{x}')}_{\mathcal{F}},
$$

where $\scal{\cdot, \cdot}_{\mathcal{F}}$ is the inner product of $\mathcal F$ and $\phi$ is called the feature map. $k$ is the kernel function which defines a non-linear similarity measure between $\mathbf{x}$ and $\mathbf{x}'$. By substituting $\scal{\mathbf{x}, \mathbf{x}'}$ with $\scal{\phi(\mathbf{x}), \phi(\mathbf{x}')}_{\mathcal{F}}$ a non-linear extension of the algorithms that consider data through $\scal{\mathbf{x}, \mathbf{x}'}$, can be made. So, a linear algorithm in $\mathcal F$ corresponds to a non-linear counterpart in the original input space. It is important to note that the algorithm stays the same, only the space in which the algorithm operates, changes. 

As an example, consider the arbitrary classification problem below, in which one wishes to find a decision function that separates the blue points from the green ones. 

![Data in Input Space](/assets/Data%20in%20Input%20Space.png) 

Consider a polynomial feature mapping on the datapoints $\phi:(x_{1}, x_{2})\mapsto ({x_{1}}^2,\sqrt{2}x_{1}x_{2}, {x_{2}}^2)$. The inner product in $\mathcal{F}$ can be calculated

$$
\scal{\phi(\mathbf{x}), \phi(\mathbf{x}')}_{\mathbb{R_{3}}} = {x_{1}}^2{x_{1}'}^2 + 2x_{1}x_{2}x_{1}'x_{2}' + {x_{2}}^2{x_{2}'}^2\\
= {(x_{1}x_{1}' + x_{2}x_{2}')}^2\\
= {\scal{x, x'}_{\mathbb{R_{2}}}}^2.
$$

So, the new similarity measure is the square of the inner product in $\mathcal X$. When applied on the datapoints, one can obtain the following result.

![Data in Feature Space](/assets/Data%20in%20Feature%20Space.png)

In this new higher dimensional feature space one can find an appropiate learning algorithm that defines a decision function which can separate the two circles easily.

The core idea of kernel methods is taking data which lives in an input space where it's not easy to perform machine learning and transform this data to a higher dimensional space where effective use of learning algorithms can be made.

As seen in the above example, to evaluate equation 2 one needs to work in two steps: i) explicitly constructing the feature maps $\phi(\mathbf{x})$, and ii) evaluating the inner product $\scal{\phi(\mathbf{x}), \phi(\mathbf{x}')}_{\mathcal{F}}$. This can become a problem when $\phi(\mathbf{x})$ defines a computationally expensive transformation to a high-dimensional featue space. Fortunately, there exists a solution to this by evaluating $\scal{\phi(\mathbf{x}), \phi(\mathbf{x}')}_{\mathcal{F}}$ without explicitly constructing the feature maps. This is a core idea of kernel methods and is often called "the kernel trick" in the machine learning community [^Review]. A visual representation of the kernel trick is illustrated below. For the above mentioned example one can just consider $k(\mathbf{x},\mathbf{x}') = {\scal{\mathbf{x}, \mathbf{x}'}}^2$ rather than calculating the feature maps explicitly.

![Visual representation of the kernel trick](/assets/Kernel%20Trick%20Visualisation.PNG) 

What are the requirements for the kernel trick to do its magic? If $k$ is positive definite there always exists a feature map $\phi : \mathcal X \to \mathcal F$ such that $k(\mathbf{x},\mathbf{x}') = \scal{\phi(\mathbf{x}), \phi(\mathbf{x}')}_{\mathcal{F}}$ [^Aronszajn] without the explicit construction of $\phi$ [^Cortes] [^Schol]. So, the kernel function is used as a way to calculate an inner product $\scal{\phi(\mathbf{x}), \phi(\mathbf{x}')}$ in a high-dimensional feature space $\mathcal H$ for some data points $\mathbf{x},\mathbf{x}' \in \mathcal X$. The collection of all these pairwise inner products within the set of data vectors $\mathbf{x}$ is defined as the $n \times n$-Gram or kernel matrix $K_{ij} := k(\mathbf{x}_{i},\mathbf{x}_{j})$. The comparison function $k$ is a positive definite kernel on $\mathcal X$ if it is symmetric, i.e., $k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}',\mathbf{x})$, and the Gram matrix is positive definite:

$$
\displaystyle\sum_{i=1}^{n}\displaystyle\sum_{n=1}^{j} c_{i}c_{j}k(\mathbf{x}_{i},\mathbf{x}_{j}) \geq 0,  
$$

for any $n \in \mathbb N$, all finite sequences of points $(x_{1},...,x_{n}) \in \mathcal X^{n}$ and any $n$ real-valued coefficients $(c_{1},...,c_{n}) \in \mathbb R^{n}$ [^Jaya] [^Mercer].

Another important property of a positive definite kernel is that it induces a space of functions from $\mathcal X$ to $\mathbb R$ called a reproducing kernel Hilbert space (RKHS) $\mathcal H$, which is why the kernel is also called a reproducing kernel [^Aronszajn]. It's important to note that the RKHS is a space of functions from $\mathcal X$ to $\mathbb R$. In other words, each data point $\mathbf{x}$ in $\mathcal X$ will be represented by a function $\phi(\mathbf{x})$ in $\mathcal H$.

An RKHS has two properties important to kernel mean embedding: (i) for any $\mathbf{x} \in \mathcal X$, the function $k(\mathbf{x},\cdot):\mathbf{y} \mapsto k(\mathbf{x},\mathbf{y})$ is an element of $\mathcal H$. So, whenever the kernel $k$ is used, the feature space $\mathcal F$ is the RKHS $\mathcal H$ associated with this kernel. This can be considered as the canonical feature map:

$$
    k:\mathcal X \to \mathcal H \subset {\mathbb R}^{\mathcal X},\\
    x \mapsto k(\mathbf{x},\cdot),
$$

where $\mathbb R^{\mathcal X}$ denotes the vector space of functions from $\mathcal X$ to $\mathbb R$; (ii) a function $k : \mathcal X \times \mathcal X \to \mathbb R$ is called a reproducing kernel of $\mathcal H$ if $k(\cdot,\mathbf{x}) \in \mathcal H$ for all $\mathbf{x} \in \mathcal X$ and the reproducing property

$$
    f(\mathbf{x}) = \scal{f,k(\cdot,\mathbf{x})}_{\mathcal H}
$$

holds for all $f \in \mathcal H$ and all $x \in \mathcal X$. In particular: if $f(\mathbf{x'} = k(\mathbf{x}',\cdot))$ for some $\mathbf{x}' \in \mathcal X$, $k(\mathbf{x},\mathbf{x}')=\scal{k(\mathbf{x},\cdot),k(\mathbf{x}',\cdot)}_{\mathcal H}$.

Aronszajn (1950): *“There is a one-to-one correspondance between the reproducing kernel $k$ and the RKHS $\mathcal H$”.*

The kernel trick not only delivers powerful (non-linear) learning algorithms, but also paves the path for domain experts to invent certain kernels which are suitable for specific applications. The kernel trick does not only apply to Euclidian data, but also to non-Euclidian structured data, functional data, and other domains on which a positive definite kernel may be applied [^Schol] [^Gartner]. Various kernels have been proposed in various application domains [^Genton] and for different types of data, such as strings, graphs and trees [^Schol] [^Gartner] [^Hofmann].

The kernel used in this thesis is called the Gaussian kernel, which is a member of the class of kernels called radial basis functions (RBFs):

$$
    k^{RBF}(\mathbf{x},\mathbf{x}') = exp(-\frac{\|\mathbf{x}-\mathbf{x}'\|^{2}}{2{\sigma}^{2}}),
$$

with $\sigma > 0$ the bandwith parameter. The Gram matrix of the Gaussian kernel becomes a matrix of ones for $\sigma \to \infty$ and an indentity matrix for $\sigma \to 0$. Which means for the former that all instances are the same, and for the latter that all instances are completely unique, making it a relevant interpretation as a similarity measure [^Vert]. RBF kernels are stationary kernels, which means that they can be described as functions of the differences of their input. RBF kernels are also universal kernels, which means they can represent any smooth function with a high degree of accuracy, assuming being able to find the right bandwith parameter [^Genton] [^Steinwart2002].

For additional information on the properties of (reproducing kernel) Hilbert spaces and the important theorems of Mercer and Bochner, the reader is advised to read Muandet et al. [^Review], Mercer [^Mercer], and Bochner [^Bochner], respectively. For examples of learning algorithms that use the implicit representation of data points in kernel methods, such as support vector machine (SVM), gaussian process (GP), and neural tangent kernel (NTK), the reader is referred to read Steinwart & Christmann [^Steinwart], Rasmussen [^Rasmussen], and Jacot et al. [^Jacot], respectively.

## Kernel Mean Embedding of Marginal Distributions

Having reviewed the above section on kernel methods one could wonder the practicality of extending kernel methods from individual data points to probability distributions. In many real-life learning problems, however, it could be argued that it is more appropriate to represent the training data as probability distributions rather than individual data points. For example, in many situations data is missing or uncertain. As a specific example, gene expression data originating from microarray experiments are known to be very noisy, due to different sources of variabilities [^Yang]. To battle this, each array can be represented as a probability distribution. Another reason for the preference of probability distributions can be computational challenges when dealing with large amounts of training data [^Muandet].

Let $k : \mathcal X \times \mathcal X \to \mathbb R$ be a real-valued positive definite kernel associated with the Hilbert space $\mathcal H$, with $\mathcal X$ a non-empty set. The reproducing property lets the kernel evaluation be interpreted as an inner product in $\mathcal H$ induced by a map from $\mathcal X$ into $\mathcal H$

$$
    \mathbf{x} \mapsto k(\mathbf{x,\cdot}).
$$

Basically, $k(\mathbf{x,\cdot})$ is a high-dimensional representer of $\mathbf{x}$ and because of the reproducing property $k(\mathbf{x,\cdot})$ is also a representer of evaluation of any function in $\mathcal H$ on the data point $\mathbf{x}$. This lets feature map $\phi$ to be extended to the space of probability distributions through the mapping of $µ$ which defines the representer in $\mathcal H$ of any distribution $\mathbb P$ [^Muandet] :

$$
    µ: M_X^{1}(\mathcal X) \to \mathcal H\\
    \mathbb P \mapsto \int_{\mathcal X} k(\mathbf{x},\cdot) \mathrm{d\mathbb P}(\mathbf{x}),
$$

with $M_X^{1}(\mathcal X)$ the space of probability measures over a measurable space $\mathcal X$. The distribution $\mathbb P$ is transformed into an element, the mean embedding ${µ}_{\mathbb P}$, in an RKHS corresponding to the positive definite kernel $k$, hence the name kernel mean embedding. The element is the expected value in the RKHS and since $\mathbb P$ is a probability density distribution, it can be interpreted as an integral [^Smola] [^Berlinet].

$$
    \phi(\mathbb P) = µ_{\mathbb P} := \mathbb E_{X\sim \mathbb P}[k(X,\cdot)] = \int_{\mathcal X} k(\mathbf{x},\cdot) \mathrm{d\mathbb P}(\mathbf{x}).
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
    \hat{µ}_{\mathbb P} := \frac{1}{n}\displaystyle\sum_{i=1}^{n}k(x_{i}, \cdot)
$$

with $\hat{µ}_{\mathbb P}$ an unbiased estimate of ${µ}_{\mathbb P}$. **add probability mass distribution thingy? It can be estimated consistently**

### Toy example 1: Inference (using Maximum mean discrepancy)
Now, let's discuss the maximum mean discrepancy with the help of a toy example. First, what is the maximum mean discrepancy or MMD? The MMD corresponds to the RKHS distance between mean embeddings:

$$
  MMD^{2}(\mathbb P, \mathbb Q, \mathcal H) = {\|µ_{\mathbb P} - µ_{\mathbb Q}\|}_{\mathcal H}^{2} = {\|µ_{\mathbb P}\|}_{\mathcal H} -2{\scal{µ_{\mathbb P}, µ_{\mathbb Q}}}_{\mathcal H} + {\|µ_{\mathbb Q}\|}_{\mathcal H}.
$$

This distance represents the distance between distributions in the input space! So we can use this to find objects in the inpust space which correspond with a specific KME in the feature space. For the empirical MMD, given $\{x_i\}_{i=1}^n \sim \mathbb P$ and $\{y_j\}_{j=1}^n \sim \mathbb Q$:

$$
    MMD_u^{2}(\mathbb P, \mathbb Q, \mathcal H) = \frac{1}{n(n-1)}\displaystyle\sum_{i=1}^{n}\displaystyle\sum_{j\neq i}^{n}k(x_{i},x_{j}) - \frac{2}{nm}\displaystyle\sum_{i=1}^{n}\displaystyle\sum_{j = 1}^{m}k(x_{i},y_{j})+ \frac{1}{m(m-1)}\displaystyle\sum_{i=1}^{m}\displaystyle\sum_{j\neq i}^{m}k(y_{i},y_{j})
$$

Let's generate some data points which lay in a circle with an unknown radius and add some noise:

![Input Data](/assets/Noisy%20Circle.png) 

Let's consider an arbitrary problem: we want to fit an ideal model to this data, 100 equally spaced points which lay on a certain radius which represents the input data, how could we do this? We can use the MMD!

First, we define our kernel, the Gaussian kernel (an RBF kernel). In this example we are using the KernelFunctions.lj package, which uses ScaleTransfrom, which is the inverse of the lengthscale.

```julia
k = SqExponentialKernel() ∘ ScaleTransform(0.1)
```

Then we generate 100 model circles with different radii ranging from 15 to 25. We compute the Gram matrix over the input data, notice the "RowVecs(X)", this means we take both the x and y coordinate as input, this becomes a value which is then compared with the other input values. For each model circle, we generate the Gram matrix over the model data and the Gram matrix over the model data and input data together. After this we have everything we need to compute the MMD such as in equation 12.

```julia
	R2 = LinRange(15, 25, n)
	mmds = Array{Float64}(undef, 0, 1)
	K1 = kernelmatrix(k, RowVecs(X))
	for i in 1:n
		A = circle(100, R2[i])
		K2 = kernelmatrix(k, RowVecs(A))
		K3 = kernelmatrix(k, RowVecs(X), RowVecs(A))
		mmd = mean(K1) - 2 * mean(K3) + mean(K2)
		mmds = [mmds; mmd]
```
Plotting out our results for the MMD versus the radius of each model circle.

![MMD Versus Radius](/assets/MMD%20versus%20Radius.png) 

Taking the minimum of the MMD and the corresponding Radius.

```julia
    index = argmin(mmds)
	minimum(mmds)
    R2[index]
```
Generating our model fit with the found radius which has the matching smallest MMD!

![Fitted Model on Input Data](/assets/Radius%20and%20Fit.png) 

## Kernel Mean Embedding of Conditional Distributions

Just like we performed KME on marginal distributions, we can perform KME on conditional distributions, capturing even more complex data! 

Let's say we have two positive definite kernels, $k : \mathcal X \times \mathcal X \to \mathbb R$ and $l : \mathcal Y \times \mathcal Y \to \mathbb R$ for the respective domains of $X$ and $Y$, with respective RKHSs $\mathcal H$ and $\mathcal G$. Then, the conditional mean embeddings of the conditional distributions $\mathbb P(Y|X)$ and $\mathbb P(Y|X=x)$ can be defined as $\mathcal U_{Y|X}: \mathcal H \to \mathcal G$ and $\mathcal U_{Y|X} \in \mathcal G$, they satisfy:

$$
    \mathcal U_{Y|x} = \mathbb E_{Y|X}[\phi(Y)|X=x] = \mathcal U_{Y|X}k(x,\cdot)\\
     \mathbb E_{Y|X}[g(Y)|X=x] = \scal{g,\mathcal U_{Y|x}}_{\mathcal G}, \forall g \in \mathcal G.
$$

$\mathcal U_{Y|X}$ is an operator from RKHS $\mathcal H$ to $\mathcal G$ and $\mathcal U_{Y|x}$ is an element in that RKHS $\mathcal G$. In equation 13 it is important to note that the conditional mean embedding of $\mathbb P(Y|X=x)$ is the conditional expectation of the feature map of $Y$ given that $X = x$. The operator $\mathcal U_{Y|X}$ is the conditioning operation that when apploef to $\phi(x) \in \mathcal H$ delivers the conditional mean embedding $\mathcal U_{Y|x}$. Equation 14 shows the reproducing property of $\mathcal U_{Y|x}$: it should be a representer of conditional expectation in $\mathcal G$ with regards to $\mathbb P(Y|X=x)$. 

Now, let $\mathcal C_{XX}: \mathcal H \to \mathcal H$ and $\mathcal C_{XY}: \mathcal H \to \mathcal G$ be the covariance operator on $X$ and corss-covariance operator from $X$ to $Y$, respectively. Then:
$$
    \mathcal U_{Y|X} := \mathcal C_{XY}\mathcal C_{XX}^{-1}\\
    \mathcal U_{Y|x} := \mathcal C_{XY}\mathcal C_{XX}^{-1}k(x,\cdot)
$$
In an infinite RKHS, $\mathcal C_{XX}^{-1}$ does not exist, so we often use a regularised version, that is:
$$
    \mathcal U_{Y|X} := \mathcal C_{XY}(\mathcal C_{XX} + \lambda \mathcal I)^{-1},
$$
Where $\lambda$ is the regularization parameter (> 0) and $\mathcal I$ is the identitiy operator in $\mathcal H$.

In practice: $\mathbb P(X,Y)$ is unknown, and $\mathcal C_{XY}$ and $\mathcal C_{XX}$ cannot be computed directly. To overcome this, consider the i.i.d. sample $(x_{1},y_{1}),...,(x_{n},y_{n})$ from $\mathbb P(X,Y)$. Let $Y := [\phi(x_{1}),...,\phi(x_{n})]^{T}$ and $\Phi := [\varphi(y_{1}),...,\varphi(y_{n})]^{T}$ where $\phi : \mathcal X \to \mathcal H$ and $\varphi : \mathcal Y \to \mathcal G$ are the feature maps associated with the kernels $k$ and $l$, respectively. The corresponding Gram matrices are defined as $K = Y^{T}Y$ and $L = \Phi^{T}\Phi$. With this information, the empirical estimator of the conditional mean embedding is given by

$$
    \hat{\mathcal C}_{XY}(\hat{\mathcal C}_{XX} + \lambda \mathcal I)^{-1}k(x,\cdot) = \frac{1}{n}\Phi Y^{T}(\frac{1}{n}YY^{T} + \lambda \mathcal I)^{-1}k(x,\cdot)\\
    = \Phi(K + n\lambda I_{n})^{-1}k_{x}
$$

### Toy example 2: Regression (Add quick summary)


### Toy example 3: Kernel PCA


## References
[^MinskyPapert]: Minsky, M., & Papert, S. (1969). Perceptrons.
[^Rosenblatt]: Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological review*, 65(6), 386.
[^Cortes]: Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine learning*, 20(3), 273-297.
[^Pearson]: Pearson, K. (1901). LIII. On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin philosophical magazine and journal of science*, 2(11), 559-572.
[^Hotelling]: Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of educational psychology*, 24(6), 417.
[^Review]: Muandet, K., Fukumizu, K., Sriperumbudur, B., & Schölkopf, B. (2017). Kernel mean embedding of distributions: A review and beyond. Foundations and Trends® in Machine Learning, 10(1-2), 1-141.
[^Aronszajn]: Aronszajn, N. (1950). Theory of reproducing kernels. *Transactions of the American mathematical society*, 68(3), 337-404.
[^Schol]: Schölkopf, B., Smola, A. J., & Bach, F. (2002). *Learning with kernels: support vector machines, regularization, optimization, and beyond*. MIT press.
[^Jaya]: Jayasumana, S., Hartley, R., Salzmann, M., Li, H., & Harandi, M. (2013). Kernel methods on the Riemannian manifold of symmetric positive definite matrices. In *proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 73-80).
[^Genton]: Genton, M. G. (2001). Classes of kernels for machine learning: a statistics perspective. *Journal of machine learning research*, 2(Dec), 299-312.
[^Gartner]: Gärtner, T. (2003). A survey of kernels for structured data. *ACM SIGKDD explorations newsletter*, 5(1), 49-58.
[^Hofmann]: Hofmann, T., Schölkopf, B., & Smola, A. J. (2008). Kernel methods in machine learning. *The annals of statistics*, 36(3), 1171-1220.
[^Vert]: Vert, J. P., Tsuda, K., & Schölkopf, B. (2004). A primer on kernel methods. *Kernel methods in computational biology*, 47, 35-70.
[^Steinwart2002]: Steinwart, I. (2002). Support vector machines are universally consistent. *Journal of Complexity*, 18(3), 768-791.
[^Mercer]: Mercer, J. (1909). Xvi. functions of positive and negative type, and their connection the theory of integral equations. *Philosophical transactions of the royal society of London. Series A, containing papers of a mathematical or physical character*, 209(441-458), 415-446.
[^Bochner]: Bochner, S. (1933). Monotone funktionen, stieltjessche integrale und harmonische analyse. *Mathematische Annalen*, 108(1), 378-410.
[^Steinwart]: Steinwart, I., & Christmann, A. (2008). *Support vector machines*. Springer Science & Business Media.
[^Rasmussen]: Rasmussen, C. E. (2003, February). Gaussian processes in machine learning. In *Summer school on machine learning* (pp. 63-71). Springer, Berlin, Heidelberg.
[^Jacot]: Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. *Advances in neural information processing systems*, 31.
[^Yang]: Yang, Y. H., & Speed, T. (2002). Design issues for cDNA microarray experiments. *Nature Reviews Genetics*, 3(8), 579-588.
[^Muandet]: Muandet, K., Fukumizu, K., Dinuzzo, F., & Schölkopf, B. (2012). Learning from distributions via support measure machines. *Advances in neural information processing systems*, 25.
[^Smola]: Smola, A., Gretton, A., Song, L., & Schölkopf, B. (2007, October). A Hilbert space embedding for distributions. In *International Conference on Algorithmic Learning Theory* (pp. 13-31). Springer, Berlin, Heidelberg.
[^Berlinet]: Berlinet, A., & Thomas-Agnan, C. (2011). *Reproducing kernel Hilbert spaces in probability and statistics*. Springer Science & Business Media.