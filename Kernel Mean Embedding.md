+++
title = "Kernel Mean Embedding"
hascode = true
date = Date(2022, 04, 21)
rss = "In this post, we show how Kernel Mean Embedding works through some toy examples."
+++
@def tags = ["syntax", "code"]

# KERNEL MEAN EMBEDDING 

*This post about Kernel Mean Embedding is an excerpt from my thesis on using Kernel Mean Embedding to predict Pharmaceutical granules' size, changed slightly to fit a more broader audience. This post should be accessible to as many people from different sorts of backgrounds as possible. Don't let the theory scare you as this will become much clearer with the toy examples. These examples are written in Julia, which many of you won't be familiar with. However, Julia's code is very easy to interpret! The outline of the theory is based on the review article of Kernel Mean Embedding by Muandet et al. [^Review]. Finally, the Pluto notebooks can be found on my Github and all of my sources are referenced at the bottom of this page. I also hope to upload my thesis' version of this part somewhere in July. Enjoy!*

\toc

## Kernel methods

### Introduction

Inner products serve as a powerful tool in many established machine learning algorithms, such as principal component analysis (PCA)[^Pearson] [^Hotelling], perceptron [^MinskyPapert] [^Rosenblatt], and support vector machine (SVM) [^Cortes]. These algorithms consider the data, e.g., $\mathbf{x},\mathbf{x}' \in \mathcal X$, with $\mathcal X$ a non empty set, through their inner product $\langle\mathbf{x}, \mathbf{x}'\rangle$, which can be interpreted as a similarity measure between $\mathbf{x}$ and $\mathbf{x}'$. But real-life data is often complex and the class of linear functions induced by the inner products might prove to be insufficient. The aim of kernel methods is to handle complex data which can't be linearly evaluated, by replacing $\langle \mathbf{x}, \mathbf{x}'\rangle$ with some other (non-linear) similarity measure. 

Naturally, an extension of $\langle \mathbf{x}, \mathbf{x}'\rangle$ can be made by explicitly applying a non-linear transformation:
\begin{equation} \label{eq1}
    \begin{split}
    \phi:\mathcal{X} &\to \mathcal{F}, \\
     x &\mapsto \phi(x), 
    \end{split}
\end{equation}
from $\mathcal X$ to the high-dimensional feature space $\mathcal F$. In this new feature space the inner product can be evaluated:
\begin{equation} \label{eqInner}
k(\mathbf{x},\mathbf{x}') := \langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle_{\mathcal{F}},
\end{equation}
where $\langle\cdot, \cdot \rangle_{\mathcal{F}}$ is the inner product of $\mathcal F$ and $\phi$ is called the feature map. $k$ is the kernel function which defines a non-linear similarity measure between $\mathbf{x}$ and $\mathbf{x}'$. By substituting $\langle\mathbf{x}, \mathbf{x}'\rangle$ with $\langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle_{\mathcal{F}}$ a non-linear extension of the algorithms that consider data through $\langle\mathbf{x}, \mathbf{x}'\rangle$, can be made. So, a linear algorithm in $\mathcal F$ corresponds to a non-linear counterpart in the original input space. It is important to note that the algorithm stays the same, only the space in which the algorithm operates, changes. 

As an example, consider the arbitrary classification problem in the figure below, in which one wishes to find a decision function that separates the blue points from the green ones. 

![Data in Input Space](/assets/Data_in_Input_Space.png) 

Consider a polynomial feature mapping on the data points $\phi:(x_{1}, x_{2})\mapsto ({x_{1}}^2,\sqrt{2}x_{1}x_{2}, {x_{2}}^2)$. The inner product of $\mathcal{F}$ can be calculated:
\begin{equation}
\begin{split}
\langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle_{\mathcal{F}} &= {x_{1}}^2{x_{1}'}^2 + 2x_{1}x_{2}x_{1}'x_{2}' + {x_{2}}^2{x_{2}'}^2 \\
&= {(x_{1}x_{1}' + x_{2}x_{2}')}^2 \\
&= {\langle \mathbf{x}, \mathbf{x}'\rangle}^2.
\end{split}
\end{equation}
So, the new similarity measure is the square of the inner product in $\mathcal X$. When applied on the data points, one can obtain the following result illustrated in the figure below.

![Data in Feature Space](/assets/Data%20in%20Feature%20Space.png)

In this new higher dimensional feature space, $\mathbb{R}^3$, one can find an appropriate learning algorithm that defines a decision function which can separate the two circles easily. So, in the original input space the underlying function defining the interactions between all the features of data might not be clear. That's why the data is transformed to higher dimensional feature space where the underlying function might become more clear. In this example, by mapping the data points to a higher dimensional feature space, one can distinguish a cone shape in the transformed data. So, in this new space it is easier to define a function describing the underlying relations between the data points than in the original input space.

The core idea of kernel methods is taking data which lives in an input space where it's not easy to perform machine learning and transform this data to a higher dimensional space where effective use of learning algorithms can be made.

### The kernel trick

As seen in the above example, to evaluate equation 2 one needs to work in two steps: i) explicitly construct the feature maps $\phi(\mathbf{x})$, and ii) subsequently evaluate the inner product $\langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle_{\mathcal{F}}$. This can become a problem when $\phi(\mathbf{x})$ defines a computationally expensive transformation to a high-dimensional feature space.
Fortunately, there exists a solution to this by evaluating $\langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle_{\mathcal{F}}$ without explicitly constructing the feature maps. This is a core idea of kernel methods and is often called "the kernel trick" in the machine learning community[^Review]. 

A visual representation of the kernel trick is illustrated in the figure below. For the above mentioned example one can just consider $k(\mathbf{x},\mathbf{x}') = {\langle\mathbf{x}, \mathbf{x}'\rangle}^2$, rather than calculating the feature maps explicitly. In other words, to avoid using the coordinates of the vectors in the new feature space, a similarity measure is used in that space, which can be put in an algorithm that only needs the value of this measure.

![Visual representation of the kernel trick](/assets/Kernel%20Trick%20Visualisation.PNG) 

What are the requirements for the kernel trick to do its magic? If $k$ is positive definite there always exists a feature map $\phi : \mathcal X \to \mathcal F$ such that $k(\mathbf{x},\mathbf{x}') = \langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle_{\mathcal{F}}$ [^Aronszajn] without the explicit construction of $\phi$ [^Cortes] [^Schol]. So, the kernel function is used as a way to calculate an inner product $\langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle$ in a high-dimensional feature space $\mathcal H$ for some data points $\mathbf{x},\mathbf{x}' \in \mathcal X$. The collection of all these pairwise inner products within the set of data vectors $\mathbf{x}$ is defined as the $N \times N$-Gram or kernel matrix $K_{ij} := k(\mathbf{x}_{i},\mathbf{x}_{j})$. The comparison function $k$ is a positive definite kernel on $\mathcal X$ if it is symmetric, i.e., $k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}',\mathbf{x})$, and the Gram matrix is positive definite:
\begin{equation}
\displaystyle\sum_{i=1}^{N}\displaystyle\sum_{j=1}^{N} c_{i}c_{j}k(\mathbf{x}_{i},\mathbf{x}_{j}) \geq 0,  
\end{equation}
for any $N \in \mathbb N$, all finite sequences of points $(x_{1},...,x_{N}) \in \mathcal X^{N}$ and any $N$ real-valued coefficients $(c_{1},...,c_{N}) \in \mathbb R^{N}$ [^Mercer].

### Recurrent kernel Hilbert space

Another important property of a positive definite kernel is that it induces a space of functions from $\mathcal X$ to $\mathbb R$ called a reproducing kernel Hilbert space (RKHS) $\mathcal H$, which is why the kernel is also called a reproducing kernel [^Aronszajn]. It's important to note that the RKHS is a space of functions from $\mathcal X$ to $\mathbb R$. In other words, each data point $\mathbf{x}$ in $\mathcal X$ will be represented by a function $\phi(\mathbf{x})$ in $\mathcal H$.

An RKHS has two properties important to kernel mean embedding: (i) for any $\mathbf{x} \in \mathcal X$, the function $k(\mathbf{x},\cdot):\mathbf{y} \mapsto k(\mathbf{x},\mathbf{y})$ is an element of $\mathcal H$. So, whenever the kernel $k$ is used, the feature space $\mathcal F$ is the RKHS $\mathcal H$ associated with this kernel. This can be considered as the canonical feature map:
\begin{equation}
\begin{split}
    k:\mathcal X &\to \mathcal H \subset {\mathbb R}^{\mathcal X}, \\
    x &\mapsto k(\mathbf{x},\cdot),
\end{split}
\end{equation}
where $\mathbb R^{\mathcal X}$ denotes the vector space of functions from $\mathcal X$ to $\mathbb R$; (ii) a function $k : \mathcal X \times \mathcal X \to \mathbb R$ is called a reproducing kernel of $\mathcal H$ if $k(\cdot,\mathbf{x}) \in \mathcal H$ for all $\mathbf{x} \in \mathcal X$ and the reproducing property
\begin{equation}
    f(\mathbf{x}) = \langle f,k(\cdot,\mathbf{x})\rangle_{\mathcal H}
\end{equation}
holds for all $f \in \mathcal H$ and all $\mathbf{x} \in \mathcal X$. In particular: if $f(\mathbf{x'} = k(\mathbf{x}',\cdot))$ for some $\mathbf{x}' \in \mathcal X$, $k(\mathbf{x},\mathbf{x}')=\langle k(\mathbf{x},\cdot),k(\mathbf{x}',\cdot)\rangle_{\mathcal H}$.

Aronszajn (1950): *“There is a one-to-one correspondence between the reproducing kernel $k$ and the RKHS $\mathcal H$”.*

### Kernel functions

The kernel trick not only delivers powerful (non-linear) learning algorithms, but also paves the path for domain experts to invent certain kernels which are suitable for specific applications. The kernel trick does not only apply to Euclidean data, but also to non-Euclidean structured data, functional data, and other domains on which a positive definite kernel may be applied [^Schol] [^Gartner]. Various kernels have been proposed in various application domains [^Genton] and for different types of data, such as strings, graphs and trees [^Schol] [^Gartner] [^Hofmann].

The kernel used in this thesis is called the Gaussian kernel, which is a member of the class of kernels called radial basis functions (RBFs):
\begin{equation} \label{gauss}
    k^{RBF}(\mathbf{x},\mathbf{x}') = exp(-\frac{\|\mathbf{x}-\mathbf{x}'\|^{2}}{2{\sigma}^{2}}),
\end{equation}
with $\sigma > 0$ the bandwidth parameter. The Gram matrix of the Gaussian kernel becomes a matrix of ones for $\sigma \to \infty$ and an identity matrix for $\sigma \to 0$. Which means for $\sigma \to \infty$ that all instances are the same, and for the $\sigma \to 0$ that all instances are completely unique, making it a relevant interpretation as a similarity measure [^Vert]. RBF kernels are stationary kernels, which means that they can be described as functions of the differences of their input. RBF kernels are also universal kernels, which means they can represent any smooth function with a high degree of accuracy, assuming being able to find the right bandwidth parameter [^Genton] [^Steinwart2002].

For additional information on the properties of (reproducing kernel) Hilbert spaces and the important theorems of Mercer and Bochner, the reader is advised to read Muandet et al. [^Review], Mercer [^Mercer], and Bochner [^Bochner], respectively. For examples of learning algorithms that use the implicit representation of data points in kernel methods, such as support vector machine (SVM), gaussian process (GP), and neural tangent kernel (NTK), the reader is referred to read Steinwart & Christmann [^Steinwart], Rasmussen [^Rasmussen], and Jacot et al. [^Jacot], respectively.

## Kernel mean embedding of marginal distributions

### From data points to probability measures

Having reviewed the above section on kernel methods one could wonder the practicality of extending kernel methods from individual data points to probability distributions. In many real-life learning problems, however, it could be argued that it is more appropriate to represent the training data as probability distributions rather than individual data points. 

For example, in many situations data is missing or uncertain. As a specific example, gene expression data originating from microarray experiments are known to be very noisy, due to different sources of variabilities [^Yang]. To battle this, each array can be represented as a probability distribution. Another reason for the preference of probability distributions can be computational challenges when dealing with large amounts of training data [^Muandet].

Let $k : \mathcal X \times \mathcal X \to \mathbb R$ be a real-valued positive definite kernel associated with the Hilbert space $\mathcal H$, with $\mathcal X$ a non-empty set. The reproducing property lets the kernel evaluation be interpreted as an inner product in $\mathcal H$ induced by a map from $\mathcal X$ into $\mathcal H$
\begin{equation}
    \mathbf{x} \mapsto k(\mathbf{x,\cdot}).
\end{equation}
Basically, $k(\mathbf{x,\cdot})$ is a high-dimensional representer of $\mathbf{x}$ and because of the reproducing property $k(\mathbf{x,\cdot})$ it is also a representer of evaluation of any function in $\mathcal H$ on the data point $\mathbf{x}$. This lets feature map $\phi$ to be extended to the space of probability distributions through the mapping of $\mu$ which defines the representer in $\mathcal H$ of any distribution $\mathbb P$ [^Muandet] :
\begin{equation}
\begin{split}
    \mu: M_+^{1}(\mathcal X) &\to \mathcal H\\
    \mathbb P &\mapsto \int_{\mathcal X} k(\mathbf{x},\cdot) \mathrm{d\mathbb P}(\mathbf{x}),
\end{split}
\end{equation}
with $M_X^{1}(\mathcal X)$ the space of probability measures over a measurable space $\mathcal X$. The distribution $\mathbb P$ is transformed into an element, the mean embedding ${\mu}_{\mathbb P}$, in an RKHS corresponding to the positive definite kernel $k$, hence the name kernel mean embedding:
\begin{equation} \label{eqP}
    \phi(\mathbb P) = \mu_{\mathbb P} := \mathbb E_{X\sim \mathbb P}[k(X,\cdot)] = \int_{\mathcal X} k(\mathbf{x},\cdot) \mathrm{d\mathbb P}(\mathbf{x}).
\end{equation}
The element is the expected value in the RKHS and since $\mathbb P$ is a probability density distribution, it can be interpreted as an integral [^Smola] [^Berlinet]. In other words, in equation 10 the distribution $\mathbb{P}$ is transformed into an element in the RKHS $\mathcal{H}$ corresponding to the kernel $k$, therefore RKHS methods can be extended to probability measures. A visual representation of this mean embedding is illustrated in the figure below.

![Embedding of marginal distributions](/assets/Embedding%20of%20Marginal%20Distributions.PNG) 

### Kernel mean representation

One can wonder how much information about the distribution $\mathbb P$ this kernel mean embedding can capture. This depends on the used kernel and can range from only the first moment of $\mathbb P$ to all information of $\mathbb P$. Consider following examples with their corresponding captured information: 

* $k(\mathbf{x},\mathbf{x}')=\langle\mathbf{x},\mathbf{x}'\rangle$ : the first moment of $\mathbb P$
* $k(\mathbf{x},\mathbf{x}')={(\langle\mathbf{x},\mathbf{x}'\rangle+1)}^{p}$: moments of $\mathbb P$ up to order $p \in \mathbb N$
* $k(\mathbf{x},\mathbf{x}')$ is universal/characteristic: all information of $\mathbb P$

The first and second example are called the linear and polynomial kernel, respectively. The linear kernel, which equals the computation of the inner product, lets $\mu_{\mathbb P}$ retain the first moment of $\mathbb P$. For the polynomial kernel of order $p$, the mean map equals up to the $p$-th moment of $\mathbb P$. For some other explicit examples the reader is referred to Smola et al. [^Smola], Fukumizu et al. [^Fuku], Sriperumbudur et al. [^Sriper], Gretton et al. [^Gretton], and Schölkopf et al. [^Scholkopf]. 

In the third example, characteristic and universal kernels are mentioned. Characteristic kernels are a class of kernel functions for which the kernel mean embedding captures all information about the distribution $\mathbb P$. A kernel $k$ is characteristic if the map $\mathbb P \mapsto \mu_{\mathbb P}$ is injective, i.e., ${\|\mu_{\mathbb P} - \mu_{\mathbb Q} \|}_{\mathcal H} = 0$ if and only if $\mathbb P = \mathbb Q$ [^Fuku2004]. The Gaussian kernel used in this thesis is a characteristic kernel [^Fuku]. It must be noted that all universal kernels are characteristic, but characteristic kernels may not be universal [^Gretton] [^Steinwart2001].

This injectivity of the map $\mathbb P \mapsto \mu_{\mathbb P}$ ensures that the RKHS embedding is suitable for regression problems, since each element in the feature space corresponds to one unique distribution in the original space. In other words, no information is lost when mapping the distribution into the RKHS.

### Empirical estimate of mean embeddings

It is important to note that in practice, the true distribution of $\mathbb P$ is often unknown, and only an i.i.d. sample $\{x_{1},...,x_{n}\}$ from $\mathbb P$ is available. $\mu_{\mathbb P}$ can be estimated by taking an empirical average:
\begin{equation}
    \hat{\mu}_{\mathbb P} := \frac{1}{n}\displaystyle\sum_{i=1}^{n}k(\mathbf{x}_{i}, \cdot),
\end{equation}
with $\hat{\mu}_{\mathbb P}$ an unbiased estimate of ${\mu}_{\mathbb P}$. The weak law of large numbers indicates that $\hat{\mu}_{\mathbb P}$ converges to the true mean embedding ${\mu}_{\mathbb P}$ [^Sriper2012]. In this thesis, the data is interpreted as a probability mass distribution of $\mathbf X$. For example,  $\hat{\mathbb P} := \frac{1}{n}\sum_{i=1}^{n}{\delta}_{\mathbf{x}_i}$, with ${\delta}_{\mathbf{x}}$ the Dirac measure defined for $\mathbf x \in \mathcal X$, such that:
\begin{equation} \label{weights}
    \hat{\mu}_{\mathbb P} := \displaystyle\sum_{i=1}^{n}w_{i}k(\mathbf{x}_{i}, \cdot),
\end{equation}
with $\mathbf w = [w_i] \in \Delta^{n-1}$, i.e., a histogram with weights subject to the constraint $\sum_{i}^{n}w_i = 1$ and $w_i > 0$ [^Song]. So, the mean embedding becomes a weighted sum of feature vectors.

### Toy problem 1: kernel PCA

Principal component analysis (PCA) is a classical linear technique for dimensionality reduction, data exploration, feature extraction and data visualization. The general outline of PCA is performing a linear projection of the data onto a lower dimensional subspace such that the variance of the projected data is maximized. Kernel PCA is simply a non-linear extension of PCA by executing these principles in the Hilbert space and making use of the kernel trick.

A brief derivation of kernel PCA is given below, based on the works of [^schol], which the reader is referred to for more detailed information.

Consider the data set $\mathcal{S} = (\mathbf{x}_1 ,\dots, \mathbf{x}_N)$, a covariance matrix $\mathbf{C}$ in the feature space of $\mathcal{S}$ can be constructed:
\begin{equation} \label{covC}
    \mathbf{C} = \frac{1}{N}\displaystyle\sum_{i=1}^{N}\phi(\mathbf{x}_n)\phi(\mathbf{x}_n)^{T}.
\end{equation}
For simplicity, assume that these feature representations have a zero mean so that $\sum_{n}\phi(\mathbf{x}_n) = 0$, this will be touched upon later in this section. The eigenvector expansion is defined by
\begin{equation} \label{expansion}
    \mathbf{C}\mathbf{v}_{i} = \lambda_{i}\mathbf{v}_{i}.
\end{equation}
The goal is to perform this expansion without explicitly working in the feature space. Using equation 13, equation 14 can be rewritten as:
\begin{equation}
    \frac{1}{N}\displaystyle\sum_{i=1}^{N}\phi(\mathbf{x}_n)\{\phi(\mathbf{x}_n)^{T}\mathbf{v}_{i}\}=\lambda_{i}\mathbf{v}_{i}.
\end{equation}
This is an interesting outcome, since it allows for the vector $\mathbf{v}_{i}$ to be written as a linear combination of $\phi(\mathbf{x}_n)$:
\begin{equation}
    \mathbf{v}_{i} = \displaystyle\sum_{i=1}^{N}a_{in}\phi(\mathbf{x}_n).
\end{equation}
Using this, the eigenvector expansion in equation 14 becomes:
\begin{equation}
    \frac{1}{N}\displaystyle\sum_{i=1}^{N}\phi(\mathbf{x}_n)\phi(\mathbf{x}_n)^{T}\displaystyle\sum_{m=1}^{N}a_{im}\phi(\mathbf{x}_n)=\lambda_{i}\displaystyle\sum_{i=1}^{N}a_{in}\phi(\mathbf{x}_n).
\end{equation}
This can be entirely expressed in the form of the kernel function $k(\mathbf{x}_i,\mathbf{x}_j) = \phi(\mathbf{x}_i)^{T}\phi(\mathbf{x}_j)$ by multiplying with $\phi(\mathbf{x}_l)^T$, which, in matrix notation, gives:
\begin{equation}
    \mathbf{K}^{2}\mathbf{a}_{i}=\lambda_{i}N\mathbf{K}\mathbf{a}_{i},
\end{equation}
with $\mathbf{a}_{i}$ an $N$-dimensional column vector with elements $a_{in}$ for $n = 1,\dots,N$. Because of the positive definite property of the kernel matrix, it is invertible:
\begin{equation}
    \mathbf{K}\mathbf{a}_{i}=\lambda_{i}N\mathbf{a}_{i}.
\end{equation}
After solving the eigenvector problem, the obtained principal component projections can be expressed in terms of the kernel function. The projection of a point $\mathbf{x}$ onto the eigenvector $i$ is given by:
\begin{equation}
   y_{i}(\mathbf{x})=\phi(\mathbf{x})^{T}\mathbf{v}_{i} = \displaystyle\sum_{n=1}^{N}a_{in}\phi(\mathbf{x})^{T}\phi(\mathbf{x}_n) = \displaystyle\sum_{n=1}^{N}a_{in}k(\mathbf{x}, \mathbf{x}_{n}).
\end{equation}
To get back at the problem of centralized feature vectors, this is can be solved in terms of the kernel function. The centered kernel matrix is defined as:
\begin{equation} \label{centeredK}
    \tilde{\mathbf{K}} = \mathbf{K} - 1_N\mathbf{K} -\mathbf{K}1_N + 1_N\mathbf{K}1_N.
\end{equation}
With $1_N$ an $N \times N$ matrix of which every element has the value $\frac{1}{N}$. Important to note is that a kernel PCA with a linear kernel is simply a normal PCA.

To illustrate kernel methods over marginal distributions and kernel PCA, consider the following input data in the figure below. The data consists of 12 data point groups, which are positioned in an expanding manner towards the right. In this arbitrary toy problem, the goal is to gather information about the data by using kernel PCA for exploratory data analysis. 

![Toy Data KPCA](/assets/Toy%20Data%20KPCA.png) 

Consider the function kernelPCA, which was implemented from scratch for this toy problem, in the following Julia code. X and Y are the respective matrices of the x and y coordinates of the data points. In each matrix, the rows correspond to one data group, e.g., row one of the X matrix corresponds to the x coordinates of one data group. 

To obtain the centered kernel matrix from equation 21, the kernel used over matrices is defined. A Gaussian kernel is used with a  given scale factor, which is the inverse of the length scale and is linked with the inverse bandwidth parameter of the Gaussian kernel. Then the kernel matrix is generated over the x and y matrices, each row is taken as one element, hence the use of RowVecs(). Now everything is obtained to solve equation 21. Finally, the eigenvector problem is solved. Kernel PCA was used on the input data, using four different values for the scale factor. The results were plotted in the figure below, only the first two principal components are shown.
```julia
function kernelPCA(X, Y, scale_factor)
	# scale_factor = inverse of lengthscale, X = matrix x1 points, Y = matrix x2 points, each row of the points matrices is an input element
	# Defining the Gaussian kernel
	k = SqExponentialKernel() ∘ ScaleTransform(scale_factor)
	# Generating the kernel over the x1 and x2 points
	K = kernelmatrix(k, RowVecs(X), RowVecs(Y))
	# Calculating the centralized kernel matrix
	K_centered = K .- mean(K, dims =1) .- mean(K, dims =2) .+ mean(K)
	# Solving the eigenvector problem
	eigenval, eigenvect = eigen(K_centered, sortby=$\lambda$ -> -real($\lambda$))
	eigenvect = real(eigenvect[:, 1:2])
	eigenval = real(eigenval[1:2])
	return eigenvect,eigenval,K
end
```

![Toy Data results](/assets/Toy%20KPCA%20Resultss.png) 

Each dot represents a data group from the input data and has a matching color. The results of the first three plots (from the upper left to the bottom right) give a clue about some occurring clustering. The data groups seem to cluster in groups up to 3, with each cluster exponentially being further away from the previous one. This could indicate that data groups which are spread out alike are clustered together, and the kernel PCA also gives an idea of the location of the data groups. The kernel PCA with a used scale factor of 0.01 does not give meaningful results.

In the figure below the heatmaps of the kernel matrices over the input data are shown for each scale factor. Note that the two upper plots use a different color scaling. These heatmaps illustrate the use of the scale factor, which corresponds with the inverse the bandwidth parameter of the Gaussian kernel. For example, in the lower right plot, a relatively high scale factor was used, which corresponds to a relatively low bandwidth parameter. Remember from equation 7, that for a low bandwidth parameter, the kernel matrix becomes an identity matrix, essentially meaning every input element in the matrix is unique. Hence, the bad results for this plot. For the upper left plot the reverse is happening, almost all elements are similar. However, there is still some differences in the values of the matrix, which cannot be noticed due to the color scaling, hence the results.

![Toy Data heatmaps](/assets/Toy%20KPCA%20Heatmapss.png) 

In the next toy problem kernel mean embedding is used to define a metric for probability functions, called the maximum mean discrepancy (MMD). This metric is very important for solving problems in statistics and machine learning when handling distributions.

### Toy problem 2: maximum mean discrepancy

Consider following arbitrary toy problem, illustrated in the figure below. Given noisy data points which lay in a circular shape, is it possible to find a model fit of 100 equally spaced points which lay on a circle with a certain radius that represents the input data?

![Input Data](/assets/Toy%20Problem%20Inference%20Input%20Data.png) 

This problem can be solved using kernel mean embedding, particularly using the maximum mean discrepancy (MMD). The maximum mean discrepancy corresponds to the RKHS distance between mean embeddings [^Gretton] [^Borgwardt] :
\begin{equation}
\begin{split}
  MMD^{2}(\mathbb P, \mathbb Q, \mathcal H) &= {\|\mu_{\mathbb P} - \mu_{\mathbb Q}\|}_{\mathcal H}^{2} \\
  &= {\|\mu_{\mathbb P}\|}_{\mathcal H} -2\langle{\mu_{\mathbb P}, \mu_{\mathbb Q}}\rangle_{\mathcal H} + {\|\mu_{\mathbb Q}\|}_{\mathcal H}.
\end{split}
\end{equation}
Given $\{\mathbf{x}_i\}_{i=1}^n \sim \mathbb P$ and $\{\mathbf{y}_j\}_{j=1}^n \sim \mathbb Q$, the empirical MMD is \citep[^Borgwardt] :
\begin{equation}
\begin{split}
    \widehat{MMD}_u^{2}(\mathbb P, \mathbb Q, \mathcal H) &= \frac{1}{n(n-1)}\displaystyle\sum_{i=1}^{n}\displaystyle\sum_{j\neq i}^{n}k(\mathbf{x}_{i},\mathbf{x}_{j}) - \frac{2}{nm}\displaystyle\sum_{i=1}^{n}\displaystyle\sum_{j= 1}^{m}k(\mathbf{x}_{i},\mathbf{y}_{j})\\ 
    &+ \frac{1}{m(m-1)}\displaystyle\sum_{i=1}^{m}\displaystyle\sum_{j\neq i}^{m}k(\mathbf{y}_{i},\mathbf{y}_{j}).
\end{split}
\end{equation}
This distance between mean embeddings of features represents the distance between distributions in the input space. In other words, the smaller the MMD, the smaller the distance is between distributions in the input space. This can be used to find a model to fit the training data of the toy problem.

First, a Gaussian kernel is defined. A scale factor of 0.01 is chosen, as this value gave the best results later on. 

```julia
k = SqExponentialKernel() ∘ ScaleTransform(0.01)
```

Second, 100 circles are generated with 100 equally spaced points. Each circle has a different radius ranging from 15 to 25. To compute the MMD between these circles and the input data, several Gram matrices need to be computed. In the code below, K1 and K2 are the Gram matrices generated by the function "kernelmatrix" over the input data and a generated model, respectively. The coordinates of each data point need to be considered as one input value $x$ in the Gaussian kernel, hence the use of "RowVecs". In K3 each element is a Gaussian kernel of two inputs, one being the coordinates of the input data and the other one being the coordinates of the generated model given a radius. With the computed Gram matrices, for each generated circle, an MMD can be calculated.

```julia
# Generate an array of 100 Radii ranging from 15 to 25
R2 = LinRange(15, 25, n)
# Assign an array for the MMDs
mmds = Array{Float64}(undef, 0, 1)
# Generate the kernelmatrix over the coordinates of the input data
K1 = kernelmatrix(k, RowVecs(X))
for i in 1:n
	# Generate a model circle consisting of 100 points with Radius i
	A = circle(100, R2[i])
	# Generate the kernelmatrix over the coordinates of the model circle
	K2 = kernelmatrix(k, RowVecs(A))
	# Generate the kernelmatrix over the coordinates of the input data 
	# and the model circle
	K3 = kernelmatrix(k, RowVecs(X), RowVecs(A))
	# Calculate the MMD
	mmd = mean(K1) - 2 * mean(K3) + mean(K2)
	# Add the MMD to the mmds array
	mmds = [mmds; mmd]
end
```
The obtained MMDs can be plotted versus the radius of each corresponding generated circle, illustrated in the figure below.

![MMD Versus Radius](/assets/MMD%20versus%20Radius.png) 

A value for the radius of the best model fitted on the input data can be obtained. In the figure below, the obtained model is plotted. The distance between the embedding of this model and the embedding of the input data is the smallest out of all generated circles, meaning the model fits the input data the best in the input space.

![Fitted Model on Input Data](/assets/Radius%20and%20Fit.png)

## Kernel mean embedding of conditional cistributions

In this section, the kernel mean embedding of marginal distributions is extended to conditional distributions $\mathbb P(Y|X)$ and $\mathbb P(Y|X=\mathbf{x})$ for some $\mathbf x \in \mathcal X$, capturing even more complex data [^Song] [^Song2009]. The figure below illustrates conditional mean embedding [^Review]. 

![Schematic Illustration of Conditional Mean Embedding](/assets/conditional.png) 

Consider the two positive definite kernels, $k : \mathcal X \times \mathcal X \to \mathbb R$ and $l : \mathcal Y \times \mathcal Y \to \mathbb R$ for the respective domains of $X$ and $Y$, and the respective RKHSs $\mathcal H$ and $\mathcal G$. To fully represent $\mathbb P(Y|X)$, conditioning and conditional expectation need to be performed. Then, the conditional mean embeddings of the conditional distributions $\mathbb P(Y|X)$ and $\mathbb P(Y|X=\mathbf{x})$ can be defined as $\mathcal U_{Y|X}: \mathcal H \to \mathcal G$ and $\mathcal U_{Y|X} \in \mathcal G$, such that:

$$
    \mathcal U_{Y|\mathbf{x}} = \mathbb E_{Y|X}[\varphi(Y)|X=\mathbf{x}] = \mathcal U_{Y|X}k(\mathbf{x},\cdot),\\
     \mathbb E_{Y|X}[g(Y)|X=\mathbf{x}] = \scal{g,\mathcal U_{Y|\mathbf{x}}}_{\mathcal G}, \forall g \in \mathcal G.
$$

$\mathcal U_{Y|X}$ is an operator from RKHS $\mathcal H$ to $\mathcal G$, and $\mathcal U_{Y|\mathbf{x}}$ is an element in RKHS $\mathcal G$. In the above equation it is stated that the conditional mean embedding of $\mathbb P(Y|X=\mathbf{x})$ is the conditional expectation of the feature map of $Y$ given that $X = \mathbf{x}$. The operator $\mathcal U_{Y|X}$ is the conditioning operation that when applied to $\phi(x) \in \mathcal H$ delivers the conditional mean embedding $\mathcal U_{Y|x}$. Equation 15 shows the reproducing property of $\mathcal U_{Y|x}$: it should be a representer of conditional expectation in $\mathcal G$ with regards to $\mathbb P(Y|X=x)$. 

Song et al. [^Song] [^Song2009] provide following definition: let $\mathcal C_{XX}: \mathcal H \to \mathcal H$ and $\mathcal C_{XY}: \mathcal H \to \mathcal G$ be the covariance operator on $X$ and cross-covariance operator from $X$ to $Y$, respectively. Then, the conditional mean embedding $\mathcal U_{Y|X}$ and $\mathcal U_{Y|x}$ are defined as:

$$
    \mathcal U_{Y|X} := \mathcal C_{XY}\mathcal C_{XX}^{-1}\\
    \mathcal U_{Y|\mathbf{x}} := \mathcal C_{XY}\mathcal C_{XX}^{-1}k(\mathbf{x},\cdot).
$$

Fukumizu et al. [^Fuku2004] [^Fuku3] state that if $\mathbb E_{Y|X}[g(Y)|X=\cdot] \in \mathcal H$ for any $g \in \mathcal G$, then

$$
    \mathcal C_{XX} \mathbb E_{Y|X}[g(Y)|X=\cdot] = \mathcal C_{XY}g.
$$

Further, for some $\mathbf{x} \in \mathcal X$, the reproducing property states that

$$
    \mathbb E_{Y|\mathbf{x}}[g(Y)|X=\mathbf{x}] = \scal{\mathbb E_{Y|X}[g(Y)|X], k(\mathbf{x}, \cdot)}_{\mathcal H}.
$$

Combining the above the equations, and taking the conjugate transpose of $\mathcal C_{XX}^{-1}\mathcal C_{XY}$ gives

$$
    \mathbb E_{Y|\mathbf{x}}[g(Y)|X=\mathbf{x}] = \scal{g, \mathcal C_{XY}\mathcal C_{XX}^{-1}k(\mathbf{x},\cdot)}_{\mathcal{G}} = \scal{g, \mathcal U_{Y|\mathbf{x}}}_{\mathcal{G}}.
$$

In an infinite RKHS, $\mathcal C_{XX}^{-1}$ does not exist, so a regularised version is often used, that is [^Song2009] [^Fuku2004] :

$$
    \mathcal U_{Y|X} := \mathcal C_{XY}(\mathcal C_{XX} + \lambda \mathcal I)^{-1},
$$

where $\lambda > 0$ is the regularization parameter and $\mathcal I$ is the identity operator in $\mathcal H$. Fukumizu et al.  [^Fuku3] stated that, under some mild conditions, the empirical estimator is a consistent estimator of $\mathbb E_{Y|\mathbf{x}}[g(Y)|X=\mathbf{x}]$.

In practice, the joint distribution $\mathbb P(X,Y)$ is unknown, and $\mathcal C_{XY}$ and $\mathcal C_{XX}$ cannot be calculated directly. To overcome this, consider the i.i.d. sample $(\mathbf{x}_{1},\mathbf{y}_{1}),...,(\mathbf{x}_{n},\mathbf{y}_{n})$ from $\mathbb P(X,Y)$. Let $Y := [\phi(\mathbf{x}_{1}),...,\phi(\mathbf{x}_{n})]^{T}$ and $\Phi := [\varphi(\mathbf{y}_{1}),...,\varphi(\mathbf{y}_{n})]^{T}$ where $\phi : \mathcal X \to \mathcal H$ and $\varphi : \mathcal Y \to \mathcal G$ are the feature maps corresponding to the respective kernels $k$ and $l$. The associated Gram matrices are  $K = Y^{T}Y$ and $L = \Phi^{T}\Phi$. With this information, the empirical estimator of the conditional mean embedding becomes [^Song2009]

$$
    \hat{\mathcal C}_{XY}(\hat{\mathcal C}_{XX} + \lambda \mathcal I)^{-1}k(\mathbf{x},\cdot) = \frac{1}{n}\Phi Y^{T}(\frac{1}{n}YY^{T} + \lambda \mathcal I)^{-1}k(\mathbf{x},\cdot)\\
    = \Phi Y^{T}(YY^{T} + n\lambda \mathcal I)^{-1}k(\mathbf{x},\cdot)\\
    = \Phi (Y^{T}Y + n\lambda \mathbf{I}_n)^{-1}Y^{T}k(\mathbf{x},\cdot)\\
    = \Phi(\mathbf{K} + n\lambda \mathbf{I}_{n})^{-1}\mathbf{k}_{\mathbf{x}}
$$

So, the conditional mean embedding of $µ_{Y|\mathbf{x}}$ is estimated by [^Song2009]

$$
    \hat{µ}_{Y|\mathbf{x}} = \Phi(\mathbf{K} + n\lambda \mathbf{I}_{n})^{-1}\mathbf{k}_{\mathbf{x}}. 
$$

Similar as the embedding of marginal distributions in equation 12, the embedding of conditional distributions can be written in terms of weights. Consider $\hat{\mathbf{ \beta}_{\lambda}} := (\mathbf{K} + n\lambda \mathbf{I}_{n})^{-1}\mathbf{k}_{\mathbf{x}} \in \mathbb R^{n}$. Subsequently, equation 22 can be written as $\hat{µ}_{Y|\mathbf{x}} = \Phi \hat{\mathbf{ \beta}_{\lambda}} = \sum_{i=1}^{n}(\hat{\mathbf{\beta}_{\lambda}})_i\varphi(\mathbf{y}_i)$. It is important to note that in this case the weights $\mathbf{\beta}$ depend on the value of the conditioning variable $X$ instead of being uniform [^Song2009]. 

## Learning on distributional data

The conditional mean embedding has a natural interpretation as a solution to a vector-valued regression problem, observed first by Zhang et al. [^Zhang] and later by Grünewälder et al. [^Grune]. As discussed earlier, the conditional mean embedding is defined via $\mathbb E_{Y|\mathbf{x}}[g(Y)|X=\mathbf{x}] = \scal{g, \hat{µ}_{Y|\mathbf{x}}}_{\mathcal{G}}$, i.e., for every $\mathbf x \in \mathcal X$, $\hat{µ}_{Y|\mathbf{x}}$ is a function on $\mathcal Y$ and by that defines a mapping from $\mathcal X$ to $\mathcal G$. Moreover, the empirical estimator $\hat{µ}_{Y|\mathbf{x}} = \Phi(\mathbf{K} + n\lambda \mathbf{I}_{n})^{-1}\mathbf{k}_{\mathbf{x}}$, suggests that the conditional mean embedding is the solution to an underlying regression problem.

Consider the i.i.d. sample $(\mathbf{x}_{1},\mathbf{z}_{1}),...,(\mathbf{x}_{n},\mathbf{z}_{n}) \in \mathcal X \times \mathcal G$, a vector-valued regression problem can be formulated as:

$$
    \hat{\varepsilon}_{\lambda}(f) = \displaystyle\sum_{i=1}^{n} {\|\mathbf{z}_{i} - f(\mathbf{x}_{i})\|}_{\mathcal G}^{2} + \lambda {\|f\|}_{\mathcal {H}_{\Gamma}}^{2},  
$$

where $\mathcal G$ is a Hilbert space, $\mathcal {H}_{\Gamma}$ an RKHS of vector-valued functions from $\mathcal X$ to $\mathcal G$, and $\hat{\varepsilon}_{\lambda}$ is error of the associated regression problem [^Micch].  Grünewälder et al. [^Grune] states that by  minimizing the optimization in equation 23, $\hat{µ}_{Y|\mathbf{x}}$ can be obtained. So, the natural optimization problem for the conditional mean embedding is to find a function $µ : \mathcal X \to \mathcal G$ that minimizes an objective. This objective can be bounded from above by a surrogate loss function, which can be described by its empirical counterpart [^Grune] :

$$
    \hat{\varepsilon}_{\mathcal S}[µ] = \displaystyle\sum_{i=1}^{n} {\|l(\mathbf{y}_i,\cdot) - µ(\mathbf{x}_i)\|}_{\mathcal G}^{2} + \lambda {\|µ\|}_{\mathcal {H}_{\Gamma}}^{2}.  
$$

The added regularization term provides a well-posed problem and prevents overfitting. Interpreting the conditional mean embedding as a solution to a vector-valued regression problem gives the advantage of being able to use cross-validation or model selection, due to the well-defined loss function. Since $\mathcal G$ is assumed to be finite-dimensional, the conditional mean embedding is the ridge regression of the feature vectors. Consider $\hat{\mathbf{ \beta}_{\lambda}} := (\mathbf{K} + n\lambda \mathbf{I}_{n})^{-1}\mathbf{k}_{\mathbf{x}}$, in a ridge regression context the hat matrix $\mathbf{H}_{\lambda}$ is:

$$
    \mathbf{H}_{\lambda}\mathbf{k}_{\mathbf{x}} = \hat{\mathbf{k}}_{\mathbf{x}} = \Phi \hat{\mathbf{ \beta}_{\lambda}}\\
    \mathbf{H}_{\lambda} = \mathbf{K}(\mathbf{K} + \lambda \mathbf{I})^{-1}.
$$

Using leave-one-out cross-validation (LOOCV), the estimated conditional embedding is defined as [^Spline] :

$$
    \hat{µ}_{Y|\mathbf{x}}^{LOOCV} = (\mathbf{I} - diag(\mathbf{H}_{\lambda}))^{-1}(\mathbf{H}_{\lambda} - diag(\mathbf{H}_{\lambda}))\Phi,
$$

with $diag(\cdot)$ the diagonal matrix. It's important to note that using the above equation, all LOOCV conditional embeddings can be calculated at once using matrix multiplications. 

To interpret the obtained results, the underlying distributions needs to be recovered from the embeddings, which is the topic of the next section.

## Recovering distributions from RKHS embeddings

Recovering information of $\mathbb P$ from the kernel mean embedding $µ_{\mathbb P}$ is known as the distributional pre-image problem [^Kwok] [^Kana]. In this context, objects in the input space which correspond with a specific kernel mean embedding in a feature space, are looked for. Consider $\mathbb{P}_{\theta}$ an arbitrary distribution parameterized by $\mathbf{\theta}$ and its mean embedding in $\mathcal H$, $µ_{\mathbb{P}_{\theta}}$. By solving following minimization problem $\mathbb{P}_{\theta}$ can be found:

$$
    \hat{\mathbf{\theta}} = \argmin_{\theta \in \Theta}{\|\hat{µ}_{Y} - µ_{\mathbb{P}_{\theta}}\|}_{\mathcal H}^{2}\\
    = \argmin_{\theta \in \Theta} \scal{\hat{µ}_{Y},\hat{µ}_{Y}} -2\scal{\hat{µ}_{Y}, µ_{\mathbb{P}_{\theta}}} + \scal{µ_{\mathbb{P}_{\theta}}, µ_{\mathbb{P}_{\theta}}},
$$

where $\theta$ is subject to appropriate constraints. As seen earlier, equation 27 describes the MMD. $\scal{\hat{µ}_{Y},\hat{µ}_{Y}}$ is constant and is left out of the minimization. Assume $µ_{\mathbb{P}_{\theta}} = \sum_{i=1}^{n}{\alpha}_i\varphi(\mathbf{y}_i)$ for some $\alpha \in \Delta^{n-1}$, i.e., $\mathbb{P}_{\theta}$ is a histogram.

$\mathbf w = [w_i] \in \Delta^{n-1}$, i.e., a histogram with weights subject to the constraint $\sum_{i}^{n}w_i = 1$ and $w_i > 0$ [^Song].

### Toy problem 3: regression


## References
[^Review]: Muandet, K., Fukumizu, K., Sriperumbudur, B., & Schölkopf, B. (2017). Kernel mean embedding of distributions: A review and beyond. Foundations and Trends® in Machine Learning, 10(1-2), 1-141.
[^Pearson]: Pearson, K. (1901). LIII. On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin philosophical magazine and journal of science*, 2(11), 559-572.
[^Hotelling]: Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of educational psychology*, 24(6), 417.
[^MinskyPapert]: Minsky, M., & Papert, S. (1969). Perceptrons.
[^Rosenblatt]: Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological review*, 65(6), 386.
[^Cortes]: Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine learning*, 20(3), 273-297.
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
[^Fuku]: Fukumizu, K., Gretton, A., Sun, X., & Schölkopf, B. (2007). Kernel measures of conditional dependence. *Advances in neural information processing systems*, 20.
[^Sriper]: Sriperumbudur, B. K., Gretton, A., Fukumizu, K., Schölkopf, B., & Lanckriet, G. R. (2010). Hilbert space embeddings and metrics on probability measures. *The Journal of Machine Learning Research*, 11, 1517-1561.
[^Gretton]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). *A kernel two-sample test*. The Journal of Machine Learning Research, 13(1), 723-773.
[^Scholkopf]: Schölkopf, B., Muandet, K., Fukumizu, K., Harmeling, S., & Peters, J. (2015). Computing functions of random variables via reproducing kernel Hilbert space representations. *Statistics and Computing*, 25(4), 755-766.
[^Fuku2004]: Fukumizu, K., Bach, F. R., & Jordan, M. I. (2004). Dimensionality reduction for supervised learning with reproducing kernel Hilbert spaces. *Journal of Machine Learning Research*, 5(Jan), 73-99.
[^Steinwart2001]: Steinwart, I. (2001). On the influence of the kernel on the consistency of support vector machines. *Journal of machine learning research*, 2(Nov), 67-93.
[^Sriper2012]: Sriperumbudur, B. K., Fukumizu, K., Gretton, A., Schölkopf, B., & Lanckriet, G. R. (2012). On the empirical estimation of integral probability metrics. *Electronic Journal of Statistics*, 6, 1550-1599.
[^Song]: Song, L., Fukumizu, K., & Gretton, A. (2013). Kernel embeddings of conditional distributions: A unified kernel framework for nonparametric inference in graphical models. *IEEE Signal Processing Magazine*, 30(4), 98-111.
[^Borgwardt]: Borgwardt, K. M., Gretton, A., Rasch, M. J., Kriegel, H. P., Schölkopf, B., & Smola, A. J. (2006). Integrating structured biological data by kernel maximum mean discrepancy. *Bioinformatics*, 22(14), e49-e57.
[^Song2009]: Song, L., Huang, J., Smola, A., & Fukumizu, K. (2009, June). Hilbert space embeddings of conditional distributions with applications to dynamical systems. In *Proceedings of the 26th Annual International Conference on Machine Learning* (pp. 961-968).
[^Fuku3]: Fukumizu, K., Song, L., & Gretton, A. (2013). Kernel Bayes' rule: Bayesian inference with positive definite kernels. *The Journal of Machine Learning Research*, 14(1), 3753-3783.
[^Zhang]: Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2012). Kernel-based conditional independence test and application in causal discovery. *arXiv preprint arXiv:1202.3775*.
[^Grune]: Grünewälder, S., Lever, G., Baldassarre, L., Patterson, S., Gretton, A., & Pontil, M. (2012). Conditional mean embeddings as regressors-supplementary. *arXiv preprint arXiv:1205.4656*.
[^Micch]: Micchelli, C. A., & Pontil, M. (2005). On learning vector-valued functions. *Neural computation*, 17(1), 177-204.
[^Spline]: Wahba, G. (1990). *Spline models for observational data*. Society for industrial and applied mathematics.
[^Kwok]: Kwok, J. Y., & Tsang, I. H. (2004). The pre-image problem in kernel methods. *IEEE transactions on neural networks*, 15(6), 1517-1525.
[^Kana]: Kanagawa, M., & Fukumizu, K. (2014, April). Recovering distributions from Gaussian RKHS embeddings. In *Artificial Intelligence and Statistics* (pp. 457-465). PMLR.