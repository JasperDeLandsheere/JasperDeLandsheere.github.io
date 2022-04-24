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

Consider some linearly separatable data which can be evaluated by the classical machine learning algorithms perceptron, support vector machine and principal component analysis. These algorithms consider the data, $x,x' \in \mathcal X$, with $\mathcal X$ a non empty set, through their inner product $\scal{x, x'}$. Another way to view this inner product is as a linear similarity measure between the elements of $\mathcal X$. But what if the data is more complex and cannot be linearly evaluated?

Let's consider this arbitrary classification problem, in which two data circles cannot be linearly separated. 

![Data in Input Space](/assets/Data%20in%20Input%20Space.png) 

Here the kernel methods come into play! Kernel methods replace the inner product $\scal{x, x'}$ as a similarity measure with a non-linear one. Consider the non-linear transformation:

$$
\phi:\mathcal X\to \mathcal F
$$*
$$
x \mapsto \phi (x)
$$

![Data in Feature Space](/assets/Data%20in%20Feature%20Space.png)


kernel trick, RKHS, ...


## Kernel Mean Embedding of Marginal Distributions

MMD, universal/characteristic kernel, recovering information

### Toy example 1: Inference


## Kernel Mean Embedding of Conditional Distributions


### Toy example 1: Regression


### Toy example 2: Kernel PCA


## References

