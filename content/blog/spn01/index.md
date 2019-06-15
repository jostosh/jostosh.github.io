---
title: Sum-Product Networks
date: "2019-06-11T22:12:03.284Z"
description: A basic introduction that aiming to transfer and compare some of the concepts from neural networks to theory of Sum-Product Networks. 
---

Sum-Product Networks (SPNs) are probabilistic graphical models (PGMs) that have been around for several years, with arguably a limited amount of attention from the machine learning community. I believe this is due to several things. First, the daunting success of advanced deep neural networks like convolutional neural networks and recurrent neural networks overruled much of the alternative methods that may seem inferior at first sight. Second, SPNs are nontrivial to implement in a tensorized fashion (especially when compared to CNNs or RNNs). Third, SPNs come with a rich terminology and a strict probabilistic interpretation that might seem less intuitive than 'cells that fire together wire together'. This series of posts aims at addressing these issues to varying extent by going through some basic intuitions behind SPNs, showing how to implement SPNs in a tensorized fashion and finally, looking at a hands-on example of SPNs for generative and discriminative learning on image datasets.

Quite some time ago, I joined a project where we worked on a library with 'tensorized' implementations of SPNs, known as libspn. The library leverages the flexibility and power of TensorFlow to bring SPNs to the broader ML audience. Although by now it is something I need to pursue in my spare time, I like working on it since it sets some unique engineering challenges. I'll explain the complexity of this challenge over the next few posts.

This series of posts assumes basic knowledge of set theory, probability theory, neural networks and 'tensor-based' programming and computing.

## What Can SPNs Be Used For?
SPNs can be used for tasks like image completion, localization for robotics, classification and can be trained to tackle either unsupervised or supervised or semi-supervised problems. Depending on the task and whether we want the SPN to optimize for generative or discriminative abilities, we can train the SPN with expectation maximization (EM), gradient descent (GD), extended Baum-Welch (EBW) and more algorithms. In some cases, SPNs are trained for generative and discriminative abilities simultaneously. Being fully probabilistic models, they naturally deal with incomplete data. SPNs are PGMs designed with efficiency in mind. They can perform exact joint, conditional or marginal inferences very efficiently. 

## Sum-Product Networks: An Intro
Since SPNs are probabilistic graphical models (PGMs) they resemble a probability distribution (either continuous or discrete). Implementation-wise, this means that our computation takes in random variables on one side and spits out probabilities on another side. Note that this is qualitatively different from multi-layer perceptrons (MLPs). For MLPs the output might be framed probabilistically, such as the softmax output for classification:
$$
p(y_i \mid \boldsymbol X) = \frac{\exp(\boldsymbol w_i^T \boldsymbol h)}{\sum_k \exp(\boldsymbol w_k ^T \boldsymbol h)}
$$

Nevertheless, the inputs (often image pixels) are just raw numbers fed to the network, without any explicit probabilistic interpretation. Also, all intermediate layers in an MLP lack a direct probabilistic interpretation.

Since SPNs are recursive tree-like structures, non-root nodes are roots of sub-SPNs with a probabilistic interpretation similar to the full SPN. In fact, often SPNs are thought of as hierarchical probabilistic models. In other words, the interpretation of nodes in SPNs differs significantly from hidden neurons in an MLP!

## A Shift In Terminology

How SPNs differ from MLPs: 

- _Inputs_ are called _leaves_;
- _Neurons_ are called _nodes_;
- Each leaf node belongs to a single random variable. There can be multiple leaf nodes per random variable. For continuous variables, the leaves that correspond to a variable are the variable's _components_. For discrete categorical variables, the leaves that belong to a variable are the variable's _indicators_ (i.e. $[X_1=0, X_1=1, \ldots]$).
- A set of variables is called a _scope_. Leaf nodes have scopes of exactly one variable e.g. $\{X_1\}$ or $\{X_2\}$. For a non-leaf node, the scope of the node corresponds to the union of the scopes of its children. If a parent node has two children with scopes $\{X_i\}$ and $\{X_j\}$, then that parent has the scope $\{X_i, X_j\}$
The node at the top of the tree-structure is called the root.

In other words, SPNs are directed acyclic graphs with a tree-like structure.
The leaves of an SPN are random variables $\boldsymbol X = [X_1, X_2, \ldots, X_N]$. Additionally, each operation in an SPN must preserve the ability to interpret the computed values as probabilities. This means that the value of every node in the forward pass computes a probability. As a result, a high degree of interpretability comes for free with SPNs.

## Guaranteeing a Probabilistic Interpretation
So exactly how do we make sure that any node at any layer in an SPN computes an exact probability? In the original SPN paper ([Poon and Domingo's, 2011](https://arxiv.org/abs/1202.3732)), this is more formally defined as preserving the property of _validity_ throughout the SPN. Valid SPNs efficiently and accurately model a joint probability. For a  thorough treatment of validity, I suggest you read the original paper [1] and have a look at Poupart's excellent lectures ([first](https://www.youtube.com/watch?v=eF0APeEIJNw) and [second](https://www.youtube.com/watch?v=9-1YE_N-lnw)) on SPNs. For the purpose of these blog posts, we mostly care about _how_ we create valid SPNs. Valid SPNs can be created by

1. Restricting ourselves to (i) non-negative weighted sums and (ii) products for propagating probabilities upward;
2. Ensuring that _scopes_ children of a sum node are _identical_;
3. Ensuring that _scopes_ of children of a product node are _pairwise disjoint_;

Let's look at the restriction of non-negative weighted sums first and products first. A non-negative weighted sum of probabilities occurs in many probabilistic (graphical) models. Usually, the non-negative weights correspond to _prior probabilities_ of the children of the sum. Note that to interpret the weights as probabilities we not only require them to be non-negative but we also must also ensure that they sum up to exactly one, i.e. $\sum_i w_{ij} = 1$.

## A Mixture of Gaussians as an SPN
A common scenario is to have a mixture of Gaussians to model a single random variable:
$$
p(x) = \sum_k \underbrace{p(x \mid z=k)}_{\text{Component: } \mathcal N(\mu, \sigma^2)} \underbrace{p(z=k)}_{\text{Prior (a scalar)}}
$$

Sum nodes in SPNs compute probabilities in the same way, but the 'components' can be either (i) leaf nodes or (ii) sub-SPNs. The equation above is equivalent to the extremely simple SPN below (when replacing $z$ with $X_1$):
![Figure 1: Gaussian Mixture as an SPN. The three leaf nodes correspond to the Gaussian components.](spn_gmixture.png)

Note that in the SPN above, we compute a sum over elements that consider the same singular set of variables, namely $\{\boldsymbol X_1\}$. The leaf nodes correspond to the Gaussian components while the edges correspond to the non-negative weights. The value of the root corresponds to $p(x)$ in Equation 1.

In SPN terminology, a set of variables is called a scope. Leaf nodes have scopes of exactly one variable e.g. $\{\boldsymbol X_1\}$ or $\{\boldsymbol X_2\}$. Note that the SPN in Figure 1 satisfies at least requirements 1 and 2 for a valid SPN, provided we have non-negative weights. The second requirement of having identical scopes for the sum's children is also known as the completeness property. The SPN in Figure 1 does not include products, so we don't really care about the third requirement. Hence, all boxes are ticked and this SPN is valid!

An SPN that violates _completeness_ is the one below:
![Figure 2: Incomplete SPN. Note that the two children have different scopes.](incomplete.png)

## Computing Joint Probabilities
What about the third requirement? Well, if we have an SPN defined over more than one variable, we should _join scopes_ to eventually obtain a root node that computes the _joint probability_ over the variables contained in those scopes. Remember that scopes are joined by a parent node that obtains the _union_ of the scopes of its children. The third requirement describes under which conditions we can join scopes in a valid SPN. It is also known as the _decomposability_ property. It might not be the first time you've heard that when computing the joint probability over two independent variables, you can simply _multiply_ the individual probabilities, e.g. 
$$
p(x,y) = p(x)p(y)
$$
Sum-product networks compute joint probabilities in the same way: they multiply several probabilities to obtain a new joint probability.

Consider the following SPN:
![Figure 3: SPN with two input variables and two indicators per variable.](spn_two_vars.png)

Note the way in which the leaf nodes are connected to the blue product nodes. The connections are such that the scopes of the children of each node are _pairwise disjoint_. For now on, let's assume we are dealing with _binary discrete variables_ (Bernoulli). I will use non-bold font $X_i$ and $\bar{X}_i$ to refer to the indicators of the random variable $\boldsymbol X_i$.

The SPN above computes the following _network polynomial_:
$$
\Phi(X_1, \bar{X_2}, X_1, \bar{X_2}) = w_1 X_1 X_2 + w_2 X_1 \bar{X_2} + w_3 \bar{X_1} X_2 + w_4 \bar{X_1} \bar{X_2} 
$$

What in the world is a network polynomial? This is the definition from the paper by ([Poon and Domingos, 2011](https://arxiv.org/abs/1202.3732)):
>  Let $\Phi(x) \geq 0$ be an unnormalized probability distribution. The network polynomial
> of $\Phi(x)$ is $\sum_x \Phi(x) \Pi(x)$, where $\Pi(x)$ is the product of the indicators that have value 1 in state $x$.

Take a moment to see how the definition fits the network polynomial of the SPN in Figure 3. This polynomial is a so-called _multilinear_ function of the variables $X_1$ and $X_2$. More info on multilinear functions can be found [here](https://en.wikipedia.org/wiki/Multilinear_map).

#### Normalizing the SPN
A network polynomial uses an _unnormalized_ probability distribution, meaning that there is some constant $Z$ such that $\sum_x \Phi(x) / Z = 1$. When an SPN is _normalized_ by normalizing each sum's weight such that $\sum_i w_{ij} = 1$, the normalized SPN's network polynomial _compute probabilities directly_. In
other words:
$$
\sum_x [\Phi](x) = 1 \Rightarrow [\Phi](x) = p(x)
$$

Where $[\Phi]$ is the network polynomial obtained from $\Phi$ after normalizing the weights for each sum.

**From here on, assume that our SPNs are normalized, so that the SPN computes a _'normalized'_ probability distribution.** Otherwise, all of our inferences would require about twice as many computations. If you want to understand why, it's best to consult ([Poon and Domingos, 2011](https://arxiv.org/abs/1202.3732)).

Clearly, the SPN in Figure 2 is a bit more sophisticated than the one in Figure 1. First of all, it is multi-layered. The first layer on top of the leaves consists of products of which the children have pairwise disjoint scopes (requirement 3). The scopes of the 4 products are identical (namely $\{\boldsymbol X_1, \boldsymbol X_2\}$). Consequently, we can combine those probabilities with a sum node at the top of the SPN (the root). To compute the probability of a particular assignment of indicators, we replace the indicators that are included with one and those that are excluded with zero:

$$
p(\{X_1, X_2\}) = w_1 \cdot 1 \cdot 1 + w_2 \cdot 0 \cdot 1 + w_3 \cdot 0 \cdot 1 + w_4 \cdot 0 \cdot 0 = w_1
$$

Or, alternatively:

$$
p(\{\bar{X_1}, X_2\}) = w_1 \cdot 0 \cdot 1 + w_2 \cdot 0 \cdot 0 + w_3 \cdot 1 \cdot 1 + w_4 \cdot 1 \cdot 0 = w_3
$$

If one of the two variables is unknown, we can easily _marginalize out_ this variable by setting both its indicators to 1. If $\boldsymbol X_2$ is unknown, we can
compute the probability of observing $X_1$ at all as follows:

$$
p(\{X_1,X_2,\bar{X_2}\}) = w_1 1\cdot 1 + w_2 \cdot 1 \cdot 1 + w_3 \cdot 0 \cdot 0 + w_4 = w_1 + w_2
$$

#### A Non-Decomposable SPN
Below is an example of a simple SPN that is not decomposable. The root has two children with the same scope.

![Figure 4: An SPN that is non-decomposable. Note that the root has two children with the same scope, while there should be no overlap in their scopes.](not_decomposable.png) 


## Stacking More Layers
The scopes of a product's children can contain more than a single random variable. Take, for instance, the following SPN:
![Figure 5: SPN with four input variables and two indicators per variable.](medium_spn.png) 

Figure 5 displays an SPN with even more layers than the one in Figure 3. The first product layer combines either $\{\boldsymbol X_1\}$ and $\{\boldsymbol X_2\}$ into $\{\boldsymbol X_1, \boldsymbol X_2\}$ (the blue products) or $\{\boldsymbol X_3\}$ and $\{\boldsymbol X_4\}$ into $\{\boldsymbol X_3, \boldsymbol X_4\}$ (the orange products). The blue product nodes are followed up by the green sum nodes and the orange product nodes by the pink sum nodes (they do not alter the scope, but merely 'mix' the probabilities of the underlying scopes). These green and pink nodes have scopes with two variables each. Moreover, if we take one green node and one pink node, then that results in pairwise disjoint sets, so we can safely join those scopes by a product node (the grey nodes at the penultimate layer). Finally, all the grey nodes at the penultimate layer have identical scopes, so we can 'mix' those probabilities in the final root sum.

Check out the overview of the scopes on the right of Figure 3 and verify yourself that the SPN is indeed (i) decomposable by checking the products' children's scopes for having no overlap in their scopes (pairwise disjoint) and (ii) complete by checking the sums' children's scopes for being identical. In case these two properties check out, that means the SPN is valid!

Note that the architecture in the SPNs above is arbitrary (as long as they are decomposable and complete that is), so we can easily increase the number of sums per layer, reduce the number of products per layer etc. However, for the purpose of illustration, I've kept things relatively simple.

#### More Advanced Architectures
Despite the constraints on the exact architecture of these networks, SPNs can also be framed as _convolutional architectures_ or _recurrent architectures_. This is, however, outside the scope of this post. Stay tuned though... 

## Wrapping Up
If you're coming from a background that with the main focus on neural networks or other main-stream machine learning approaches, the amount of theory and terminology coming at you in this post might seem a bit overwhelming at first. I'll try and show in the next two posts that it's nevertheless feasible to create implementations of SPNs that feel similar to the _Keras_ style of stacking layers in an object-oriented fashion, even though we have to take care of constraints like decomposability and completeness. 

At least remember the following things:

1. SPNs are probabilistic models that can be used for doing inferences like 
    * Marginalizing out variables (useful for when part of your input is missing).
    * Conditional inference (useful for classification). We'll get to this in a later blog post.
2. SPNs can be seen as feed-forward architectures with a few constraints:
    * Decomposability: a product should not have children with overlapping scopes (think about the product rule of multiplying independent probabilities to get a joint distribution).
    * Completeness: a sum's children must have identical scopes
3. Normalize your SPNs so that they compute a normalized probability distribution, rather than an unnormalized probability distribution.
4. An SPN's nonlinearities are its products and can be seen as a special type of MLP.

## What's Next
Hope you're still with me because there's more to come! At this point, we have only touched on the basic terminology and some simple architectures. In the next post, we'll be looking into the actual implementation of SPNs in a tensorized fashion, after which it becomes easy to experiment with them.

## References
1. Poon, Hoifung, and Pedro Domingos. "Sum-product networks: A new deep architecture." 2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops). IEEE, 2011.
2. Zheng, Kaiyu, Andrzej Pronobis, and Rajesh PN Rao. "Learning graph-structured sum-product networks for probabilistic semantic maps." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
3. Gens, Robert, and Pedro Domingos. "Discriminative learning of sum-product networks." Advances in Neural Information Processing Systems. 2012.
