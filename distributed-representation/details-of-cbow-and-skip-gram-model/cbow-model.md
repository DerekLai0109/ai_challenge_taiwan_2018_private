# CBOW Model with One-context word

Derek 03/09/2018

The CBOW model predicts one target word by its context \[1\].

Considering the CBOW model with the context word number of one, the CBOW model is reduced to a bigram model.

In the bigram model, a word is given to predict the next word.

![](/assets/import.png)

Architecture of the CBOW model with only one word in the context. $$\bar{\bar{W}}$$ and $$\bar{\bar{W}}'$$ are the input-to-hidden and hidden-to-output weight matrices.  
 $$\bar{v}_{w_k}$$ is the $$k$$-th row of $$\bar{\bar{W}}$$ ; $$\bar{v}_{w_j}'$$ is the $$j$$-th column of $$\bar{\bar{W}}'$$.

Fig shows the architecture of the CBOW model with only one context word.

The vocabulary size is _V_, which is the number of neurons in the input layer and the output layer, and the hidden layer size is _N_.  
 Usually, _V_ &gt;&gt; _N_.

The input is a one-hot encoded vector as


$$
\bar{x} = [x_1, x_2, \cdots, x_V]^t 
\tag{1}
$$


The one-hot encoding means that, for word $$w_k (k = 1,2, \cdots, V)$$, the $$k$$-th component is $$x_k = 1$$ and other components are $$x_{i \neq k} = 0$$.

The input-to-hidden weight between the neuron $$k$$ in the input layer and the neuron $$i$$ in the hidden layer is denoted as $$w_{ki}$$ , forming a $$V \times N$$ weight matrix as


$$
\bar{\bar{W}} = \left[
\begin{matrix}
w_{11} & w_{12} & \cdots & \cdots & w_{1N} \\ 
w_{21} & w_{22} & \cdots& \cdots  & w_{2N} \\ 
\vdots & \cdots & \ddots& \cdots  & \vdots \\ 
w_{k1} & \cdots & w_{ki}& \cdots  & w_{kN} \\ 
\vdots & \cdots & \ddots& \cdots  & \vdots \\ 
w_{V1} & w_{V2} & \cdots & \cdots & w_{VN}
\end{matrix}
\right] \tag{2}
$$


Thus, output of the hidden layer is obtained as


$$
\bar{h} = \bar{\bar{W}}^t \cdot \bar{x} = \bar{v}_{w_k}
\tag{3}
$$


where the superscript $$t$$ means transpose and $$\bar{v}_{w_k}$$ is the $$N$$-dimensional vector representation of the input word $$w_k$$ and its components are the values in the $$k$$-th row of $$\bar{\bar{W}}$$.

The hidden-to-output weights are denoted as $$w_{ij}'$$, which form a $$N \times V$$ weight matrix $$\bar{\bar{W}}'$$ as


$$
\bar{\bar{W}}' = \left[
\begin{matrix}
w_{11}' & w_{12}' & \cdots & \cdots & w_{1V}' \\ 
w_{21}' & w_{22}' & \cdots& \cdots  & w_{2V}' \\ 
\vdots & \cdots & \ddots& \cdots  & \vdots \\ 
w_{i1}' & \cdots & w_{ij}' & \cdots  & w_{iV}' \\ 
\vdots & \cdots & \ddots& \cdots  & \vdots \\ 
w_{N1}' & w_{N2}' & \cdots & \cdots & w_{NV}'
\end{matrix}
\right] \tag{3}
$$


The vector $$\bar{h}$$ is weighted by $$\bar{\bar{W}}'$$ to obtain the input of the output layer as


$$
\bar{u} = \bar{\bar{W}}'^t \cdot \bar{h} = \left[ \bar{v}'_{w_1}, \cdot, \bar{v}'_{w_j}, \cdot, \bar{v}'_{w_V}\right]^t \cdot \bar{h} 
\tag{4}
$$


namely,


$$
u_j = \bar{v}'_{w_j} \cdot \bar{v}_{w_k}, \ j = 1, \cdots, V 
\tag{5}
$$


where $$\bar{v}_{w_j}'$$ is the $$j$$-th column of $$\bar{\bar{W}}'$$. The $$j$$-th neuron in the output layer has a softmax output as


$$
y_j = p(w_j | w_k)= \frac{e^{u_j}}{\displaystyle \sum_{j = 1}^V e^{u_j}} 
\tag{6}
$$
By substituting $$(5)$$ into $$(6)$$, we have 
$$
y_j = p(w_j | w_k) = \frac{e^{\bar{v}'_{w_j} \cdot \bar{v}_{w_k}}}{\displaystyle \sum_{j = 1}^V e^{\bar{v}'_{w_j} \cdot \bar{v}_{w_k}}} 
\tag{7}
$$
Note that $$\bar{v}_w$$ and $$\bar{v}'_w$$ are two representations of word $$w$$ and are called input vector and output vector, respectively.The training objective is to maximize the probability of observing the target word $$w_{j_o}$$ given the input context word $$w_k$$.

\[1\]

T. Mikolov, K. Chen, G. Corrado and J. Dean, Efficient estimation of word representations in vector space,

arXiv:1301.3781, 2013.

