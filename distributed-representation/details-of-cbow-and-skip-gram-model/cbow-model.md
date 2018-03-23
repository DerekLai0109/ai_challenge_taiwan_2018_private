# CBOW Model with One-word Context

Derek, 03/09/2018 created; 03/23/2018 last modified

The CBOW model predicts one target word by its context words \[1\].

Considering the CBOW model with one context word, which is reduced to a bigram model.

In the bigram model, a word \(context word\) is given to predict the next word \(target word\).

Define a vocabulary containing V words as


$$
{\cal V} = \{ d_1, d_2, \cdots, d_k, \cdots, d_V\}
$$


where $$d_k$$ is the  $$k$$-th word in the vocabulary.

![](/assets/data_flow_of_CBOW_one_word_context.jpg)**Fig.1. Data flow of CBOW model with one-word context.**$$\bar{x}^k$$** is the one-hot encoded vector of word **$$d_k$$** and is the input of the neural network \(NN\) for CBOW model with one-word context. **$$\bar{y}$$** is the output of the NN. **$$\bar{v}_k$$** and **$$\bar{v}_j'$$** are word vectors and are named input vector and ouput vector, respectively.**

Fig.1. shows the  data flow of CBOW model with one-word context. $$\bar{x}^k$$ is the one-hot encoded vector of word $$d_k$$ and is the input of the neural network \(NN\) for CBOW model with one-word context, expanded as


$$
\bar{x}^k = [x_1^k, x_2^k, \cdots, x_k^k, \cdots, x_V^k]^t, \tag{1}
$$


where $$x_n^k = 0$$ for $$n \neq k$$ and $$x_k^k = 1$$; $$t$$ stands for transpose operation.

![](/assets/import.png)

**Fig.2. Architecture of NN for the CBOW model with one context word. **$$\bar{\bar{W}}$$** and **$$\bar{\bar{W}}'$$** are the input-to-hidden and hidden-to-output weight matrices. **$$\bar{v}_k$$** is the **$$k$$**-th row of **$$\bar{\bar{W}}$$**  and is a word vector representation of **$$d_k$$, **named input vector; **$$\bar{v}_j'$$** is the **$$j$$**-th column of **$$\bar{\bar{W}}'$$ **and is a word vector representation of **$$d_j$$, **named output vector.**

Fig.2 shows the architecture of the neural network \(NN\) for the CBOW model with one context word.

The the number of neurons in the input layer and in the output layer are both chosen to be the vocabulary size $$V$$, and the hidden layer size is _N_. Usually, _V_ &gt;&gt; _N_.

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


Note that $$\bar{v}_w$$ and $$\bar{v}'_w$$ are two representations of word $$w$$ and are called input vector and output vector, respectively.

The training objective is to maximize the probability of observing the target word $$w_{j_o}$$ given the input context word $$w_k$$.Define the loss function as


$$
E = -\ln p(w_{j_o} | w_k) 
\tag{8}
$$


Thus, to maximize $$p(w_{j_o} | w_k)$$ is to minimize $$E$$.

The loss function is a special case of the cross-entropy measurement between two probabilistic distributions.

\[1\]

T. Mikolov, K. Chen, G. Corrado and J. Dean, Efficient estimation of word representations in vector space,

arXiv:1301.3781, 2013.

