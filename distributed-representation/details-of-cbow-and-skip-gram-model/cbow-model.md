# CBOW Model with One-word Context

Derek, 03/24/2018

The CBOW model predicts one target word by its context words \[1\].

Considering the CBOW model with one context word, which is reduced to a bigram model.

In the bigram model, a word \(context word\) is given to predict the next word \(target word\).

Define a vocabulary containing V words as


$$
{\cal V} = \{ d_1, d_2, \cdots, d_k, \cdots, d_V\}
$$


where $$d_k$$ is the  $$k$$-th word in the vocabulary.

![](/assets/data_flow_of_CBOW_one_word_context.jpg)**Fig.1. Data flow of CBOW model with one-word context.**$$\bar{x}^k$$** is the one-hot encoded vector of word **$$d_k$$** and is the input of NN for CBOW model with one-word context. **$$\bar{y}$$** is the output of the NN. **$$\bar{v}_k$$** and **$$\bar{v}_j'$$** are word vectors and are named input vector and ouput vector, respectively.**

Fig.1. shows the  data flow of CBOW model with one-word context. $$\bar{x}^k$$ is the one-hot encoded vector of word $$d_k$$ and is the input of the neural network \(NN\) for CBOW model with one-word context, expanded as


$$
\bar{x}^k = [x_1^k, x_2^k, \cdots, x_k^k, \cdots, x_V^k]^t \tag{1}
$$


where $$x_n^k = 0$$ for $$n \neq k$$ and $$x_k^k = 1$$; $$t$$ stands for transpose operation. A corpus $${\cal C}$$ will be used to train the NN, which can be constituted by $$N_a$$ articles as $${\cal C} = \{ {\rm article}_1, {\rm article}_2, \cdots, {\rm article}_{N_a} \}$$. Each article is constituted by the words in the vocabulary $${\cal V}$$. For example, $${\rm article}_1 = 'd_1 \ d_5 \ d_{18} \ d_{56} \ d_2 \ \cdots '$$.  The NN is trained by inputting the articles to the NN word by word.

![](/assets/CBOW_1word_1.jpg)

**Fig.2. Architecture of the NN for the CBOW model with one context word. **$$\bar{\bar{W}}$$** and **$$\bar{\bar{W}}'$$** are the input-to-hidden and hidden-to-output weight matrices. **$$\bar{v}_k$$** is the transpose of the **$$k$$**-th row of **$$\bar{\bar{W}}$$**  and is a word vector representation of **$$d_k$$, **named input vector; **$$\bar{v}_j'$$** is the **$$j$$**-th column of **$$\bar{\bar{W}}'$$ **and is a word vector representation of **$$d_j$$, **named output vector.**

Fig.2 shows the architecture of the NN for the CBOW model with one context word.

The neuron numbers in the input layer and in the output layer are both chosen to be the vocabulary size $$V$$, and the hidden layer size is _N_. Usually, _V_ &gt;&gt; _N_. For example, $$V = 8000$$ and $$N = 60$$ or $$100$$.

## Forward pass

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


where the $$k$$-th row of $$\bar{\bar{W}}$$ contains the weights whcih connect the neuron $$k$$ in the input layer to all neurons in the hidden layer as shown in Fig.2. Define the transpose of the $$k$$-th row of $$\bar{\bar{W}}$$ as the input vector $$\bar{v}_k$$, namely,


$$
\bar{v}_{k} \doteq \left[ 
\begin{matrix}
w_{k1}, \cdots, w_{ki}, \cdots, w_{kN}
\end{matrix}
\right]^t
$$


which is the $$N$$-dimensional vector representation of the input word $$d_k$$.

The hidden-layer output is obtained as


$$
\bar{h} = \bar{\bar{W}}^t \bar{x}^k 
\\
 = \left[
\begin{matrix}
w_{11} & w_{21} & \cdots & w_{k1} & \cdots & w_{V1} \\ 
w_{12} & w_{22} & \cdots & \cdots & \cdots & w_{V2} \\ 
\vdots & \cdots & \cdots & w_{ki}& \cdots  & \vdots \\ 
\vdots & \cdots & \ddots& \cdots & \vdots & \vdots \\ 
w_{1N} & w_{2N} & \cdots & w_{kN} & \cdots & w_{VN}
\end{matrix}
\right] 
\left[ 
\begin{matrix}
0 \\
\vdots \\
0 \\
x_k^k = 1 \\
0 \\
\vdots \\
0
\end{matrix}
\right] 
= \left[ 
\begin{matrix}
w_{k1} \\
\vdots \\
w_{ki} \\
\vdots \\
w_{kN}
\end{matrix}
\right]
= \bar{v}_{k}
\tag{3}
$$


The hidden-to-output weights are denoted as $$w_{ij}'$$, which connect the neuron $$i$$ in the hidden layer and the neuron $$j$$ in the output layer and form an $$N \times V$$ weight matrix $$\bar{\bar{W}}'$$ as


$$
\bar{\bar{W}}' = \left[
\begin{matrix}
w_{11}' & w_{12}' & \cdots & w_{1j}' & \cdots & w_{1V}' \\ 
w_{21}' & w_{22}' & \cdots & w_{2j}'& \cdots  & w_{2V}' \\ 
\vdots & \cdots & \ddots & \cdots & \cdots  & \vdots \\ 
w_{i1}' & \cdots & \cdots & w_{ij}' & \cdots  & w_{iV}' \\ 
\vdots & \cdots & \ddots & \cdots & \cdots  & \vdots \\ 
w_{N1}' & w_{N2}' & \cdots & w_{Nj}' & \cdots & w_{NV}'
\end{matrix}
\right] \tag{4}
$$


where the $$j$$-th column contains the weights which connect all neurons in the hidden layer to the $$j$$-th neuron in the output layer as shown in Fig.2. Define the $$j$$-th column of $$\bar{\bar{W}}'$$ as the output vector $$\bar{v}_j'$$, namely,


$$
\bar{v}_j' = [w_{1j}', w_{2j}', \cdots, w_{ij}', \cdots, w_{Nj}']^t \tag{5}
$$


Note that the output vector $$\bar{v}_k'$$ is another $$N$$-dimensional vector representation of the input word $$d_k$$. By substituting $$(5)$$ into $$(4)$$, we can represent $$\bar{\bar{W}}'$$ as


$$
\bar{\bar{W}}' = \left[ \bar{v}'_1, \cdots, \bar{v}'_j, \cdots, \bar{v}'_V \right] \tag{6}
$$


The vector $$\bar{h}$$ in $$(3)$$ is weighted by $$\bar{\bar{W}}'$$ to obtain the input of the output layer as


$$
\bar{u} = \bar{\bar{W}}'^t \cdot \bar{h} = \left[ 
\begin{matrix}
w_{11}' & w_{21}' & \cdots & w_{i1}' & \cdots & w_{N1}' \\ 
w_{12}' & w_{22}' & \cdots & \vdots & \cdots & w_{N2}' \\ 
\vdots & \vdots & \ddots & w_{ij}' & \cdots & \vdots \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 
w_{1V}' & w_{2V}' & \cdots & w_{iV}' & \cdots & w_{NV}'
\end{matrix}
\right] \left[ 
\begin{matrix}
w_{k1} \\
\vdots \\
w_{ki} \\
\vdots \\
w_{kN}
\end{matrix}
\right] = \left[ \bar{v}'_1, \cdots, \bar{v}'_j, \cdots, \bar{v}'_V \right]^t \bar{v}_k
$$


which can be represented as


$$
\bar{u} = \left[ \bar{v}'_1, \cdots, \bar{v}'_j, \cdots, \bar{v}'_V \right]^t \bar{v}_k
\tag{7}
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

\[0\]

X. Rong, word2vec parameter learning explained, arXiv:1411.2738, 2014.

\[1\]

T. Mikolov, K. Chen, G. Corrado and J. Dean, Efficient estimation of word representations in vector space,

arXiv:1301.3781, 2013.

