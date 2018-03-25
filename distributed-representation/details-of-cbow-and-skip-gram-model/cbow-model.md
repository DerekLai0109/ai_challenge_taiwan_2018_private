# CBOW Model with One-word Context

Derek, 03/24/2018

The CBOW model predicts one target word by its context words \[1\].

Consider the CBOW model with one context word, which is reduced to a bigram model.

In the bigram model, a word \(context word\) is given to predict the next word \(target word\).

Define a vocabulary containing V words as


$$
{\cal V} = \{ d_1, d_2, \cdots, d_k, \cdots, d_V\}
$$


where $$d_k$$ is the  $$k$$-th word in the vocabulary. The training corpus $${\cal C}$$  can be constituted by $$N_a$$ articles as $${\cal C} = \{ {\rm article}_1, {\rm article}_2, \cdots, {\rm article}_{N_a} \}$$.Each article is constituted by the words in the vocabulary $${\cal V}$$. For example, $${\rm article}_1 = 'd_1 \ d_5 \ d_{18} \ d_{56} \ d_2 \ \cdots '$$.

![](/assets/data_flow_of_CBOW_one_word_context.jpg)**Fig.1. Data flow of CBOW model with one-word context.**$$\bar{x}^k$$** is the one-hot encoded vector of word **$$d_k$$** and is input to the NN for CBOW model with one-word context. **$$\bar{y}$$** is the output of the NN. The input vector **$$\bar{v}_k$$** and output vector **$$\bar{v}_j'$$** are two kinds of word vector representations.**

Fig.1. shows the  data flow of CBOW model with one-word context.The word $$d_k$$ is one-hot encoded into $$\bar{x}^k$$ and $$\bar{x}^k$$ is input to the neural network \(NN\) for CBOW model with one-word context, expanded as


$$
\bar{x}^k = [x_1^k, x_2^k, \cdots, x_k^k, \cdots, x_V^k]^t \tag{1}
$$


where $$x_n^k = 0$$ for $$n \neq k$$ and $$x_k^k = 1$$; $$t$$ stands for transpose operation. The output $$\bar{y} = [y_1, \cdots, y_j, \cdots, y_V]^t$$ has the size of $$V$$ and $$y_j$$ is a probability that the next word is $$d_j$$ given the one-hot encoded vector $$\bar{x}^k$$. The input vector $$\bar{v}_k$$ and output vector $$\bar{v}_j'$$ are two  kinds of word vector representations and will be elaborated later. The NN is trained by inputting the articles in the training corpus $${\cal C}$$ to the NN word by word.

![](/assets/schematic_of_y.jpg)**Fig.2. Schematic of NN output **$$\bar{y}$$** given a specific **$$\bar{x}^k$$** with the target word **$$d_{j_o}$$**. \(a\) non-trained NN, \(b\) well-trained NN**.

Fig.2. shows the schematic of NN output $$\bar{y}$$ given a specific $$\bar{x}^k$$ with the target word $$d_{j_o}$$. Fig.2\(a\) and \(b\) shows $$\bar{y}$$ of a non-trained NN and a well-trained NN, respectively.

![](/assets/CBOW_1word_1.jpg)

**Fig.2. Architecture of the NN for the CBOW model with one context word. **$$\bar{\bar{W}}$$** and **$$\bar{\bar{W}}'$$** are the input-to-hidden and hidden-to-output weight matrices. **$$\bar{v}_k$$** is the transpose of the **$$k$$**-th row of **$$\bar{\bar{W}}$$**  and is the first kind of word vector representation of **$$d_k$$, **named input vector; **$$\bar{v}_j'$$** is the **$$j$$**-th column of **$$\bar{\bar{W}}'$$ **and is the second kind of word vector representation of **$$d_j$$, **named output vector.**

Fig.2 shows the architecture of the NN for the CBOW model with one context word.$$\bar{\bar{W}}$$ and $$\bar{\bar{W}}'$$ are the input-to-hidden and hidden-to-output weight matrices. $$\bar{v}_k$$ is the transpose of the $$k$$-th row of $$\bar{\bar{W}}$$  and is the first kind of word vector representation of $$d_k$$, named input vector; $$\bar{v}_j'$$ is the $$j$$-th column of $$\bar{\bar{W}}'$$ and is the second kind of word vector representation of $$d_j$$, named output vector.

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
\bar{v}_j' \doteq [w_{1j}', w_{2j}', \cdots, w_{ij}', \cdots, w_{Nj}']^t \tag{5}
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
= \left[ \begin{matrix}
u_1 \\
\vdots \\
u_j \\
\vdots \\
u_V
\end{matrix} \right] 
= \left[ \begin{matrix}
\bar{v}_1'^t \bar{v}_k \\
\vdots \\
\bar{v}_j'^t \bar{v}_k \\
\vdots \\
\bar{v}_V'^t \bar{v}_k
\end{matrix} \right] 
= \left[ \begin{matrix}
\bar{v}_1' \cdot \bar{v}_k \\
\vdots \\
\bar{v}_j' \cdot \bar{v}_k \\
\vdots \\
\bar{v}_V' \cdot \bar{v}_k
\end{matrix} \right]
\tag{7}
$$


The output $$y_j$$ of the $$j$$-th neuron in the output layer is a probability that the next word is $$d_j$$ given the one-hot encoded vector $$\bar{x}_k$$ as


$$
y_j = p(d_j | \bar{x}_k)= \frac{e^{u_j}}{\displaystyle \sum_{j = 1}^V e^{u_j}} = \frac{e^{\bar{v}'_j \cdot \bar{v}_k}}{\displaystyle \sum_{j = 1}^V e^{\bar{v}'_j \cdot \bar{v}_k}} 
\tag{8}
$$


The training objective is to maximize the probability $$y_{j_o}$$ of observing the target word $$d_{j_o}$$ given $$\bar{x}_k$$. The loss function is defined as


$$
E = -\ln p(d_{j_o} | \bar{x}_k) 
\tag{9}
$$


Thus, to maximize $$y_{j_o} = p(d_{j_o} | \bar{x}_k)$$ is to minimize $$E$$.  The loss function is a special case of the cross-entropy measurement between two probabilistic distributions.

\[0\]

X. Rong, word2vec parameter learning explained, arXiv:1411.2738, 2014.

\[1\]

T. Mikolov, K. Chen, G. Corrado and J. Dean, Efficient estimation of word representations in vector space,

arXiv:1301.3781, 2013.

