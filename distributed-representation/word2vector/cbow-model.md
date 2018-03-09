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

$$\bar{x} = [x_1, x_2, \cdots, x_V]^t$$



$$\left[\begin{matrix}1 & x & x^2 \\1 & y & y^2 \\1 & z & z^2 \\\end{matrix}\right]$$





\[1\]

T. Mikolov, K. Chen, G. Corrado and J. Dean, Efficient estimation of word representations in vector space,

arXiv:1301.3781, 2013.

