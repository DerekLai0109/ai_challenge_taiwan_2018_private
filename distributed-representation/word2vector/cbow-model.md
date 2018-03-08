# One-context word

The CBOW model predicts one target word by its context \[1\].

Considering the CBOW model with the context word number of one, the CBOW model is reduced to a bigram model.

In the bigram model, a word is given to predict the next word.

![](/assets/import.png)Architecture of the CBOW model with only one word in the context. _W_ and _W'_ are the input-to-hidden and hidden-to-output weight matrices.v\_{w\_k} is the k-th row of W; v\_{w\_j}' is the j-th column of _W'._

Fig shows the architecture of the CBOW model with only one context word.

The vocabulary size is _V_, which is the number of neurons in the input layer and the output layer, and the hidden layer size is _N_.  
 Usually, _V_ &gt;&gt; _N_.

The input is a one-hot encoded vector as

$$a \ne 0$$

\[1\]

T. Mikolov, K. Chen, G. Corrado and J. Dean, Efficient estimation of word representations in vector space,

arXiv:1301.3781, 2013.

