# CBOW Model with Multi-word Context

Derek 03/09/2018

![](/assets/CBOW_cword.png)Architecture of CBOW model with $$C$$ context words.

Fig. shows the architecture of CBOW model with $$C$$ context words. A softmax function is still imposed at the end of the output layer.  
Define the context set formed by the context of target word $$w_{j_o}$$ as


$$
Cx (w_{j_o}) = \left\{ w_{c_m}| w_{c_m} \hbox{ is the word in the context of } w_{j_o}, m = 1, 2, \cdots, C \right\} \tag{1}
$$


where the subscript $$c_m$$ can be the integers between $$1$$ and $$V$$.

The hidden layer output is calcuted as


$$
\bar{h} = \frac{1}{C} \bar{\bar{W}}^t \cdot \left( \bar{x}_{c_1} + \bar{x}_{c_2} + \cdots + \bar{x}_{c_C} \right)
 = \frac{1}{C} \left( \bar{v}_{w_{c_1}} + \bar{v}_{w_{c_2}} + \cdots + \bar{v}_{w_{c_C}} \right) \tag{2}
$$


or represented component-wise as


$$
h_i = \frac{1}{C} \sum_{m = 1}^C w_{c_m i}, \ i = 1, 2, \cdots, N 
\tag{3}
$$


The output of the neural network at $$j$$-th neuron is the probability of word $$w_j$$ given the context of $$w_{j_o}$$, namely,


$$
y_j = p(w_j | Cx(w_{j_o}) ) = \frac{e^{\bar{v}_{w_j}' \cdot \bar{h}} }{\displaystyle \sum_{j = 1}^V e^{\bar{v}_{w_j}' \cdot \bar{h}}} \tag{4}
$$


The loss function is defined as


$$
E = -\ln p(w_{j_o}| Cx(w_{j_o})) \tag{5}
$$

