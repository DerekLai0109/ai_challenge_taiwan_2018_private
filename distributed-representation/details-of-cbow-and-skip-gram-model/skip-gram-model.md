# Skip-gram Model

Derek 03/09/2018

![](/assets/skip_gram_model.png)Architecture of the skip-gram model.

Fig. shows the architecture of the skip-gram model.A word is input at the input layer to predict its context words set as the target words at the output layer.

The output of the hidden layer is


$$
\bar{h} = \bar{v}_{w_k} \tag{1}
$$


or represented component-wise as


$$
h_i = v_{w_k, i} = w_{ki} \tag{2}
$$


On the output layer, instead of outputing one multinomial distribution, output $$C$$ multinomial distribtions with each multinomial distribtion computed using the same hidden-to-output weight matrix $$\bar{\bar{W}}'$$. The input of $$j$$-th neuron on $m$-th panel in the output layer is obtained as


$$
u_{j, m} = u_j = \bar{v}_{w_j}' \cdot \bar{h} = \sum_{i = 1}^N w_{ij}' h_i = \sum_{i = 1}^N w_{ij}' w_{ki} \tag{3}
$$


where $$u_{j, m}$$ of all panels are the same since they share the same weights. The probability of $$j$$-th \($$j = 1, 2, \cdots, V$$\) word on $$m$$-th panel is


$$
y_{j, m} = p(w_{j, m}| w_k) = \frac{e^{u_{j, m}}}{\displaystyle \sum_{j' = 1}^V e^{u_{j'}}}
 = \frac{e^{\bar{v}_{w_j}' \cdot \bar{h} }}{\displaystyle \sum_{j' = 1}^V e^{\bar{v}_{w_{j'}}' \cdot \bar{h}}} \tag{4}
$$


It is desired to maximize the probability of the context words given the input word.

The loss function is defined as


$$
E = -\ln p(w_{j_{o, 1}}, w_{j_{o, 2}}, \cdots, w_{j_{o, C}} | w_k) = -\ln \prod_{m = 1}^C p(w_{j_{o, m}} | w_k)
= -\ln \prod_{m = 1}^C y_{j_{o,m}, m} = -\ln \prod_{m = 1}^C \frac{e^{\bar{v}_{w_{j_{o,m}}}' \cdot \bar{h} }}{\displaystyle \sum_{j' = 1}^V e^{\bar{v}_{w_{j'}}' \cdot \bar{h}}} \tag{5}
$$


where the subscript $$j_{o,m}$$ means the word with the subscript is the $$m$$-th target context word in $$Cx(w_k)$$.

The derivative of $$E$$  to $$u_{j,m}$$ is


$$
\frac{\partial E}{\partial u_{j,m}} = y_{j,m} - \delta_{j j_{o,m}} \doteq e_{j,m} \tag{6}
$$


