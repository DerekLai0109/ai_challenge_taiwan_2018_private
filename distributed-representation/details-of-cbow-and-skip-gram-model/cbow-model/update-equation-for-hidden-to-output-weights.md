# Backward Pass

Derek 03/24/2018

The loss function is


$$
E = -\ln y_{j_o} = - u_{j_o} + \ln \left(\sum_{j=1}^V e^{u_j} \right)
\tag{1}
$$


The partial derivative of $$E$$ with respect to $$u_j$$ is


$$
\frac{\partial E}{\partial u_j} = y_j - \delta_{j j_o} \doteq e_j \tag{2}
$$


where $$\delta_{j j_o} = 1$$ for $$j = j_o$$ and $$\delta_{j j_o} = 0$$ for $$j \neq j_o$$. The supporting material is


$$
\frac{\partial E}{\partial u_{j_o}} = -1 + \frac{e^{u_{j_o}}  }{\displaystyle \sum_{j=1}^V e^{u_j}} 
 = y_{j_o} - 1 \\ 
\frac{\partial E}{\partial u_j} = \frac{e^{u_j}  }{\displaystyle \sum_{j=1}^V e^{u_j}} = y_j,  \ j \neq j_o
$$


The derivative of $$E$$ to the hidden-to-output weight $$w_{ij}'$$ is


$$
\frac{\partial E}{\partial w'_{ij}} = \frac{\partial E}{\partial u_j} \frac{\partial u_j}{\partial w'_{ij}} = e_j w_{ki} 
\tag{2}
$$


The supporting material of $$(2)$$ is


$$
u_j = \bar{v}_j' \cdot \bar{v}_k = \sum_{i = 1}^N w_{ij}' w_{ki} \\  
\frac{\partial u_j}{\partial w'_{ij}} = w_{ki}
$$


By using stochastic gradient descent, we obtain the updating equation for  hidden-to-output weights $$w_{ij}'$$ as


$$
w_{ij}'^{(new)} = w_{ij}'^{(old)} - \eta \frac{\partial E}{\partial w'_{ij}}
= w_{ij}'^{(old)} - \eta e_j w_{ki}^{(old)} \tag{3}
$$


or equivalently


$$
\bar{v}_j'^{(new)} = \bar{v}_j'^{(old)} - \eta e_j \bar{v}_k^{(old)} \tag{4}
$$


where $$\eta > 0$$ is the learning rate.

At $$j \neq j_o$$, the error in $$(2)$$ is greater than zero $$e_j > 0$$ \(overestimating\) and, in $$(4)$$,  $$\bar{v}_j'^{(old)}$$ subtract a scaled $$\bar{v}_k^{(old)}$$such that the angle between $$\bar{v}_j'^{(new)}$$ and $$\bar{v}_k^{(old)}$$ increases.At $$j = j_o$$, the error in $$(2)$$ is smaller than zero $$e_j < 0$$ \(underestimating\) and $$\bar{v}_j'^{(old)}$$ add a scaled $$\bar{v}_k^{(old)}$$such that the angle between $$\bar{v}_j'^{(new)}$$ and $$\bar{v}_k^{(old)}$$ decreases. If $$y_{j_o}$$ is close to 1, the error is close to 0 and $$\bar{v}'_{j_o}$$ is nearly unchanged.

Next, find the update equation for input-to-hidden weights. The derivative of $$E$$  to the output of the hidden layer $$h_i$$ is


$$
\frac{\partial E}{\partial h_i} = \sum_{j = 1}^V \frac{\partial E}{\partial u_j} \frac{\partial u_j}{\partial h_i} 
 = \sum_{j = 1}^V e_j w'_{ij} \tag{5}
$$


The supporting material of $$(5)$$ is
$$
h_i = w_{ki}
$$
in the CBOW model with one context word and
$$
\frac{\partial u_j}{\partial h_i} = \frac{\partial u_j}{\partial w_{ki}} = w_{ij}'
$$
The derivative of $$E$$ to the input-to-hidden weights $$w_{ki}$$ is


$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial h_i} \frac{\partial h_i}{\partial w_{ki}} = \frac{\partial E}{\partial h_i}
  = {\displaystyle \sum_{j = 1}^V} e_j w'_{ij} \tag{6}
$$


Thus, the update equation for input-to-hidden weights is


$$
w_{ki}^{(new)} = w_{ki}^{(old)} - \eta \frac{\partial E}{\partial w_{ki}}  = w_{ki}^{(old)} - \eta \sum_{j = 1}^V e_j w_{ij}'^{(old)}
$$


or


$$
\bar{v}_{w_k}^{(new)} = \bar{v}_{w_k}^{(old)} - \eta \sum_{j = 1}^V e_j \bar{v}_{w_j}'^{(old)} \tag{8}
$$


The input vector $$\bar{v}_{w_k}$$ is updated by adding the sum of scaled output vectors $$\bar{v}_{w_j}'$$.For a specific neuron $$j$$, if the probability $$y_j$$ is overestimated \($$y_j > \delta_{j j_o}$$\), the contribution of the output vector $$\bar{v}_{w_j}'$$ will put the input vector $$\bar{v}_{w_k}$$ farther away from $$\bar{v}_{w_j}'$$. Conversely, if the probability $$y_j$$ is underestimated \($$y_j < \delta_{j j_o}$$\), the contribution of the output vector $$\bar{v}_{w_j}'$$ will move the input vector $$\bar{v}_{w_k}$$ closer to $$\bar{v}_{w_j}'$$. If the contribution of all the output vectors is nearly zero, the input vector remains nearly unchanged.

The weights will be updated iteratively by inputting the context-target word pairs generated from a training corpus.

