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
\bar{v}_{w_j}'^{(new)} = \bar{v}_{w_j}'^{(old)} - \eta e_j \bar{h} 
= \bar{v}_{w_j}'^{(old)} - \eta e_j \bar{v}_{w_k}^{(old)}, \hbox{ for } j = 1, 2, \cdots, V \tag{4}
$$


where $$\eta > 0$$ is the learning rate.

If $$j \neq j_o$$, the error is greater than zero $$e_j > 0$$ \(overestimating\) and $$\bar{v}'_{w_j}$$ subtract a scaled $$\bar{v}_{w_k}$$such that the angle between $$\bar{v}'_{w_j}$$ and $$\bar{v}_{w_k}$$ increases.If $$j = j_o$$, the error is smaller than zero $$e_j < 0$$ \(underestimating\) and $$\bar{v}'_{w_j}$$ add a scaled $$\bar{v}_{w_k}$$such that the angle between $$\bar{v}'_{w_j}$$ and $$\bar{v}_{w_k}$$ decreases. If $$y_{j_o}$$ is close to 1, the error is close to 0 and $$\bar{v}'_{w_{j_o}}$$.

## Update equation for input-to-hidden weights

The derivative of $$E$$  to $$h_i$$ is


$$
\frac{\partial E}{\partial h_i} = \sum_{j = 1}^V \frac{\partial E}{\partial u_j} \frac{\partial u_j}{\partial h_i} 
 = \sum_{j = 1}^V e_j w'_{ij} \tag{5}
$$


Since the inputs are one-hot encoding vectors, the output of the hidden layer is


$$
h_i = v_{w_k, i} = w_{ki} \tag{6}
$$


by which, the derivative of $$E$$ to the input-to-hidden weights $$w_{ki}$$ is equal to that to the output of the hidden layer $$h_i$$ as


$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial h_i} \frac{\partial h_i}{\partial w_{ki}} = \frac{\partial E}{\partial h_i}
  = {\displaystyle \sum_{j = 1}^V} e_j w'_{ij} \tag{7}
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

