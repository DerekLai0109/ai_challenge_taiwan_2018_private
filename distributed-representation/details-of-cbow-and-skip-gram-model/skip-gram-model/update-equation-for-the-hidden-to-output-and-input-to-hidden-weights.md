# Update equation for the hidden-to-output weights

Derek 03/09/2018

The input of $$j$$-th neuron on $m$-th panel in the output layer is obtained as


$$
u_{j, m} = u_j = \bar{v}_{w_j}' \cdot \bar{h} = \sum_{i = 1}^N w_{ij}' h_i = \sum_{i = 1}^N w_{ij}' w_{ki} \tag{1}
$$


and the derivative of $$E$$  to $$u_{j,m}$$ is


$$
\frac{\partial E}{\partial u_{j,m}} = y_{j,m} - \delta_{j j_{o,m}} \doteq e_{j,m} \tag{2}
$$


By using $$(1)$$ and $$(2)$$, the derivative of $$E$$ to $$w_{ij}'$$ is obtained as


$$
\frac{\partial E}{\partial w_{ij}'} = \sum_{m = 1}^C \frac{\partial E}{\partial u_{j,m}} \frac{\partial u_{j,m}}{\partial w_{ij}'}
 = \sum_{m = 1}^C e_{j,m} h_i \tag{3}
$$


by which, the update equation for the hidden-to-output weights is


$$
w_{ij}'^{(new)} = w_{ij}'^{(old)} - \eta \frac{\partial E}{\partial w_{ij}'} = w_{ij}'^{(old)} - \eta \sum_{m = 1}^C e_{j,m} h_i \tag{4}
$$


which can also be represented as


$$
\bar{v}_{w_j}'^{(new)} = \bar{v}_{w_j}'^{(old)} - \eta \sum_{m = 1}^C e_{j,m} \bar{h}  = \bar{v}_{w_j}'^{(old)} - \eta \sum_{m = 1}^C e_{j,m} \bar{v}_{w_k}
, \ j = 1, 2, \cdots, V \tag{5}
$$


where the prediction error $$e_{j,m}$$ is summed across all context words in the output layer.  
 Note that it is needed to apply the update equation for every hidden-to-output weight for each training instance.

# Update equation for input-to-hidden weights

The output of the hidden layer is
$$
h_i = v_{w_k, i} = w_{ki} \tag{6}
$$
By using $$(1)$$ and $$(2)$$, the derivative of $$E$$ to $$h_i$$ is obtained as


$$
\frac{\partial E}{\partial h_i} = \sum_{j = 1}^V \sum_{m = 1}^C \frac{\partial E}{\partial u_{j,m}} \frac{\partial u_{j,m}}{\partial h_i}
 = \sum_{j = 1}^V \sum_{m = 1}^C e_{j,m} w_{ij}' \tag{7}
$$
By using $$(6)$$ and $$(7)$$, we can obtain the derivative of $$E$$ to $$w_{ki}$$ as 
$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial h_i} \frac{\partial h_i}{\partial w_{ki}} = \sum_{j = 1}^V \sum_{m = 1}^C e_{j,m} w_{ij}' \tag{8}
$$
Thus, the update equation for the input-to-hidden weights is 
$$
w_{ki}^{(new)} = w_{ki}^{(old)} - \eta \frac{\partial E}{\partial w_{ki}} = w_{ki}^{(old)} - \eta \sum_{j = 1}^V \sum_{m = 1}^C e_{j,m} w_{ij}' \tag{9}
$$
or equivalently 
$$
\bar{v}_{w_k}^{(new)} = \bar{v}_{w_k}^{(old)} - \eta \sum_{j = 1}^V \sum_{m = 1}^C e_{j,m} \bar{v}_{w_j}' \tag{10}
$$


