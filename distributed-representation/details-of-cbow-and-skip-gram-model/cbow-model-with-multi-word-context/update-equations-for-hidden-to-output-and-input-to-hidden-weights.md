# Update equation for the hidden-to-output weights

The update equation for the hidden-to-output weights is derived as


$$
w_{ij}'^{(new)} = w_{ij}'^{(old)} - \eta \frac{\partial E}{\partial w_{ij}'} = w_{ij}'^{(old)} - \eta e_j h_i \tag{1}
$$


or equivalently


$$
\bar{v}_{w_j}'^{(new)} = \bar{v}_{w_j}'^{(old)} - \eta e_j \bar{h} \tag{2}
$$


# Update equation for the input-to-hidden weights

The derivative of $$E$$  to $$h_i$$ is repeated as


$$
\frac{\partial E}{\partial h_i} = \sum_{j = 1}^V \frac{\partial E}{\partial u_j} \frac{\partial u_j}{\partial h_i} 
 = \sum_{j = 1}^V e_j w'_{ij} \tag{3}
$$


and


$$
h_i = \frac{1}{C} \sum_{m = 1}^C w_{c_m i}, \ i = 1, 2, \cdots, N 
\tag{4}
$$
By using $$(3)$$ and $$(4)$$, the derivative of $$E$$ to $$w_{ki}$$ is 
$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial h_i} \frac{\partial h_i}{\partial w_{ki}} = \frac{1}{C} \sum_{j = 1}^V e_j w'_{ij} \tag{5}
$$
by which, the update equation for the input-to-hidden weights is derived as 
$$
w_{ki}^{(new)} = w_{ki}^{(old)} - \eta \frac{\partial E}{\partial w_{ki}} = w_{ki}^{(old)} - \frac{\eta}{C} \sum_{j = 1}^V e_j w_{ij}'^{(old)} \tag{6}
$$
or equivalently 
$$
\bar{v}_{w_k}^{(new)} = \bar{v}_{w_k}^{(old)} - \frac{\eta}{C} \sum_{j = 1}^V e_j \bar{v}_{w_j}'^{(old)}, \ w_k \in Cx(w_{j_o}) \tag{7}
$$




