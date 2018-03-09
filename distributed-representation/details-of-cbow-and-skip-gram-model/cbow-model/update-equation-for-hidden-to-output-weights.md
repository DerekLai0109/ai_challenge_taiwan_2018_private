# Update equation for hidden-to-output weights

The partial derivative of $$E$$ with respect to $$j$$-th neuron input $$u_j$$ is 
$$
\frac{\partial E}{\partial u_j} = y_j - \delta_{j j_o} \doteq e_j \tag{1}
$$
where $$\delta_{j j_o} = 1$$ for $$j = j_o$$ and $$\delta_{j j_o} = 0$$ for $$j \neq j_o$$.

The derivative to the hidden-to-output weight $$w_{ij}'$$ is 
$$
\frac{\partial E}{\partial w'_{ij}} = \frac{\partial E}{\partial u_j} \frac{\partial E}{\partial w'_{ij}} = e_j h_i = e_j w_{ki} 
\tag{2}
$$
Therefore, using stochastic gradient descent, we obtain the weight updating equation for hidden-to-output weights as 
$$
w_{ij}'^{(new)} = w_{ij}'^{(old)} - \eta \frac{\partial E}{\partial w'_{ij}}
= w_{ij}'^{(old)} - \eta e_j h_i = w_{ij}'^{(old)} - \eta e_j w_{ki}^{(old)} \tag{3}
$$
or equivalently 
$$
\bar{v}_{w_j}'^{(new)} = \bar{v}_{w_j}'^{(old)} - \eta e_j \bar{h} 
= \bar{v}_{w_j}'^{(old)} - \eta e_j \bar{v}_{w_k}^{(old)}, \hbox{ for } j = 1, 2, \cdots, V \tag{4}
$$
where $$\eta > 0$$ is the learning rate. 

If $$j \neq j_o$$, the error is greater than zero $$e_j > 0$$ \(overestimating\) and $$\bar{v}'_{w_j}$$ subtract a scaled $$\bar{v}_{w_k}$$such that the angle between $$\bar{v}'_{w_j}$$ and $$\bar{v}_{w_k}$$ increases.If $$j = j_o$$, the error is smaller than zero $$e_j < 0$$ \(underestimating\) and $$\bar{v}'_{w_j}$$ add a scaled $$\bar{v}_{w_k}$$such that the angle between $$\bar{v}'_{w_j}$$ and $$\bar{v}_{w_k}$$ decreases. If $$y_{j_o}$$ is close to 1, the error is close to 0 and $$\bar{v}'_{w_{j_o}}$$.

