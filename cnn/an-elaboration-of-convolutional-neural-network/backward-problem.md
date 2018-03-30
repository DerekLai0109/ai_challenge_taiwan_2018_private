#Network Training

Training algorithm are devised to perform different visual recognition tasks.The objective of network training is to minimize an error function, which is defined in terms of the network actual outputs and the desired outputs.

**Table 1 Notation for CNN training algorithm**

| description | symbol | formula |
| -- | -- | -- |
| training image index | $$k$$ | $$k=1,2,\cdots,K$$ |
| training image $$k$$ | $$\bar{\bar{x}}^k$$ | $$\bar{\bar{x}}=\{ x^k(i,j) \}, i=1,\cdots,H_0; j=1,\cdots,W_0$$ |
| desired output sample $$k$$ | $$\bar{\bar{d}}^k$$ | $$\bar{\bar{d}}^k=(d_1^k, d_2^k, \cdots, d^k_{N_L})^t$$ |
| network input or output of layer 0 | $$y^{0,k}(i,j)$$ | $$y^{0,k}=(x^k(i,j)),i=1,\cdots, H_0;j=1,\cdots,W_0$$ |
| weighted sum input to neuron ($$i,j$$) in convolution layer $$\ell$$, feature map $$n$$ | $$s_n^{\ell,k}(i,j)$$ | $$n=1,\cdots,N_\ell$$; $$i=1,\cdots, H_\ell$$; $$j=1,\cdots,W_\ell$$ |
| weighted sum input to neuron ($$i,j$$) in sub-sampling layer $$\ell$$, feature map $$n$$ | $$s_n^{\ell,k}(i,j)$$ | $$n=1,\cdots,N_\ell$$; $$i=1,\cdots, H_\ell$$; $$j=1,\cdots,W_\ell$$ |
| output of neuron ($$i,j$$) in convolution layer $$\ell$$, feature map $$n$$ for input image $$k$$ | $$y_n^{\ell,k}(i,j)$$ | $$y_n^{\ell,k}(i,j)=f_\ell(s_n^{\ell,k}(i,j))$$ |
| output of neuron ($$i,j$$) in sub-sampling layer $$\ell$$, feature map $$n$$ for input image $$k$$ | $$y_n^{\ell,k}(i,j)$$ | $$y_n^{\ell,k}(i,j)=f_\ell(s_n^{\ell,k}(i,j))$$ |
| $$n$$th error for image $$k$$ | $$e^k_n$$ | $$e^k_n = y_n^{L,k}-d_n^k$$ |
| error function | $$E(\bar{w})$$ | $$\displaystyle E(\bar{w})=E(\bar{w})=\frac{1}{N_L} \sum_{k=1}^K \sum_{n=1}^{N_L} (y_n^k - d_n^k)^2$$ |
| error sensitivity of pixel ($$i,j$$) in 2D layer $$\ell$$ | $$\delta_{i,j}^{\ell,k}$$ | $$\delta_{i,j}^{\ell,k}= \partial E/ \partial s_{i,j}^{\ell,k}$$ |
| error sensitivity of neuron $$n$$ in 1D layer $$\ell$$ | $$\delta_n^{\ell,k}$$ | $$\delta_n^{\ell,k}= \partial E/ \partial s_n^{\ell,k}$$ |

Table 1 summarizes the definitions used in CNN training, where $$\bar{\bar{x}}^k$$ is the $$k$$th training image, and $$\bar{\bar{d}}^k$$ is the corresponding desired output vector. $$K$$ is the number of input images in the training set. The loss function $$E(\bar{w})$$ is defined as

$$
E(\bar{\bar{w}},\bar{b}) = \frac{1}{K N_L} \sum_{k=1}^K \sum_{n=1}^{N_L} (y_n^{L,k}-d_n^{k})^2 
$$

where $$y_n^{L,k}$$ is the $n$th neuron output of output layer for $$k$$th pattern.

The loss function for softmax function is usually 

$$
E(\bar{\bar{w}},\bar{b}) = - \sum_{k=1}^{K} \sum_{n=1}^{N_L} d_{n}^k \ln y_{n}^{L,k}  \\
=- \sum_{n_L=1}^{N} d_{n}^k \ln \left( \frac{\exp( s_{n}^{L,k}) }{\displaystyle \sum_{n=1}^{N_L} \exp( s_{n}^{L,k})} \right) \nonumber \\
=- \sum_{n_L=1}^{N} d_{n}^k \left[ s_{n}^{L,k} - \ln \left(  \sum_{n=1}^{N_L} \exp( s_{n}^{L,k}) \right)  \right]
\nonumber
$$

where $$d_{n}^k$$ is the data label of the $k$th data, and is usually defined as

$$
d_{n}^k= \left \{
\begin{array}{ll}
1, & \hbox{if $n$ is the desired class for $k$th data} \\
0, & \hbox{otherwise}
\end{array}
\right.
\nonumber 
$$

Therefore, 

$$
\frac{\partial E}{\partial s_{n}^{L,k}} = \sum_{n_L=1}^{N_L} d_{n_L}^k \frac{\partial \ln y_{n_L}^{L,k}}{\partial s_{n}^{L,k}} = 
- \sum_{n_L=1}^{N_L}  d_{n_L}^k \frac{1}{y_{n_L}^{L,k}} \frac{\partial y_{n_L}^{L,k}}{\partial s_n^{L,k}} \nonumber \\
=- d_n^k (1-y_n^{L,k}) - \sum_{n_L \neq n} d_{n_L}^k \frac{1}{y_{n_L}^{L,k}} ( - y_{n_L}^{L,k} y_{n}^{L,k} ) \nonumber \\
=-d_n^k (1-y_{n}^{L,k}) + \sum_{n_L \neq n} d_{n_L}^k y_{n}^{L,k} \nonumber \\
=- d_n^k + d_n^k y_n^{L,k} + \sum_{n_L \neq n} d_{n_L}^k y_{n}^{L,k} \nonumber \\
= \sum_{n_L =1}^{N_L} d_{n_L}^k y_{n}^{L,k} -   d_n^k  =   \left( \sum_{n_L =1}^{N_L} d_{n_L}^k \right) y_{n}^{L,k} -   d_n^k  =y_{n}^{L,k} -   d_n^k \nonumber
$$

There are two major approaches to CNN training. The first approach is known as online training, which updates network parameters after each training sample is presented. This approach requires less memory, but it is less stable because each training sample can push the network parameters along a new direction.
The second approach is known as batch training, which updates network weights and biases after all training samples are presented. The batch training is focused in this work, because online training can be seen as a special case of batch training. An evaluation of network outputs for all training samples and an update of all network parameters are referred to collectively as a training epoch.

The error gradient, $$\delta_{w_{m,n}}^\ell = \partial E / \partial w_{m,n}^\ell$$ (for convolution layer), $$\delta_{w_{n}}^\ell  = \partial E / \partial w^\ell_n$$ (for sub-sampling layer) and $$\delta_{b_{n}}^\ell =  \partial E/ \partial b_n^\ell$$ (for both convolution and sub-sampling layers) are computed through error sensitivities $$\delta_n^{\ell}$$. The error sensitivity of neuron ($$i,j$$) in feature map $$n$$ of convolution layer or sub-sampling layer $$\ell$$, or output layer is defined as
$$
\delta_n^{\ell}(i,j)= \frac{\partial E}{\partial s_n^{\ell}(i,j)},  \\
\left\{ \begin{array}{ll}
\ell=1,3, \cdots, 2a+1, & \hbox{for convolution layer} \\
\ell=2,4, \cdots, 2a, & \hbox{for sub-sampling layer} \\
\ell=L, &  \hbox{for output layer}
\end{array}
\right.
\nonumber
$$

## Error Sensitivity Computation

### Output Layer
Using the chain rule of differentiation, the error sensitivity of output layer can be computed from (\ref{error_sensitivity_definition}), (\ref{error_function_CNN_definition}) and (\ref{outupt_layer_output_reduced}), as 

$$
\delta_n^{L,k} = \frac{\partial E}{\partial s_n^{L,k}}= \frac{\partial E}{\partial y_n^{L,k}} \frac{\partial y_n^{L,k}}{\partial s_n^{L,k}} \nonumber \\
 = \frac{1}{K \times N_L} \times 2 \times (y_n^{L,k}-d_n^{k}) \times \frac{\partial y_n^{L,k}}{\partial s_n^{L,k}} \nonumber \\
\hspace{-0.2 in} = \frac{2}{K \times N_L} (y_n^{L,k}-d_n^{k}) f_L'(s_n^{L,k}), \hspace{0.1 in} n=1,2,\cdots, N_L
$$

where 

$$
\frac{\partial y_n^{L,k}}{\partial s_n^{L,k} }=f_L'(s_n^{L,k}) \nonumber 
$$

The supporting material is 

$$
y_n^{L,k} = f_L(s_n^{L,k})
\nonumber
$$

If softmax function is adopted, the derivative can be computed as
$$
\delta_n^{L,k} = \frac{\partial E}{\partial s_n^{L,k}} = \frac{1}{K \times N_L} \frac{1}{y_{n_d}^{L,k}} \frac{\partial y_{n_d}^{L,k}}{\partial y_n^L} 
$$

where $$n_d$$ is the label class, and $$\frac{\partial y_{n_d}^{L,k}}{\partial y_n^L}$$ is computed as
$$
\frac{\partial y_{n_d}^{L,k}}{\partial y_n^L} = \nonumber \\
\left \{
\begin{array}{ll}
\frac{\displaystyle \exp( s_{n_d}^{L,k}) \displaystyle \sum_{i=1}^{N_L} \exp( s_i^{L,k}) -  \exp( s_{n_d}^{L,k}) \exp( s_{n_d}^{L,k})}{\left( \displaystyle  \sum_{i=1}^{N_L} \exp( s_i^{L,k} \right)}, & n=n_d \\
\frac{\displaystyle  0 - \exp(s_{n_d}^{L,k}) \exp (s_n^{L,k}) }{\left( \displaystyle  \sum_{i=1}^{N_L} \exp( s_i^{L,k} \right)^2}, & n \neq  n_d
\end{array}
\right. \nonumber \\
= \left \{
\begin{array}{ll}
y_{n_d}^{L,k} (1- y_{n_d}^{L,k} ), & n=n_d \\
 -y_{n_d}^{L,k} y_n^{L,k}, & n \neq  n_d
\end{array}
\right. \nonumber
$$

#### Last Convolution Layer
The error sensitivity of the last convolution layer $$\ell=L-1$$ can be expressed as

$$
\delta_n^{L-1,k} = \frac{\partial E}{\partial s^{L-1,k}_n}= \sum_{m=1}^{N_L}\frac{\partial E}{\partial s^{L,k}_m} 
\frac{\partial s^{L,k}_m}{\partial s^{L-1,k}_n} \nonumber \\
=\sum_{m=1}^{N_L}\frac{\partial E}{\partial s^{L,k}_m} \times \frac{\partial s^{L,k}_m}{\partial y^{L-1,k}_n} \frac{\partial y_n^{L-1,k} }{\partial s_n^{L-1,k}} \nonumber \\
= \sum_{m=1}^{N_L} \delta_m^{L,k} \times w^L_{n,m} \times f_{L-1}'(s_n^{L-1,k}) \nonumber \\
= f_{L-1}'(s_n^{L-1,k}) \times \sum_{m=1}^{N_L} \delta_m^{L,k} w^L_{n,m} 
$$

where 

$$
\frac{\partial s_m^{L,k}}{\partial y_n^{L-1,k}} = w_{n,m}^L  \nonumber  \\
\frac{\partial y_n^{L-1,k}}{\partial s_n^{L-1,k}} = f_{L-1}'(s_n^{L-1,k})
\nonumber 
$$

some supporting equations are as following

$$
s_m^{L,k} = \sum_{n=1}^{N_{L-1}} y_n^{L-1,k} \times w_{n,m}^L + b_m^L \nonumber \\
y_n^{L-1} = f_{L-1}(s_n^{L-1} ) \nonumber 
$$

Note that the feature maps in the last convolution layer $$\ell=L-1$$ have a size of $$1 \times 1$$ pixel, $$\delta_n^{L-1,k}$$ is equivalent to $$\delta_n^{L-1,k}(1,1)$$ (scalar).
Similarly, 
$$
\hspace{-0.2 in} s_n^{L-1} = s_n^{L-1,k}(1,1), \hspace{0.1 in} s_n^{L-1,k} = s_n^{L,k}(1,1), \hspace{0.1 in}y_n^{L-1,k} = y_n^{L-1}(1,1) \nonumber
$$

#### Last Sub-sampling Layer
![](/assets/error_sensitivity_last_subsampling_layer_sche.jpg)
**Fig.1 Schematic of error sensitivity computation for last sub-sampling layer.**

Fig.1 shows the schematic of error sensitivity computation for last sub-sampling layer.

The error sensitivity of the last sub-sampling layer $$\ell=L-2$$ can be expressed as
$$
\delta_n^{L-2,k}(i,j) = \frac{\partial E}{\partial s^{L-2,k}_n(i,j)}  \nonumber \\
= \sum_{m \in U_n^{L-2}} \frac{\partial E}{\partial s_m^{L-1,k}} \times \frac{\partial s_m^{L-1,k}}{\partial s_n^{L-2,k}(i,j)} \nonumber \\
=\sum_{m=m \in U_n^{L-2}} \delta_m^{L-1,k} \times  \frac{\partial s_m^{L-1,k}}{\partial y_n^{L-2}(i,j)}    \frac{\partial y_n^{L-2}(i,j)}{\partial s_n^{L-2,k}(i,j)}  \nonumber \\
\hspace{-0.1 in}=\sum_{m \in U_n^{L-2}} \delta_m^{L-1,k}   w_{n,m}^{L-1}(i,j)   f'_{L-2}[ s_n^{L-2,k}(i,j) ] \nonumber \\
\hspace{-0.1 in}=  f'_{L-2}[ s_n^{L-2,k}(i,j) ]  \sum_{m \in U_n^{L-2}} \delta_m^{L-1,k}   w_{n,m}^{L-1}(i,j)  
$$

where $$n=1,2,\cdots,N_{L-2}$$; $$i=1,2,\cdots,H_{L-2}$$; $$j=1,2,\cdots,W_{L-2}$$, and 

$$
\frac{\partial s_m^{L-1,k}}{\partial s_n^{L-2,k}(i,j)} =  w_{n,m}^{L-1}(i,j) f_{L-2}'[s_n^{L-2,k}(i,j)], \nonumber \\
\hspace{0.1 in} n \in V_m^{L-1} (\hbox{i.e.}, m \in U_n^{L-2})  \nonumber 
$$

Because
$$
\frac{\partial s_m^{L-1,k}}{\partial y_n^{L-2,k}(i,j)} = w_{n,m}^{L-1}(i,j), \hspace{0.1 in} n \in V_m^{L-1}  \nonumber \\
\frac{\partial y_n^{L-2,k}(i,j)}{\partial s_n^{L-2,k}(i,j)} =f_{L-2}'(s_n^{L-2,k}(i,j))
\nonumber 
$$

The supporting materials are
$$
s_m^{L-1,k} = b_m^{L-1}  + \sum_{n  \in V_m^{L-1}} \sum_{i=1}^{h_{L-1}} \sum_{j=1}^{w_{L-1}} \nonumber \\
y_n^{L-2,k}(i,j) \times w_{n,m}^{L-1}(i,j) \nonumber \\
f_{L-2}[ s_n^{L-2,k}(i,j)] = y_n^{L-2,k}(i,j) \nonumber
$$

#### Other Convolution Layer
![](/assets/error_sensitivity_other_conv_layer_sche.jpg)
**Fig.2 Schematic in error sensitivity computation for other convolution layer.**

Fig.2 shows the error sensitivity computation for other convolution layer.

The error sensitivity of other convolution layer ($$\ell=2a+1$$) can be expressed as

$$
\delta_n^{\ell,k}(i,j) = \frac{\partial E}{\partial s_n^{\ell,k}(i,j)} \nonumber \\
=  \sum_{i'=1}^{H_{\ell+1}} \sum_{j'=1}^{W_{\ell+1}} \frac{\partial E}{\partial s_n^{\ell+1,k} (i',j')} \times \frac{\partial s_n^{\ell+1,k}(i',j')}{\partial s_n^{\ell, k}(i,j)}  \nonumber \\
= \frac{\partial E}{\partial s_n^{\ell+1,k} (i_c,j_c)} \times \frac{\partial s_n^{\ell+1,k}(i_c,j_c)}{\partial z_n^{\ell+1, k}(i_c,j_c)}  \times \frac{\partial z_n^{\ell+1,k}(i_c,j_c)}{\partial s_n^{\ell, k}(i,j)}  \nonumber \\
= \delta_n^{\ell+1,k}(i_c,j_c) \times  w_n^{\ell+1} \times \frac{\partial z_n^{\ell+1,k}(i_c,j_c)}{\partial s_n^{\ell, k}(i,j)} 
$$

where $$i_c=\lfloor (i+1)/2 \rfloor $$ and $$j_c=\lfloor (j+1)/2 \rfloor $$, and table.\ref{notation_CNN_training_algorithm} lists the illustrative mapping between $$i_c$$ and $$i$$ index.



Table.2 Illustrative mapping between $$i_c$$ and $$i$$ index.

| $$i$$ | 1 | 2 | 3 | 4 | 5 | $$\cdots$$ |
| -- | -- | -- | -- | -- | --| -- |
| $$i_c$$ | 1 | 1 | 2 | 2 | 3 | $$\cdots$$ |

**Average Pooling: **

(\ref{error_sensitivity_computation_other_convolution_layer}) is computed as

$$
\delta_n^{\ell,k}(i,j) = \frac{\partial E}{\partial s_n^{\ell,k}(i,j)} \nonumber \\
=\delta_n^{\ell+1,k}(i_c,j_c) \times w_n^{\ell+1} \times f_\ell' [s_n^{\ell,k}(i,j)] 
$$

because
$$\frac{\partial z_n^{\ell+1}(i_c,j_c)}{\partial s_n^\ell(i,j)}$$ is computed as

$$
\frac{\partial s_n^{\ell+1,k}(i_c,j_c)}{\partial z_n^{\ell+1,k}(i_c,j_c)} = w_n^{\ell+1} \nonumber \\
\frac{\partial z_n^{\ell+1,k}(i_c,j_c)}{\partial s_n^{\ell,k}(i,j)} = f_\ell'(s_n^{\ell,k}(i,j)) \nonumber
$$

The suporting materials are as follows

$$
\bar{\bar{s}}_n^{\ell+1,k}(i_c,j_c) = \bar{\bar{z}}_n^{\ell+1,k} (i_c,j_c) \times w_n^{\ell+1} + b_n^{\ell+1} \nonumber \\
z_n^{\ell+1}(i_c,j_c) = y_n^\ell(2i_c-1,2j_c-1)+ y_n^\ell(2i_c -1,2 j_c) \nonumber \\
+ y_n^\ell(2i_c, 2j_c-1) + y_n^\ell(2i_c, 2j_c) \nonumber \\
y_n^{\ell,k}(i,j) = f_\ell[ s_n^{\ell,k}(i,j) ] \nonumber
$$ 

** Max Pooling: **

(\ref{error_sensitivity_computation_other_convolution_layer}) is computed as

$$
\delta_n^{\ell,k}(i,j) = \frac{\partial E}{\partial s_n^{\ell,k}(i,j)} \nonumber \\
\hspace{-0.3 in} \left \{ 
\begin{array}{ll}
f_\ell' [s_n^{\ell,k}(i,j)] \delta_n^{\ell+1,k}(i_c,j_c)  w_n^{\ell+1} & (i,j) \hbox{ max pooling condition} \\
0 & \hbox{otherwise}
\end{array}
\right.
$$

where $$i_c=\lfloor (i+1)/2 \rfloor $$ and $$j_c=\lfloor (j+1)/2 \rfloor $$, and table.\ref{notation_CNN_training_algorithm} lists the illustrative mapping between $$i_c$$ and $$i$$ index.

$$(i,j)$$ max pooling condition is 

$$
\hbox{if $i$ is odd and $j$ is odd:} \nonumber \\
(i,j) = \max_{(i',j') \in (i,j), (i+1,j), (i,j+1), (i+1,j+1)} s_n^\ell(i',j') \nonumber \\
\hbox{if $i$ is even and $j$ is odd:} \nonumber \\
(i,j) = \max_{(i',j') \in (i,j), (i+1,j), (i,j+1), (i+1,j+1)} s_n^\ell(i',j') \nonumber \\
\hbox{if $i$ is odd and $j$ is even:} \nonumber \\
(i,j) = \max_{(i',j') \in (i,j), (i+1,j), (i,j+1), (i+1,j+1)} s_n^\ell(i',j') \nonumber \\
\hbox{if $i$ is even and $j$ is even:} \nonumber \\
(i,j) = \max_{(i',j') \in (i,j), (i+1,j), (i,j+1), (i+1,j+1)} s_n^\ell(i',j') \nonumber \\
$$

$$\displaystyle \frac{\partial s_n^{\ell+1}(i',j')}{\partial s_n^\ell(i,j)}$$ in (\ref{error_sensitivity_other_convolution_layer_avg_pooling}) is computed as

$$
\frac{\partial s_n^{\ell+1,k}(i',j')}{\partial s_n^{\ell,k}(i,j)} = \frac{\partial s_n^{\ell+1,k}(i',j')}{\partial z_n^{\ell+1,k}(i',j')} \times \frac{\partial z_n^{\ell+1,k}(i',j')}{\partial s_n^{\ell,k}(i,j)}
\nonumber \\
= w_n^{\ell+1}  \frac{\partial z_n^{\ell+1,k}(i',j')}{\partial s_n^{\ell,k}(i,j)} \nonumber 
$$

Because, 
$$
\frac{\partial z_n^{\ell+1,k}(i',j')}{\partial s_n^{\ell,k}(i,j)} 
\nonumber \\
\hspace{-0.3 in} = \left\{ 
\begin{array}{ll}
w_n^{\ell+1} f_\ell(s_n^{\ell,k}(i,j)), & (i',j')=(\lfloor \frac{i+1}{2} \rfloor , \lfloor \frac{j+1}{2} \rfloor)  \\
0, & \hbox{otherwise}
\end{array}
\right. \nonumber
$$

and

$$
\frac{\partial s_n^{\ell+1,k}(i',j')}{\partial z_n^{\ell+1,k}(i',j')} = w_n^{\ell+1} \nonumber \\
\frac{\partial z^{\ell+1,k}_n(i',j')}{\partial s_n^{\ell,k}(i,j)} = f_\ell'(s^{\ell,k}_n(i,j)), \hspace{0.1 in}
(i',j')=(\lfloor \frac{i+1}{2} \rfloor , \lfloor \frac{j+1}{2} \rfloor)  \nonumber 
$$

some suporting materials are as follows
$$
\bar{\bar{s}}_n^{\ell+1,k}(i',j') = \bar{\bar{z}}_n^{\ell+1,k} (i',j') \times w_m^{\ell+1} + b_n^{\ell+1} \nonumber 
$$

and

$$
 \frac{\partial z^{\ell+1,k}_n(i',j')}{\partial s_n^{\ell,k}(i,j)} = \nonumber \\
\left\{ 
\begin{array}{ll}
f_\ell'(s^{\ell,k}_n(i,j)),& (i,j) \hbox{ condition for max pooing} \\
0, & \hbox{otherwise}
\end{array}
\right. \nonumber 
$$

where the $(i,j)$ condition for max pooing is
$$
(i,j) =\displaystyle \max_{(i,j) \in (2i'-1,2j'-1), (2i'-1,2j'), (2i',2j'-1), (2i',2j')} s_n^\ell(i,j)
\nonumber
$$

The supporting material are appended as

$$
z_n^{\ell+1}(i,j) = \max \{ y_n^\ell(2i-1,2j-1), y_n^\ell(2i -1,2 j), \nonumber \\
y_n^\ell(2i, 2j -1) , y_n^\ell(2i, 2j) \} \nonumber
$$


#### Other Sub-sampling Layer
![](/assets/error_sensitivity_computation_other_subsamp_layer.jpg)
**Fig.3 Schematic in error sensitivity computation for other sub-sampling layer.**

Fig.3 shows the error sensitivity computation for other sub-sampling layer.
The error sensitivity of other sub-sampling layer ($$\ell=2a$$) can be expressed as

$$
\delta_n^{\ell,k}(i,j) = \frac{\partial E}{\partial s_n^{\ell,k}(i,j)} \nonumber \\
= \sum_{m \in U_n^\ell} \sum_{i'=1}^{H_{\ell+1}} \sum_{j'=1}^{W_{\ell+1}} \frac{\partial E}{\partial s_m^{\ell+1,k}(i',j')} \times \frac{\partial s_m^{\ell+1,k}(i',j')}{\partial s_n^{\ell,k}(i,j)}
\nonumber \\
=f_\ell' [s_n^{\ell,k}(i,j)] \times \sum_{m \in U_n^\ell} \sum_{i'=1}^{H_{\ell+1} } \sum_{j'=1}^{W_{\ell+1}} \nonumber \\
 \delta_m^{\ell+1,k}(i',j') \times w_{n,m}^{\ell+1}(i-i'+1,j - j'+1)
$$

Noting 

$$
w_{n,m}^{\ell+1}(i-i'+1,j-j'+1) =  \nonumber \\
\left \{ 
\begin{array}{ll}
\neq 0, & 1 \leq i-i'+1 \leq h_{\ell+1} \\
        &  1 \leq  j-j'+1 \leq w_{\ell+1}  \\
 = 0, & \hbox{otherwise} 
\end{array}
\right. \nonumber
$$

where
$$
\frac{\partial s_m^{\ell+1,k}(i',j')}{\partial s_n^{\ell,k}(i,j)} =\frac{\partial s_m^{\ell+1,k}(i',j')}{\partial y_n^{\ell,k} (i,j)} \frac{\partial y_n^{\ell,k}(i,j)}{\partial s_n^{\ell,k}(i,j)} \nonumber \\
\hspace{-0.4 in}\left \{
\begin{array}{ll}
w_{n,m}^{\ell+1}(i-i'+1,j-j'+1) ,&n \in V_m^{\ell+1} (\hbox{i.e.}, m \in U_n^{\ell} ) \\
\times f_\ell'[s_n^{\ell,k}(i,j)] & 1\leq i-i'+1 \leq h_{\ell+1}, \\
 & 1\leq j-j'+1 \leq w_{\ell+1}, \\
0, & \hbox{otherwise} 
\end{array}
\right. \nonumber
$$

because 

$$
\frac{\partial s_m^{\ell+1,k}(i',j')}{\partial y_n^{\ell,k}(i,j)} = w_{n,m}^{\ell+1}(i-i'+1,j-j'+1) \nonumber \\
\frac{\partial y_n^{\ell,k}(i,j)}{\partial s_n^{\ell,k}(i,j)} = f_\ell'(s_n^{\ell,k}(i,j)) \nonumber
$$

The supporting material is
$$
s_m^{\ell+1,k}(i',j') =b_m^{\ell+1}  + \sum_{n \in V_m^{\ell+1}} \sum_{i=i'}^{h_{\ell+1} +i'-1} \sum_{j=j'}^{w_{\ell+1} + j'-1} \nonumber \\
f_{\ell}[ s_n^{\ell,k}(i,j)] \times w_{n,m}^{\ell+1,k}(i-i'+1,j-j'+1) \nonumber \\
y_n^{\ell,k}(i,j)=f_{\ell}[ s_n^{\ell,k}(i,j)] 
\nonumber
$$

Noting that 

$$
w_{n,m}^{\ell+1}(i-i'+1,j-j'+1) = 0, \hspace{0.1 in} \nonumber \\
\hbox{if } i-i'+1 <=0 \hbox{, or }  i-i'+1 >h_{\ell+1}, \nonumber \\
\hbox{ or } j-j'+1 <=0 \hbox{, or } j-j'+1 > w_{\ell+1}
\nonumber \\
w_{n,m}^{\ell+1}(i-i'+1,j-j'+1) \neq 0, \nonumber \\
1 \leq i-i'+1 \leq h_{\ell+1}, \hspace{0.1 in} 1 \leq  j-j'+1 \leq w_{\ell+1} \nonumber
$$

Combining these relations, the effective index $$i'$$ and $$j'$$ can be derived as 

$$
1 \leq i-i'+1 \leq h_{\ell+1} \to i-h_{\ell+1}+1 \leq i' \leq i \nonumber \\
1 \leq j - j'+1 \leq w_{\ell+1} \to j-w_{\ell+1}+1 \leq j' \leq j \nonumber \\
1 \leq i' \leq H_{\ell+1}, \hspace{0.1 in} 1 \leq j' \leq W_{\ell+1} \nonumber \\
\max(1, i- h_{\ell+1}+1) \leq i' \leq \min(H_{\ell+1},i) \nonumber \\
\max(1,j-w_{\ell+1}+1 \leq j' \leq \min(W_{\ell+1},j) \nonumber
$$

Therefore, (\ref{error_sensitivity_computation_other_subsampling_layer}) can be  re expresed as

$$
\delta_n^{\ell,k}(i,j) =f_\ell' [s_n^{\ell,k}(i,j)] \nonumber \\
\times \sum_{m \in U_n^\ell} \sum_{i'=\max(1, i- h_{\ell+1}+1)}^{\min(H_{\ell+1},i)} \sum_{j'=\max(1,j-w_{\ell+1}+1)}^{ \min(W_{\ell+1},j)} \nonumber  \\
\delta_m^{\ell+1,k}(i',j') \times w_{n,m}^{\ell+1}(i-i'+1,j - j'+1) \nonumber
$$

### Error Gradient Computation

#### Output Layer, $$\ell=L$$

$$
\delta_{w_{m,n}}^L  =\frac{\partial E}{\partial w_{m,n}^L} = \sum_{k=1}^K  \frac{\partial E}{\partial s_n^{L,k}} \times \frac{\partial s_n^{L,k}}{\partial w_{m,n}^L } \nonumber \\
= \sum_{k=1}^{K} \delta_n^{L,k} y_m^{L-1,k}
\nonumber
$$

where $$m=1,2,\cdots,N_{L-1}$$ and $$n=1,2,\cdots,N_L$$.
$$\frac{\partial s_n^{L,k}}{\partial w_{m,n}^L}$$ is computed as

$$
\frac{\partial s_n^{L,k}}{\partial w_{m,n}^L} =y_m^{L-1,k} \nonumber
$$

The gradient of bias is computed as
$$
\delta_{b_n}^L =\frac{\partial E}{\partial b_n^L}= \sum_{k=1}^K  \frac{\partial E}{\partial s_n^{L,k}} \frac{\partial s_n^{L,k}}{\partial b_n^L} = \sum_{k=1}^K \delta_{n}^{L,k}
\nonumber
$$

where $$n=1,2,\cdots,N_L$$.

#### Last Convolution Layer, $$\ell=L-1$$
The gradient computation of last convolution layer can be derived as

$$
\hspace{-0.2 in} \delta_{w_{m,n}}^{L-1}  =\frac{\partial E}{\partial w_{m,n}^{L-1} (i,j)} = \sum_{k=1}^K  \frac{\partial E}{\partial s_n^{L-1,k}} \times \frac{\partial s_n^{L-1,k}}{\partial w^{L-1}_{m,n}(i,j)} \nonumber \\
\hspace{-0.2 in} = \sum_{k=1}^K  \delta_n^{L-1,k} y_m^{L-2,k}(i,j) \nonumber 
$$

where $$n=1,2,\cdots,N_{L-1}$$. $$\frac{\partial s_n^{L-1,k}}{\partial w^{L-1}_{m,n}(i,j)}$$ is computed as

$$
\frac{\partial s_n^{L-1,k}}{\partial w^{L-1}_{m,n}(i,j)} =\frac{\partial s_n^{L-1,k}(1,1)}{\partial w^{L-1}_{m,n}(i,j)} =    \nonumber \\
\hspace{-0.2 in} \left \{
\begin{array}{ll}
y_m^{L-2,k}(i-1+1,j-1+1), & m \in V_n^{L-1} (n \in U_m^{L-2}) \\
0, & \hbox{otherwise}
\end{array}
\right. \nonumber
$$

The supporting material equation is as follows,

$$
s_n^{L-1,k}(1,1) = b_n^{L-1} + \sum_{m \in V_n^{L-1}} \sum_{i=1}^{h_{L-1}} \sum_{j=1}^{w_{L-1}} \nonumber \\
y_m^{L-2,k}(i-1+1, j - 1 + 1) \times w_{m,n}^{L-1}(i,j) \nonumber
$$

The gradient of bias is computed as

$$
\delta_{b_n}^{L-1} = \frac{\partial E}{\partial b_n^{L-1}} = \sum_{k=1}^K \frac{\partial E}{\partial s_n^{L-1,k}} \frac{\partial s_n^{L-1,k}}{\partial b_n^{L-1}}=  \sum_{k=1}^K \delta_{n}^{L-1,k}
\nonumber
$$

where $$n=1,2,\cdots,N_{L-1}$$.


#### Sub-Sampling Layer, $$\ell=2a$$

\begin{figure}[h]
\vskip 3 cm
\hskip 0 cm
\special{wmf:schematic_for_gradient_w_sub_sampling_layer.jpg x=9 cm y=3cm}
\caption{Schematic of gradient computation for sub-sampling layer.}
\label{schematic_for_gradient_w_sub_sampling_layer}
\end{figure}
Fig.\ref{schematic_for_gradient_w_sub_sampling_layer} shows the schematic for gradient computation of sub-sampling layer. The gradient computation of sub-sampling layer can be derived as

$$
\delta_{w_n}^\ell = \frac{\partial E}{\partial w_n^\ell} =  
\sum_{k=1}^K \sum_{i=1}^{H_\ell} \sum_{j=1}^{W_\ell} \frac{\partial E}{\partial s_n^{\ell,k}(i,j)} \frac{\partial s_n^{\ell,k}(i,j)}{\partial w_n^\ell} \nonumber \\
=\sum_{k=1}^K \sum_{i=1}^{H_\ell} \sum_{j=1}^{W_\ell} \delta_n^{\ell,k}(i,j) z_n^{\ell,k}(i,j) 
\nonumber
$$

where $$n=1,2,\cdots,N_\ell$$.

$$
\delta_{b_n}^\ell = \frac{\partial E}{\partial b_n^\ell} = \sum_{k=1}^K \sum_{i=1}^{H_\ell} \sum_{j=1}^{W_\ell}  \delta_n^{\ell,k}(i,j)
\nonumber
$$

where $$n=1,2,\cdots,N_\ell$$.

#### Other Convolution Layer, $$\ell=2a+1$$

\begin{figure}[h]
\vskip 5 cm
\hskip 0 cm
\special{wmf:gradient_computation_other_convolution_layer.jpg x=9 cm y=5 cm}
\caption{Schematic of gradient computation for other convolution layer.}
\label{gradient_computation_other_convolution_layer}
\end{figure}
Fig.\ref{gradient_computation_other_convolution_layer} shows the schematic of gradient computation for other convolution layer.

The gradient computation can be derived as
$$
\hspace{-0.3 in} \delta_{w_{m,n}}^\ell = \frac{\partial E}{\partial w_{m,n}^\ell (i,j)} = \sum_{k=1}^K  \sum_{i'=1}^{H_\ell} \sum_{j'=1}^{W_\ell} \frac{\partial E}{\partial s_n^{\ell,k}(i',j')} \times \frac{\partial s_n^{\ell,k}(i',j')}{\partial w^{\ell}_{m,n}(i,j)} \nonumber \\
\hspace{-0.3 in}=\sum_{k=1}^K  \sum_{i'=1}^{H_\ell} \sum_{j'=1}^{W_\ell} \delta_n^{\ell,k}(i',j')  y_m^{\ell-1,k}(i-1+i',j-1+j') 
\label{dE_dw_convolution_layer}
$$

where $$n=1,2,\cdots,N_\ell$$. Note that,

$$
1 \leq i \leq h_{\ell}, \hspace{0.1 in} 1 \leq j \leq w_\ell, \nonumber \\
1 \leq i' \leq H_\ell, \hspace{0.1 in}  1 \leq j' \leq W_\ell, \nonumber \\
\to 1 \leq i-1 + i' \leq h_\ell + H_\ell -1 = H_{\ell-1} \nonumber \\
1 \leq j-1 + j' \leq w_\ell + W_\ell -1 = W_{\ell-1} \nonumber 
$$

$$\frac{\partial s_n^{\ell,k}(i',j')}{\partial w^{\ell}_{m,n}(i,j)}$$ is computed as

$$
\hspace{-0.4 in} \frac{\partial s_n^{\ell,k}(i',j')}{\partial w^\ell_{m,n}(i,j)} =  \left \{
\begin{array}{ll}
y_m^{\ell-1,k}(i-1+i',j-1+j'), & m \in V_n^\ell \\
0, & \hbox{otherwise}
\end{array}
\right. \nonumber
$$

The supporting material is as follows,
$$
s_n^{\ell,k}(i',j') = \sum_{m \in V_n^\ell} \sum_{i=1}^{h_{\ell}} \sum_{j=1}^{w_\ell} \nonumber \\
y_m^{\ell-1,k}(i-1+i', j - 1 + j') \times w_{m,n}^{\ell}(i,j)  + b_n^\ell \nonumber
$$

$$
\delta_{b_n}^\ell = \frac{\partial E}{\partial b_n^\ell} = \sum_{k=1}^K \sum_{i=1}^{H_\ell} \sum_{j=1}^{W_\ell}  
\frac{\partial E}{\partial s_n^{\ell,k}(i,j)} \frac{\partial s_n^{\ell,k}(i,j)}{\partial b_n^\ell}\nonumber \\
=  \sum_{k=1}^K \sum_{i=1}^{H_\ell} \sum_{j=1}^{W_\ell}   \delta_n^{\ell,k}(i,j)
\nonumber
$$

where $$n=1,2,\cdots,N_\ell$$.

#### CNN Training Algorithm

Once the error gradient is derived, numerous optimization algorithm for minimizing $E$ can be applied to train the network, which are gradient descent (GD), gradient descent with momentum and variable learning rate (GDMV), resilient back-propagation (RPROP), conjugate gradient (CG) and Levenberg-Marquardt (LM).

GD, GDMV and RPROP algorithms are first-order optimization methods.

CG algorithm can be considered as an intermediate between first- and second-order methods, whereas as the LM algorithm is trust-region method that uses the Gauss-Newton approximation of the Hessian matrix.

Table summarize the main characteristics of these algorithms.

\caption{Notation for CNN training algorithm}
\label{notation_CNN_training_algorithm}

| algorithm | description |
| -- | -- |
| GD | weights are updated along the negative gradient $$\Delta \bar{\bar{w}} (t) = - \alpha \nabla E(t)$$, where $$\alpha>0$$ is a scalar learning rate. |
| GDMV | weight update is a linear combination of gradient and previous weight update, $$\Delta \bar{\bar{w}}(t) = \lambda \Delta \bar{\bar{w}}(t-1) - (1-\lambda) \alpha(t) \nabla E(t)$$, where $$0 < \lambda < 1$$ is momentum parameter and $$\alpha(t)$$ is adaptive scalar learning rate. |
| RPROP | weight update depends on the sign of gradient, $$\displaystyle \Delta \bar{\bar{w}}_i(t) = -{\rm sign}\{ \frac{\partial E}{\partial w_i}(t) \} \times \Delta_i(t)$$, where $$\Delta_i(t)$$ is adaptive step specific to weight $$\bar{\bar{w}}_i$$, as $$
\Delta_i(t) = \left \{
\begin{array}{ll}
\eta_{\rm inc} \Delta_i (t-1), & \frac{\partial E}{\partial \bar{\bar{w}}_i}(t) \frac{\partial E}{\partial  \bar{\bar{w}}_i (t-1)} >0 \\
\eta_{\rm dec} \Delta_i (t-1), & \frac{\partial E}{\partial \bar{\bar{w}}_i(t)} \frac{\partial E}{\partial \bar{\bar{w}}_i(t-1) } <0 \\
\Delta_i( t-1), & {\rm otherwise} 
\end{array}
\right. \nonumber
$$ where $\eta_{\rm inc}>1$, $0<\eta_{\rm dec}<1$ are scalar terms. |
| CG | weights updated along directions mutually conjugated with respect to Hessian matrix, $$\Delta \bar{\bar{w}}(t) = \alpha(t) \bar{\bar{s}}(t)$$, where $$\bar{\bar{s}}$$ is the search direction, defined as $$
\bar{\bar{s}}(t) = \left \{ 
\begin{array}{ll}
- \nabla E(t), & t = 1 (\hbox{mod }P) \\
- \nabla E(t) + \beta(t) \bar{\bar{s}}(t-1) & {\rm otherwise} 
\end{array}
\right. \nonumber
$$ where $$\alpha(t)$$ is learning step derived with line search. $$\beta(t)$$ is updated according to the Polak-Ribiere formula, $$\displaystyle \beta(t) = \frac{| \nabla E(t)T - \nabla E(t-1) |^t \nabla E(t)}{\Vert \nabla E(t-1) \Vert^2}$$ |
| LM | 2nd-order Tylor expansion and Gauss-Newton approximation of Hessian matrix, $$\Delta \bar{\bar{w}}(t) = -[ \bar{\bar{J}}^t \cdot \bar{\bar{J}} + \mu \bar{\bar{I}}]^{-1} \cdot \Delta E$$, where $$\bar{\bar{J}}$$ is Jacobian matrix defined as $$
J_{(q-1)K+k,i}= \frac{\partial e^k_q}{\partial w_i}, \nonumber \\
q=1,\cdots, N_L \hspace{0.1 in} k=1,\cdots,K \hspace{0.1 in} i=1,\cdots, P
\nonumber
$$ $$\nabla E$$ is computed through the Jacobian matrix $$\bar{\bar{J}}$$, and $$\mu$$ is an adaptive parameter controlling the size of the trust region. |

Computation of the Jacobian matrix is similar to computation of the gradient $$\nabla E$$.

However, the definition of error sensitivities is modified.

For the Jacobian matrix, error sensitivities are defined for each network error $$e^k_q$$, where $$q=1,\cdots,N_L$$ (instead of the overall error function $$E$$).

The details of RPROP algorithm can found in \cite{M.Riedmiller_1993}.
