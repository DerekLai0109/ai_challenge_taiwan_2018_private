#Forward problem using an 6-layer illustrative example

## First Layer
Feature map $$n$$ of the first layer (convolution layer)  is calculated as
$$
\bar{\bar{y}}_n^1 = f_1 \left( \sum_{m \in V_n^1} \bar{\bar{y}}^{0}_m \otimes \bar{\bar{w}}_{m,n}^1 + b_n^1 \right)
\nonumber \\
=f_1 \left( \sum_{m \in V_n^1} \bar{\bar{C}}_{m,n}^1 + b_n^1 \right) = f_1 ( \bar{\bar{s}}_n^1 )
\nonumber 
$$

where 

$$
\bar{\bar{C}}_{m,n}^1= \bar{\bar{y}}_m^{0} \otimes  \hspace{0.1 in}  \bar{\bar{w}}_{m,n}^1
\nonumber
$$

where $$\otimes$$ is the 2D convolution operator. 

The computation of convolution operation can be explicitly expressed as
$$
\bar{\bar{C}}^1_{m,n}(i,j) = \sum_{i'=1}^{ h_1} \hspace{0.1 in} \sum_{j'=1}^{w_1} \nonumber \\
\bar{\bar{y}}_m^{0}(i'-1+i, j'-1+j) \times \bar{\bar{w}}^1_{m,n}(i',j')
\nonumber \\
1 \leq i \leq H_{0} - h_1+1, \hspace{0.1 in} 1 \leq j \leq W_{0} - w_1+1 \nonumber 
$$

Or 

$$
\bar{\bar{C}}^1_{m,n}(i,j)=\sum_{i'=i}^{h_1 + i -1} \sum_{j'=j}^{w_1+j-1} \nonumber \\
\bar{\bar{y}}^{0}_m(i',j') \times \bar{\bar{w}}_{m,n}^1 (i'-i+1,j'-j+1)
\nonumber \\
1 \leq i \leq H_{1}, \hspace{0.1 in} 1 \leq j \leq W_{1} \nonumber \\
1 \leq i' - i +1 \leq h_1, \hspace{0.1 in} 1 \leq j' - j +1 \leq w_1 \nonumber
$$

## Second Layer
The feature map $$n$$ of sub-sampling layer $$2$$ is calculated as

$$
\bar{\bar{y}}_n^2 = f_2( \bar{\bar{z}}_n^{2} \times w^2_n + b_n^2 ) =f_2(\bar{\bar{s}}_n^2)
\nonumber
$$

where  $$w^2_n$$ and $$b_n^2$$ are the weight and bias term, respectively. The size of feature map $$\bar{\bar{y}}_n^2$$ in sub-sampling layer $$2$$ is

$$
H_2=  H_{1}/2, \hspace{0.1 in} W_2 = W_{1}/2 
\nonumber
$$

### Average Pooling
$$\bar{\bar{z}}_n^{2}$$ is a matrix, and its element is obtained by summing the four pixels in each block, 
$$
z_n^{2}(i,j)= y_n^{1} (2i-1,2j-1) + y_n^{1} (2i-1, 2j) \nonumber \\
+  y_n^{1} (2i, 2j-1) +  y_n^{1} (2i, 2j) 
\nonumber
$$

### Max Pooling
$$\bar{\bar{z}}_n^{2}$$ is a matrix, and its element is assigned with the maximal entry among the four pixels in each block, 

$$
z_n^{2}(i,j)= \max \{ y_n^{1} (2i-1,2j-1), y_n^{1} (2i-1, 2j), \nonumber \\
y_n^{1} (2i, 2j-1),  y_n^{1} (2i, 2j) \}
\nonumber
$$


## Third Layer
Feature map $$n$$ of the first layer (convolution layer)  is calculated as
$$
\bar{\bar{y}}_n^3 = f_3 \left( \sum_{m \in V_n^3} \bar{\bar{y}}^{2}_m \otimes \bar{\bar{w}}_{m,n}^3 + b_n^3 \right)
\nonumber \\
=f_3 \left( \sum_{m \in V_n^3} \bar{\bar{C}}_{m,n}^3 + b_n^3 \right) = f_3 ( \bar{\bar{s}}_n^3 )
\nonumber 
$$

where 

$$
\bar{\bar{C}}_{m,n}^3= \bar{\bar{y}}_m^{2} \otimes  \bar{\bar{w}}_{m,n}^3
\nonumber
$$

where $$\otimes$$ is the 2D convolution operator. 

The computation of convolution operation can be explicitly expressed as

$$
\bar{\bar{C}}^3_{m,n}(i,j) = \sum_{i'=1}^{ h_3} \hspace{0.1 in} \sum_{j'=1}^{w_3} \nonumber \\
\bar{\bar{y}}_m^{2}(i'-1+i, j'-1+j) \times \bar{\bar{w}}^3_{m,n}(i',j')
\nonumber \\
1 \leq i \leq H_{2} - h_3+1, \hspace{0.1 in} 1 \leq j \leq W_{2} - w_3+1 \nonumber 
$$

Or 

$$
\bar{\bar{C}}^3_{m,n}(i,j)=\sum_{i'=i}^{h_3 + i -1} \sum_{j'=j}^{w_3+j-1} \nonumber \\
\bar{\bar{y}}^{2}_m(i',j') \times \bar{\bar{w}}_{m,n}^3 (i'-i+1,j'-j+1)
\nonumber \\
1 \leq i \leq H_{3}, \hspace{0.1 in} 1 \leq j \leq W_{3} \nonumber \\
1 \leq i' - i +1 \leq h_3, \hspace{0.1 in} 1 \leq j' - j +1 \leq w_3 \nonumber
$$

## Fourth Layer
The feature map $$n$$ of sub-sampling layer $$4$$ is calculated as

$$
\bar{\bar{y}}_n^4 = f_4( \bar{\bar{z}}_n^{4} \times w^4_n + b_n^4 ) =f_4(\bar{\bar{s}}_n^4)
\nonumber
$$

where  $$w^4_n$$ and $$b_n^4$$ are the weight and bias term, respectively. The size of feature map $$\bar{\bar{y}}_n^4$$ in sub-sampling layer $$4$$ is

$$
H_4=  H_{3}/2, \hspace{0.1 in} W_4 = W_{3}/2 
\nonumber
$$

### Average Pooling
$$\bar{\bar{z}}_n^{4}$$ is a matrix, and its element is obtained by summing the four pixels in each block, 

$$
z_n^{4}(i,j)= y_n^{3} (2i-1,2j-1) + y_n^{3} (2i-1, 2j) \nonumber \\
+  y_n^{3} (2i, 2j-1) +  y_n^{3} (2i, 2j) 
\nonumber
$$

### Max Pooling
$$\bar{\bar{z}}_n^{4}$$ is a matrix, and its element is assigned with the maximal entry among the four pixels in each block, 

$$
z_n^{4}(i,j)= \max \{ y_n^{3} (2i-1,2j-1), y_n^{3} (2i-1, 2j), \nonumber \\
y_n^{3} (2i, 2j-1),  y_n^{3} (2i, 2j) \}
\nonumber
$$

## Fifth Layer
Feature map $$n$$ of the first layer (convolution layer) is calculated as

$$
\bar{\bar{y}}_n^5 = f_5 \left( \sum_{m \in V_n^5} \bar{\bar{y}}^{4}_m \otimes \bar{\bar{w}}_{m,n}^5 + b_n^5 \right)
\nonumber \\
=f_5 \left( \sum_{m \in V_n^5} \bar{\bar{C}}_{m,n}^5 + b_n^5 \right) = f_5( \bar{\bar{s}}_n^5 )
\nonumber 
$$

where 

$$
\bar{\bar{C}}_{m,n}^5= \bar{\bar{y}}_m^{4} \otimes \bar{\bar{w}}_{m,n}^5
\nonumber
$$

where $$\otimes$$ is the 2D convolution operator. 

The computation of convolution operation can be explicitly expressed as

$$
\bar{\bar{C}}^5_{m,n}(i=1,j=1) = \sum_{i'=1}^{ h_5} \hspace{0.1 in} \sum_{j'=1}^{w_5} \nonumber \\
\bar{\bar{y}}_m^{4}(i', j') \times \bar{\bar{w}}^5_{m,n}(i',j')
\nonumber 
$$

Or 

$$
\bar{\bar{C}}^5_{m,n}(i=1,j=1)=\sum_{i'=1}^{h_3} \sum_{j'=1}^{w_3} \nonumber \\
\bar{\bar{y}}^{4}_m(i',j') \times \bar{\bar{w}}_{m,n}^5 (i',j')
\nonumber \\
1 \leq i \leq H_{5}=1, \hspace{0.1 in} 1 \leq j \leq W_{5}=1 \nonumber \\
1 \leq i'  \leq h_5, \hspace{0.1 in} 1 \leq j' \leq w_5 \nonumber
$$

## Output Layer Fully Connected Layer
The output of sigmoidal neuron $$n$$ is calculated as

$$
y_n^L = f_L \left( \sum_{m=1}^{N_{L-1}} y_m^{L-1} w_{m,n}^L + b_n^L \right) \nonumber \\
= f_L ( s_n^L )    \nonumber
$$

where $$N^L$$ is the number of output sigmoidal neuron, $$w_{m,n}^L$$ denotes the weight from feature map $$m$$ of the last convolutional layer, to neuron $$n$$ of the output layer. $$b_n^L$$ be the bias term associated with neuron of layer $$L$$. The outputs of all sigmoidal neurons form the network outputs:

$$
\bar{y}= \left[ y_1^L, y_2^L, \cdots, y_{N_L}^L \right]
\nonumber
$$

### SoftMax function (usually adopted for output layer)
Softmax function model generalizes logistic regression to classification problems where the class label can take on multiple values.

$$
y_n^{\ell}= \frac{\exp( s_n^\ell) }{\displaystyle \sum_{n_\ell=1}^{N_\ell} \exp( s_{n_\ell}^\ell)}
\nonumber
$$

All output values in the range (0,1) and sum up to 1 for softmax function, and this property make it suitable for a probablistic interpretation. If we have $$N_L$$ output classes, the softmax can be interpreted as

$$
y_n^{L}= \hbox{P}(x \in \hbox{n-th class}) \nonumber 
$$
