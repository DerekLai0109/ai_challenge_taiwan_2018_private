#An Elaboration of Convolutional Neural Network
2018/3/22 M.M. Chiou

###Abstract 
The computation of Convolution Neural Network (CNN) has been elaborated.
One illustrative example of CNN with 6 layers is present.

###Introduction
 Conovlutional neural network (CNN) are desinged to process 2D image.
CNN, originally proposed by LeCun \cite{Y.L_1998}, is a neural network model with three key architectural ideas: local receptive fields, weight sharing, and sub-sampling in the spatial domain.
The network is designed for the recognition of two-dimensional visual patterns.
CNN has two strengths. First, feature extraction and classification are integrated into one structure which are adaptive.
Second, it is relatively invariant to gemetric, local distoritons in the image.
CNN has been used for applications including hand-written digit recognition, face detection, and face recognition.

The report is structured as follows.
Section I describes architectural aspects of the convolutional neural networks.
The forward and backward problems (computation) are present in section II and III, respectively.
The illustrative example is given in section IV.

###Heuristic View on CNN\begin{figure}[h]
![heuristic_on_CNN](/assets/heuristic_on_CNN.jpg)
*Fig.1 Heuristic view on CNN.*
Fig. 1 <\ref{heuristic_on_CNN}> shows the heuristic view of CNN, where $$\bar{\bar{x}}$$ is the input image. $$\bar{y}^L$$ is the output vector of CNN. $$\bar{d}$$ is the desired output vector. $$\bar{e} = \bar{y}^L - \bar{d}$$  is the error vector. $$w_n^\ell$$ ($$\bar{\bar{w}}_n^{\ell}$$) and $$b_n^\ell$$ are the weights and bias, respectively, of $$n$$th neuron in $$\ell$$th layer. $$L$$ is the number of layer. $$N_\ell$$ is the number of neuron in $$\ell$$ layer. These parameters needs to be initialized. The **UpdateNet** block updates weights and biases $$\bar{\bar{w}}_n^{\ell, (t)}, b_n^{\ell, (t)}, w_n^{\ell, (t)}$$ into the new ones, i.e., $$\bar{\bar{w}}_n^{\ell, (t+1)}, b_n^{\ell, (t+1)}, w_n^{\ell, (t+1)}$$.

###CNN Network Model Overview
<\begin{figure}[h]
<\vskip 6.5 cm
<\hskip -0.2 cm
<\special{wmf:Layers_architecture_in_CNN.jpg x=9.3 cm y=6.5 cm}
<\caption{Layers architecture in a CNN. Note that the output layer is fully connected. $\bar{\bar{y}}^\ell_n, \ell=1, 2, \cdots, L-2$, is the $n$th  feature map (2-D output) in $\ell$ layer.}
<\label{Layers_architecture_in_CNN}
<\end{figure}
![Layers_architecture_in_CNN](/assets/Layers_architecture_in_CNN.jpg)
*Fig.2 Layers architecture in a CNN. Note that the output layer is fully connected. $$\bar{\bar{y}}^\ell_n, \ell=1, 2, \cdots, L-2$$, is the $$n$$th  feature map (2-D output) in $$\ell$$ layer.*

Fig. 2 <\ref{Layers_architecture_in_CNN}> shows the layers architecture in a CNN, which consists of three main types of layers: (i) convolution layers, (ii) sub-sampling layers, and (iii) an output layer. $$\bar{\bar{x}} = \bar{\bar{y}}^0$$ is the input image. $$\bar{\bar{y}}^\ell_n, \ell=1, 2, \cdots, L-2$$, is the $n$th feature map in $$\ell$$ layer.$$y_n^\ell$$, $$\ell=L-1, L$$ are the $$n$$th output of the $$\ell$$ layer. Noting that the super script $$\ell$$ denotes the order of layer, instead of power of that parameter. Network layers are arranged in a feed-forward structure: each convolution layer is followed by a sub-sampling layer, and each subsampling layer is followed by convolution layer. The last convolution layer is followed by the output layer. The convolution and sub-sampling layers are considered as 2D layers, while the output layer is considered as a 1D layer. The neurons are arranged in a 2D array in 2D layers, and they are also called feature map.

In a convolutional layer, each output feature map $$\bar{\bar{y}}_n^\ell$$ is connected to one or more feature maps of the preceding layer, $$\bar{\bar{y}}_{m}^{\ell-1}$$. A connection is associated with a convolution mask, which is a 2D matrix of adjustable entries called weights. The convolution outputs, $$\bar{\bar{C}}_{m,n}^\ell$$ is computed as the convolution between its 2D intputs $$\bar{\bar{y}}_m^{\ell-1}$$ and its convolution masks $$\bar{\bar{w}}_{m,n}^\ell$$. The convolution outputs are summed together and then added with an adjustable scalar, known as bias term. An activation function is applied on the result to obtain the plane's output. The output of convolution layer is generally a 2D matrix (called a feature map); this name arises because each convolution output indicates the presence of a visual feature at a given pixel location. There will be multiple feature maps produced after the convolution layer. Each feature map is connected to one feature map in the next sub-sampling layer.

A sub-sampling layer has the same number of feature maps as the preceding convolution layer. A subsampling layer divides its 2D input into non-overlapping blocks of size 2x2 pixels (it can also be 3x3, 4x4, etc.). For each block, the sum of four pixels is calculated; this sum is multiplied by an adjustable weight before being added to a bias term. The result is passed through an activation function to produce an output for the 2x2 block. Each subsampling layer reduces its input size (by half for the case of 2x2), along each dimension. A feature map in a sub-sampling layer is connected to one or more planes in the next convolution layer.

In the last convolution layer, the convolution masks have exactly the same size as its input feature maps. Each plane in the last convolution layer will produce one scalar output. The outputs from all planes in this layer are connected to the output layer.

The output layer can be constructed from sigmoidal neurons or radial-basis function (RBF) neurons. The outputs of output layer are considered as the network outputs. In applications of visual pattern classification, these outputs indicate the category of the input image.


<\begin{figure}[h]
<\vskip 6.5 cm
<\hskip -0.2 cm
<\special{wmf:other_kind_Layers_architecture_in_CNN.jpg x=9.3 cm y=6.5 cm}
<\caption{Other Layers architecture in a CNN. The difference between fig.\ref{other_kind_Layers_architecture_in_CNN} and fig.\ref{Layers_architecture_in_CNN} is the last convolution layer is replaced by the spreading out layer. }
<\label{other_kind_Layers_architecture_in_CNN}
<\end{figure}

![other_kind_Layers_architecture_in_CNN](/assets/other_kind_Layers_architecture_in_CNN.jpg)
*Fig.3 Other Layers architecture in a CNN. The difference between fig.3* <\ref{other_kind_Layers_architecture_in_CNN}> *and fig.2*<\ref{Layers_architecture_in_CNN}> *is the last convolution layer is replaced by the spreading out layer.*

The layers architecture can be combined in other order. For example, fig.3 <\ref{other_kind_Layers_architecture_in_CNN}> shows another layers architecture, and the difference between fig.3 <\ref{other_kind_Layers_architecture_in_CNN}> and fig.2 <\ref{Layers_architecture_in_CNN}> is the last convolution layer is replaced by the spreading out layer.