#Mathematical Model for Forward Problem

<\begin{table}
<\caption{Mathematical Notation for CNN}
\section{Mathematical Model}

\begin{table}
\caption{Mathematical Notation for CNN}
\label{mathematic_notation_CNN}
\vskip 0cm
\hskip 1cm
\bigskip
\centering
\begin{tabular}{|p{4.5 cm}|p{3 cm}|}
\hline 
description & symbol \\ \hline
input image size  & $H_0 \times W_0$    \\ \hline
input image pixel & $x(i,j)$ or $y_1^0(i,j)$ \\ \hline
layer index & $\ell$ \\ \hline
number of layers & $L= 2a+2$ \\ \hline
convolution layers & $C^1$, $C^3$, $\cdots$, $C^{2a+1}$ \\ \hline
sub-sampling layers & $S^1$, $S^3$, $\cdots$, $S^{2a}$ \\ \hline
output layer & $F^{2a+2}$ \\ \hline
activation function of layer $\ell$ & $f_\ell$ \\ \hline
number of feature maps in layer $\ell$ & $N_\ell$ \\ \hline
size of convolution mask for layer & $h_\ell \times w_\ell$ \\ \hline
convolution mask from feature map $m$ in layer $S^{\ell-1}$ to feature map $n$ in layer $C^\ell$ & $\{ w^\ell_{m,n}(i,j) \}$ \\ \hline
weight for feature map $n$ in layer $S^\ell$ & $w_n^\ell$ \\ \hline
bias for feature map $n$ in convolution layer $C^\ell$ & $b^\ell_n$ \\ \hline
bias for feature map $n$ in sub-sampling layer $S^\ell$ & $b^\ell_n$ \\ \hline
feature map $n$ in layer $\ell$ & $y_n^\ell(i,j)$ \\ \hline
size of a feature map in layer $\ell$ & $H_\ell \times W_\ell$ \\ \hline
\end{tabular}
\end{table}


Table.\ref{mathematic_notation_CNN} summarizes the notation used to describe the functional aspects of CNN.
The symbol $\ell$ denotes the index of a network layer.
The layer index $\ell$ goes from 1 to $L$, and $L$ is the number of network layers.
In this tutorial, there will be $a+1$ convolution layer, $a$ subsampling layer and one output layer.
$a$ is a positive integer, and $L = 2a +2$. 
$N_\ell$ is the number of feature maps in layer $\ell$, and $f_\ell(\cdot)$ is the activation function of layer $\ell$.
$\bar{\bar{y}}_n^\ell$ is the $n$th feature map (output) of layer $\ell$.


\subsection{Convolution layer}
\begin{figure}[h]
\vskip 6.5 cm
\hskip 0 cm
\special{wmf:a_convolution_layer_in_CNN.jpg x=9 cm y=6.5 cm}
\caption{A convolution layer in a CNN.}
\label{a_convolution_layer_in_CNN}
\end{figure}

Fig.\ref{a_convolution_layer_in_CNN} shows a convolution layer in CNN.
Considering the $n$th feature map in a convolution layer $\ell$, $\ell=1,3,\cdots, 2a+1$, 
$\bar{\bar{w}}_{m,n}^\ell = \{ w_{m,n}^\ell(i,j) \}$ is the convolution mask, which will take in feature map $m$ in layer $(\ell-1)$ and generate  feature map $n$ in layer $\ell$.
$b^\ell_n$ is the bias term assoicated with feature map $n$. 


\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case2.jpg x=9.3 cm y=6. cm}
\caption{The second convolution layer (layer 3) as an illustrative example.}
\label{convolution_layer_in_CNN_case2}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case2} shows an illustrative example for the second convolution layer, layer 3.
The second and fourth feature maps in second layer, $\bar{\bar{y}}_2^2$ and $\bar{\bar{y}}_4^2$, serve as input sources for the fourth feature map in third layer, $\bar{\bar{y}}_4^3$. 




\begin{figure}[h]
\vskip 6.75 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case1.jpg x=9 cm y=6.75 cm}
\caption{First convolution layer as an illustrative example.}
\label{convolution_layer_in_CNN_case1}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case1} shows another illustrative example for the first convolution layer.
The feature maps of 1, 2, 3 and 4 in first layer, $\bar{\bar{y}}_{1}^1$, $\bar{\bar{y}}_{2}^1$, $\bar{\bar{y}}_{3}^1$, $\bar{\bar{y}}_{4}^1$, have idendical input source of the image map $\bar{\bar{y}}_1^0$ in the input layer; however, different convolution masks are applied, $\bar{\bar{w}}_{1,1}^1$, $\bar{\bar{w}}_{1,2}^1$, $\bar{\bar{w}}_{1,3}^1$, $\bar{\bar{w}}_{1,4}^1$. 

\begin{figure}[h]
\vskip 3.5 cm
\hskip 0.3 cm
\special{wmf:schematic_convolution_U_and_V.jpg x=7.5 cm y=3.5 cm}
\caption{Schematic of $U^\ell$ and $V^{\ell+1}$, which are applied to describe the connection between successive layer.}
\label{schematic_convolution_U_and_V}
\end{figure}
Fig.\ref{schematic_convolution_U_and_V} shows the schematic of $U^\ell$ and $V^{\ell+1}$, which are applied to describe the connection between successive layer. 
$U^\ell_n$ stores the next elements connection for $n$th neuron (feature map) in layer $\ell$ to $\ell+1$ layer.
$V^{\ell+1}_m$ stores the preceding elements conection for $m$th neuron (feature map) in layer $\ell+1$ to $\ell$ layer.
In this illustrative example, the first neuron in layer $\ell$ is connected to the first, second and fourth neurons in the layer $\ell+1$ (next layer), and it renders 
\begin{eqnarray}
U_1^\ell=[U_1^\ell(1), U_1^\ell(2), U_1^\ell(3) ] = [1,2,4] \nonumber 
\end{eqnarray}
The second neuron in layer $\ell+1$ is connected to the first, second and third neruons in the $\ell$ layer (preceding neuron), and it renders 
\begin{eqnarray}
V_2^{\ell+1} = [V_2^{\ell+1}(1), V_2^{\ell+1}(2), V_2^{\ell+1}(3) ] =[1,2,3] \nonumber
\end{eqnarray}

Note that the U in $\ell$ layer, $U^\ell$, and V in $\ell+1$ layer, $V^{\ell+1}$ carry equivalent information.
From fig.\ref{schematic_convolution_U_and_V}, it indicates that 
\begin{eqnarray}
&&\hbox{if } m \in U_n^\ell \hspace{0.1 in} \to n \in V_m^{\ell+1} \nonumber \\
&&\hbox{if } n \in V_m^{\ell+1} \to m \in U_n^\ell  \nonumber           
\end{eqnarray}

Typically, fully connections are adopted for CNN, i.e., 
\begin{eqnarray}
&&U_n^\ell=\{ 1,2,\cdots, N_{\ell+1} \} \nonumber \\
&&V_n^\ell=\{ 1,2, \cdots, N_{\ell-1} \} \nonumber
\end{eqnarray}


\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case1_general.jpg x=9 cm y=6 cm}
\caption{Convolution layer one-multiple mapping ($U^\ell_n$).}
\label{convolution_layer_in_CNN_case1_general}
\end{figure}

\begin{figure}[h]
\vskip 5 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case2_general.jpg x=9 cm y=5 cm}
\caption{Convolution layer multiple-one mapping ($V^\ell_m$).}
\label{convolution_layer_in_CNN_case2_general}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case1_general} and Fig.\ref{convolution_layer_in_CNN_case2_general} show the schematic of convolution layer by exploiting the concept of U (one-multiple mapping) and V (multiple-one mapping).

Feature map $n$ of convolution layer $\ell$ is calculated as
\begin{eqnarray}
&&\bar{\bar{y}}_n^\ell = f_\ell \left( \sum_{m \in V_n^\ell} \bar{\bar{y}}^{\ell -1}_m \bigcirc \!\!\!\!\!\!\! \times \hspace{0.1 in} \bar{\bar{w}}_{m,n}^\ell + b_n^\ell \right)
\label{convolution_layer_output} \\
&&=f_\ell \left( \sum_{m \in V_n^\ell} \bar{\bar{C}}_{m,n}^\ell + b_n^\ell \right) = f_\ell ( \bar{\bar{s}}_n^\ell )
\nonumber 
\end{eqnarray}
where 
\begin{eqnarray}
\bar{\bar{C}}_{m,n}^\ell= \bar{\bar{y}}_m^{\ell-1} \bigcirc \!\!\!\!\!\!\! \times  \hspace{0.1 in}  \bar{\bar{w}}_{m,n}^\ell
\nonumber
\end{eqnarray}
where $\bigcirc \!\!\!\!\! \times$ is the 2D convolution operator. 

The computation of convolution operation can be explicitly expressed as
\begin{eqnarray}
&&\bar{\bar{C}}^\ell_{m,n}(i,j) = \sum_{i'=1}^{ h_\ell} \hspace{0.1 in} \sum_{j'=1}^{w_\ell} \nonumber \\
&&\bar{\bar{y}}_m^{\ell-1}(i'-1+i, j'-1+j) \times \bar{\bar{w}}^\ell_{m,n}(i',j')
\nonumber \\
&&1 \leq i \leq H_{\ell-1} - h_\ell+1, \hspace{0.1 in} 1 \leq j \leq W_{\ell-1} - w_\ell+1 \nonumber 
\end{eqnarray}
Or 
\begin{eqnarray}
&&\bar{\bar{C}}^\ell_{m,n}(i,j)=\sum_{i'=i}^{h_\ell + i -1} \sum_{j'=j}^{w_\ell+j-1} \nonumber \\
&&\bar{\bar{y}}^{\ell-1}_m(i',j') \times \bar{\bar{w}}_{m,n}^\ell (i'-i+1,j'-j+1)
\nonumber \\
&&1 \leq i \leq H_{\ell}, \hspace{0.1 in} 1 \leq j \leq W_{\ell} \nonumber \\
&& 1 \leq i' - i +1 \leq h_\ell, \hspace{0.1 in} 1 \leq j' - j +1 \leq w_\ell \nonumber
\end{eqnarray}

\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:schematic_convolution_operation_01242017.jpg x=8.5 cm y=6 cm}
\caption{Schematic of convolution operation, $\bar{\bar{C}}_{m,n}^{\ell}=  \bar{\bar{y}}^{\ell -1}_m \bigcirc \!\!\!\!\!\!\! \times \hspace{0.1 in} \bar{\bar{w}}_{m,n}^\ell$.}
\label{schematic_convolution_operation_ell_nth_neuron}
\end{figure}
Fig.\ref{schematic_convolution_operation_ell_nth_neuron} shows the schematic of convolution operation.
If the size of input feature maps $\bar{\bar{y}}^{\ell-1}_m$ is $H_{\ell-1} \times W_{\ell-1}$ pixels and the size of convolution masks $\bar{\bar{w}}_{m,n}^\ell$ is $h_\ell \times w_\ell$, the size of output feature map $\bar{\bar{y}}_n^\ell$ is
\begin{eqnarray}
H_\ell \times W_\ell = \left( H_{\ell-1} - h_\ell+1 \right) \times \left( W_{\ell-1} - w_\ell+1 \right) 
\nonumber
\end{eqnarray}



For the last convolution layer, the output will become a scalar.
From (\ref{convolution_layer_output}), the $n$th neuron for the last convolution layer ($L-1$) can be expressed as
\begin{eqnarray}
&&y_n^{L-1} = f_{L-1} (s_n^{L-1}) \nonumber \\
&&s_n^{L-1} = s_n^{L-1}(1,1) = b_n^{L-1} + \sum_{m \in V_n^{L-1}} \sum_{i'=1}^{h_{L-1}} \sum_{j'=1}^{w_{L-1}} \nonumber \\
&&y_m^{L-2}(i',j') \times w_{m,n}^{L-1}(i',j') \nonumber
\end{eqnarray}
From (\ref{convolution_layer_output}), the $n$th neuron for the other convolutioh layer ($\ell=2a+1, a=0,1,2, \cdots$) can be expressed as
\begin{eqnarray}
&&\bar{\bar{y}}_n^\ell = f_\ell (\bar{\bar{s}}_n^\ell) \nonumber \\
&&s_n^\ell(i,j) =b_n^{\ell} + \sum_{m \in V_n^\ell} \sum_{i'=i}^{h_\ell +i-1} \sum_{j'=j}^{w_\ell + j-1} \nonumber \\
&&y_m^{\ell-1}(i',j') \times w_{m,n}^\ell(i'-i+1,j'-j+1) \nonumber
\end{eqnarray}


\subsection{Sub-sampling layer}
\begin{figure}[h]
\vskip 4.3 cm
\hskip 0 cm
\special{wmf:a_subsampling_layer_CNN_01262017.jpg x=9 cm y=4.3 cm}
\caption{A sub-sampling layer in a CNN, where $W^\ell=W^{\ell-1}/2$ and $H^\ell=H^{\ell-1}/2$.}
\label{a_subsampling_layer_CNN}
\end{figure}

Fig.\ref{a_subsampling_layer_CNN} shows a sub-sampling layer in a CNN.
Considering the feature map $n$ in a sub-sampling layer $\ell$, $\ell=2,4,\cdots,2a$, the feature map $n$ of convolution layer $\ell-1$ is divided into non-overlapping blocks of size $2 \times 2$ pixels.

\subsubsection{Average Pooling}
$\bar{\bar{z}}_n^{\ell}$ is a matrix, and its element is obtained by summing the four pixels in each block, 
\begin{eqnarray}
&&z_n^{\ell}(i,j)= y_n^{\ell-1} (2i-1,2j-1) + y_n^{\ell-1} (2i-1, 2j) \nonumber \\
&&+  y_n^{\ell-1} (2i, 2j-1) +  y_n^{\ell-1} (2i, 2j) 
\label{z_y_relation}
\end{eqnarray}


\subsubsection{Max Pooling}

$\bar{\bar{z}}_n^{\ell}$ is a matrix, and its element is assigned with the maximal entry among the four pixels in each block, 
\begin{eqnarray}
&&z_n^{\ell}(i,j)= \max \{ y_n^{\ell-1} (2i-1,2j-1), y_n^{\ell-1} (2i-1, 2j), \nonumber \\
&&  y_n^{\ell-1} (2i, 2j-1),  y_n^{\ell-1} (2i, 2j) \}
\label{z_y_relation_max_pooling}
\end{eqnarray}



The feature map $n$ of sub-sampling layer $\ell$ is calculated as
\begin{eqnarray}
\bar{\bar{y}}_n^\ell = f_\ell( \bar{\bar{z}}_n^{\ell} \times w^\ell_n + b_n^\ell ) =f_\ell(\bar{\bar{s}}_n^\ell)
\label{subsampling_layer_output}
\end{eqnarray}
where  $w^\ell_n$ and $b_n^\ell$ are the weight and bias term, respectively.
The size of feature map $\bar{\bar{y}}_n^\ell$ in sub-sampling layer $\ell$ is
\begin{eqnarray}
H_\ell=  H_{\ell-1}/2, \hspace{0.1 in} W_\ell = W_{\ell-1}/2 
\nonumber
\end{eqnarray}


\subsection{Output Layer Fully Connected Layer}
The output of sigmoidal neuron $n$ is calculated as
\begin{eqnarray}
&&y_n^L = f_L \left( \sum_{m=1}^{N_{L-1}} y_m^{L-1} w_{m,n}^L + b_n^L \right) \label{outupt_layer_output} \\
&& = f_L ( s_n^L )    \label{outupt_layer_output_reduced}
\end{eqnarray}
where  $N^L$ is the number of output sigmoidal neuron, $w_{m,n}^L$ denotes the weight from feature map $m$ of the last convolutional layer, to neuron $n$ of the output layer. $b_n^L$ be the bias term associated with neuron of layer $L$.
The outputs of all sigmoidal neurons form the network outputs:
\begin{eqnarray}
\bar{y}= \left[ y_1^L, y_2^L, \cdots, y_{N_L}^L \right]
\nonumber
\end{eqnarray}


\subsection{SoftMax function (usually adopted for output layer)}
Softmax function model generalizes logistic regression to clasification problems where the class label can take on multiple values.
\begin{eqnarray}
y_n^{\ell}= \frac{\exp( s_n^\ell) }{\displaystyle \sum_{n_\ell=1}^{N_\ell} \exp( s_{n_\ell}^\ell)}
\label{soft_max_function}
\end{eqnarray}
All output values in the range (0,1) and sum up to 1 for softmax function, and this property make it suitable for a probablistic interpretation.
If we have $N_L$ output classes, the softmax can be interpreted as
\begin{eqnarray}
y_n^{L}= \hbox{P}(x \in \hbox{n-th class}) \nonumber 
\end{eqnarray}
\section{Mathematical Model}

\begin{table}
\caption{Mathematical Notation for CNN}
\label{mathematic_notation_CNN}
\vskip 0cm
\hskip 1cm
\bigskip
\centering
\begin{tabular}{|p{4.5 cm}|p{3 cm}|}
\hline 
description & symbol \\ \hline
input image size  & $H_0 \times W_0$    \\ \hline
input image pixel & $x(i,j)$ or $y_1^0(i,j)$ \\ \hline
layer index & $\ell$ \\ \hline
number of layers & $L= 2a+2$ \\ \hline
convolution layers & $C^1$, $C^3$, $\cdots$, $C^{2a+1}$ \\ \hline
sub-sampling layers & $S^1$, $S^3$, $\cdots$, $S^{2a}$ \\ \hline
output layer & $F^{2a+2}$ \\ \hline
activation function of layer $\ell$ & $f_\ell$ \\ \hline
number of feature maps in layer $\ell$ & $N_\ell$ \\ \hline
size of convolution mask for layer & $h_\ell \times w_\ell$ \\ \hline
convolution mask from feature map $m$ in layer $S^{\ell-1}$ to feature map $n$ in layer $C^\ell$ & $\{ w^\ell_{m,n}(i,j) \}$ \\ \hline
weight for feature map $n$ in layer $S^\ell$ & $w_n^\ell$ \\ \hline
bias for feature map $n$ in convolution layer $C^\ell$ & $b^\ell_n$ \\ \hline
bias for feature map $n$ in sub-sampling layer $S^\ell$ & $b^\ell_n$ \\ \hline
feature map $n$ in layer $\ell$ & $y_n^\ell(i,j)$ \\ \hline
size of a feature map in layer $\ell$ & $H_\ell \times W_\ell$ \\ \hline
\end{tabular}
\end{table}


Table.\ref{mathematic_notation_CNN} summarizes the notation used to describe the functional aspects of CNN.
The symbol $\ell$ denotes the index of a network layer.
The layer index $\ell$ goes from 1 to $L$, and $L$ is the number of network layers.
In this tutorial, there will be $a+1$ convolution layer, $a$ subsampling layer and one output layer.
$a$ is a positive integer, and $L = 2a +2$. 
$N_\ell$ is the number of feature maps in layer $\ell$, and $f_\ell(\cdot)$ is the activation function of layer $\ell$.
$\bar{\bar{y}}_n^\ell$ is the $n$th feature map (output) of layer $\ell$.


\subsection{Convolution layer}
\begin{figure}[h]
\vskip 6.5 cm
\hskip 0 cm
\special{wmf:a_convolution_layer_in_CNN.jpg x=9 cm y=6.5 cm}
\caption{A convolution layer in a CNN.}
\label{a_convolution_layer_in_CNN}
\end{figure}

Fig.\ref{a_convolution_layer_in_CNN} shows a convolution layer in CNN.
Considering the $n$th feature map in a convolution layer $\ell$, $\ell=1,3,\cdots, 2a+1$, 
$\bar{\bar{w}}_{m,n}^\ell = \{ w_{m,n}^\ell(i,j) \}$ is the convolution mask, which will take in feature map $m$ in layer $(\ell-1)$ and generate  feature map $n$ in layer $\ell$.
$b^\ell_n$ is the bias term assoicated with feature map $n$. 


\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case2.jpg x=9.3 cm y=6. cm}
\caption{The second convolution layer (layer 3) as an illustrative example.}
\label{convolution_layer_in_CNN_case2}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case2} shows an illustrative example for the second convolution layer, layer 3.
The second and fourth feature maps in second layer, $\bar{\bar{y}}_2^2$ and $\bar{\bar{y}}_4^2$, serve as input sources for the fourth feature map in third layer, $\bar{\bar{y}}_4^3$. 




\begin{figure}[h]
\vskip 6.75 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case1.jpg x=9 cm y=6.75 cm}
\caption{First convolution layer as an illustrative example.}
\label{convolution_layer_in_CNN_case1}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case1} shows another illustrative example for the first convolution layer.
The feature maps of 1, 2, 3 and 4 in first layer, $\bar{\bar{y}}_{1}^1$, $\bar{\bar{y}}_{2}^1$, $\bar{\bar{y}}_{3}^1$, $\bar{\bar{y}}_{4}^1$, have idendical input source of the image map $\bar{\bar{y}}_1^0$ in the input layer; however, different convolution masks are applied, $\bar{\bar{w}}_{1,1}^1$, $\bar{\bar{w}}_{1,2}^1$, $\bar{\bar{w}}_{1,3}^1$, $\bar{\bar{w}}_{1,4}^1$. 

\begin{figure}[h]
\vskip 3.5 cm
\hskip 0.3 cm
\special{wmf:schematic_convolution_U_and_V.jpg x=7.5 cm y=3.5 cm}
\caption{Schematic of $U^\ell$ and $V^{\ell+1}$, which are applied to describe the connection between successive layer.}
\label{schematic_convolution_U_and_V}
\end{figure}
Fig.\ref{schematic_convolution_U_and_V} shows the schematic of $U^\ell$ and $V^{\ell+1}$, which are applied to describe the connection between successive layer. 
$U^\ell_n$ stores the next elements connection for $n$th neuron (feature map) in layer $\ell$ to $\ell+1$ layer.
$V^{\ell+1}_m$ stores the preceding elements conection for $m$th neuron (feature map) in layer $\ell+1$ to $\ell$ layer.
In this illustrative example, the first neuron in layer $\ell$ is connected to the first, second and fourth neurons in the layer $\ell+1$ (next layer), and it renders 
\begin{eqnarray}
U_1^\ell=[U_1^\ell(1), U_1^\ell(2), U_1^\ell(3) ] = [1,2,4] \nonumber 
\end{eqnarray}
The second neuron in layer $\ell+1$ is connected to the first, second and third neruons in the $\ell$ layer (preceding neuron), and it renders 
\begin{eqnarray}
V_2^{\ell+1} = [V_2^{\ell+1}(1), V_2^{\ell+1}(2), V_2^{\ell+1}(3) ] =[1,2,3] \nonumber
\end{eqnarray}

Note that the U in $\ell$ layer, $U^\ell$, and V in $\ell+1$ layer, $V^{\ell+1}$ carry equivalent information.
From fig.\ref{schematic_convolution_U_and_V}, it indicates that 
\begin{eqnarray}
&&\hbox{if } m \in U_n^\ell \hspace{0.1 in} \to n \in V_m^{\ell+1} \nonumber \\
&&\hbox{if } n \in V_m^{\ell+1} \to m \in U_n^\ell  \nonumber           
\end{eqnarray}

Typically, fully connections are adopted for CNN, i.e., 
\begin{eqnarray}
&&U_n^\ell=\{ 1,2,\cdots, N_{\ell+1} \} \nonumber \\
&&V_n^\ell=\{ 1,2, \cdots, N_{\ell-1} \} \nonumber
\end{eqnarray}


\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case1_general.jpg x=9 cm y=6 cm}
\caption{Convolution layer one-multiple mapping ($U^\ell_n$).}
\label{convolution_layer_in_CNN_case1_general}
\end{figure}

\begin{figure}[h]
\vskip 5 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case2_general.jpg x=9 cm y=5 cm}
\caption{Convolution layer multiple-one mapping ($V^\ell_m$).}
\label{convolution_layer_in_CNN_case2_general}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case1_general} and Fig.\ref{convolution_layer_in_CNN_case2_general} show the schematic of convolution layer by exploiting the concept of U (one-multiple mapping) and V (multiple-one mapping).

Feature map $n$ of convolution layer $\ell$ is calculated as
\begin{eqnarray}
&&\bar{\bar{y}}_n^\ell = f_\ell \left( \sum_{m \in V_n^\ell} \bar{\bar{y}}^{\ell -1}_m \bigcirc \!\!\!\!\!\!\! \times \hspace{0.1 in} \bar{\bar{w}}_{m,n}^\ell + b_n^\ell \right)
\label{convolution_layer_output} \\
&&=f_\ell \left( \sum_{m \in V_n^\ell} \bar{\bar{C}}_{m,n}^\ell + b_n^\ell \right) = f_\ell ( \bar{\bar{s}}_n^\ell )
\nonumber 
\end{eqnarray}
where 
\begin{eqnarray}
\bar{\bar{C}}_{m,n}^\ell= \bar{\bar{y}}_m^{\ell-1} \bigcirc \!\!\!\!\!\!\! \times  \hspace{0.1 in}  \bar{\bar{w}}_{m,n}^\ell
\nonumber
\end{eqnarray}
where $\bigcirc \!\!\!\!\! \times$ is the 2D convolution operator. 

The computation of convolution operation can be explicitly expressed as
\begin{eqnarray}
&&\bar{\bar{C}}^\ell_{m,n}(i,j) = \sum_{i'=1}^{ h_\ell} \hspace{0.1 in} \sum_{j'=1}^{w_\ell} \nonumber \\
&&\bar{\bar{y}}_m^{\ell-1}(i'-1+i, j'-1+j) \times \bar{\bar{w}}^\ell_{m,n}(i',j')
\nonumber \\
&&1 \leq i \leq H_{\ell-1} - h_\ell+1, \hspace{0.1 in} 1 \leq j \leq W_{\ell-1} - w_\ell+1 \nonumber 
\end{eqnarray}
Or 
\begin{eqnarray}
&&\bar{\bar{C}}^\ell_{m,n}(i,j)=\sum_{i'=i}^{h_\ell + i -1} \sum_{j'=j}^{w_\ell+j-1} \nonumber \\
&&\bar{\bar{y}}^{\ell-1}_m(i',j') \times \bar{\bar{w}}_{m,n}^\ell (i'-i+1,j'-j+1)
\nonumber \\
&&1 \leq i \leq H_{\ell}, \hspace{0.1 in} 1 \leq j \leq W_{\ell} \nonumber \\
&& 1 \leq i' - i +1 \leq h_\ell, \hspace{0.1 in} 1 \leq j' - j +1 \leq w_\ell \nonumber
\end{eqnarray}

\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:schematic_convolution_operation_01242017.jpg x=8.5 cm y=6 cm}
\caption{Schematic of convolution operation, $\bar{\bar{C}}_{m,n}^{\ell}=  \bar{\bar{y}}^{\ell -1}_m \bigcirc \!\!\!\!\!\!\! \times \hspace{0.1 in} \bar{\bar{w}}_{m,n}^\ell$.}
\label{schematic_convolution_operation_ell_nth_neuron}
\end{figure}
Fig.\ref{schematic_convolution_operation_ell_nth_neuron} shows the schematic of convolution operation.
If the size of input feature maps $\bar{\bar{y}}^{\ell-1}_m$ is $H_{\ell-1} \times W_{\ell-1}$ pixels and the size of convolution masks $\bar{\bar{w}}_{m,n}^\ell$ is $h_\ell \times w_\ell$, the size of output feature map $\bar{\bar{y}}_n^\ell$ is
\begin{eqnarray}
H_\ell \times W_\ell = \left( H_{\ell-1} - h_\ell+1 \right) \times \left( W_{\ell-1} - w_\ell+1 \right) 
\nonumber
\end{eqnarray}



For the last convolution layer, the output will become a scalar.
From (\ref{convolution_layer_output}), the $n$th neuron for the last convolution layer ($L-1$) can be expressed as
\begin{eqnarray}
&&y_n^{L-1} = f_{L-1} (s_n^{L-1}) \nonumber \\
&&s_n^{L-1} = s_n^{L-1}(1,1) = b_n^{L-1} + \sum_{m \in V_n^{L-1}} \sum_{i'=1}^{h_{L-1}} \sum_{j'=1}^{w_{L-1}} \nonumber \\
&&y_m^{L-2}(i',j') \times w_{m,n}^{L-1}(i',j') \nonumber
\end{eqnarray}
From (\ref{convolution_layer_output}), the $n$th neuron for the other convolutioh layer ($\ell=2a+1, a=0,1,2, \cdots$) can be expressed as
\begin{eqnarray}
&&\bar{\bar{y}}_n^\ell = f_\ell (\bar{\bar{s}}_n^\ell) \nonumber \\
&&s_n^\ell(i,j) =b_n^{\ell} + \sum_{m \in V_n^\ell} \sum_{i'=i}^{h_\ell +i-1} \sum_{j'=j}^{w_\ell + j-1} \nonumber \\
&&y_m^{\ell-1}(i',j') \times w_{m,n}^\ell(i'-i+1,j'-j+1) \nonumber
\end{eqnarray}


\subsection{Sub-sampling layer}
\begin{figure}[h]
\vskip 4.3 cm
\hskip 0 cm
\special{wmf:a_subsampling_layer_CNN_01262017.jpg x=9 cm y=4.3 cm}
\caption{A sub-sampling layer in a CNN, where $W^\ell=W^{\ell-1}/2$ and $H^\ell=H^{\ell-1}/2$.}
\label{a_subsampling_layer_CNN}
\end{figure}

Fig.\ref{a_subsampling_layer_CNN} shows a sub-sampling layer in a CNN.
Considering the feature map $n$ in a sub-sampling layer $\ell$, $\ell=2,4,\cdots,2a$, the feature map $n$ of convolution layer $\ell-1$ is divided into non-overlapping blocks of size $2 \times 2$ pixels.

\subsubsection{Average Pooling}
$\bar{\bar{z}}_n^{\ell}$ is a matrix, and its element is obtained by summing the four pixels in each block, 
\begin{eqnarray}
&&z_n^{\ell}(i,j)= y_n^{\ell-1} (2i-1,2j-1) + y_n^{\ell-1} (2i-1, 2j) \nonumber \\
&&+  y_n^{\ell-1} (2i, 2j-1) +  y_n^{\ell-1} (2i, 2j) 
\label{z_y_relation}
\end{eqnarray}


\subsubsection{Max Pooling}

$\bar{\bar{z}}_n^{\ell}$ is a matrix, and its element is assigned with the maximal entry among the four pixels in each block, 
\begin{eqnarray}
&&z_n^{\ell}(i,j)= \max \{ y_n^{\ell-1} (2i-1,2j-1), y_n^{\ell-1} (2i-1, 2j), \nonumber \\
&&  y_n^{\ell-1} (2i, 2j-1),  y_n^{\ell-1} (2i, 2j) \}
\label{z_y_relation_max_pooling}
\end{eqnarray}



The feature map $n$ of sub-sampling layer $\ell$ is calculated as
\begin{eqnarray}
\bar{\bar{y}}_n^\ell = f_\ell( \bar{\bar{z}}_n^{\ell} \times w^\ell_n + b_n^\ell ) =f_\ell(\bar{\bar{s}}_n^\ell)
\label{subsampling_layer_output}
\end{eqnarray}
where  $w^\ell_n$ and $b_n^\ell$ are the weight and bias term, respectively.
The size of feature map $\bar{\bar{y}}_n^\ell$ in sub-sampling layer $\ell$ is
\begin{eqnarray}
H_\ell=  H_{\ell-1}/2, \hspace{0.1 in} W_\ell = W_{\ell-1}/2 
\nonumber
\end{eqnarray}


\subsection{Output Layer Fully Connected Layer}
The output of sigmoidal neuron $n$ is calculated as
\begin{eqnarray}
&&y_n^L = f_L \left( \sum_{m=1}^{N_{L-1}} y_m^{L-1} w_{m,n}^L + b_n^L \right) \label{outupt_layer_output} \\
&& = f_L ( s_n^L )    \label{outupt_layer_output_reduced}
\end{eqnarray}
where  $N^L$ is the number of output sigmoidal neuron, $w_{m,n}^L$ denotes the weight from feature map $m$ of the last convolutional layer, to neuron $n$ of the output layer. $b_n^L$ be the bias term associated with neuron of layer $L$.
The outputs of all sigmoidal neurons form the network outputs:
\begin{eqnarray}
\bar{y}= \left[ y_1^L, y_2^L, \cdots, y_{N_L}^L \right]
\nonumber
\end{eqnarray}


\subsection{SoftMax function (usually adopted for output layer)}
Softmax function model generalizes logistic regression to clasification problems where the class label can take on multiple values.
\begin{eqnarray}
y_n^{\ell}= \frac{\exp( s_n^\ell) }{\displaystyle \sum_{n_\ell=1}^{N_\ell} \exp( s_{n_\ell}^\ell)}
\label{soft_max_function}
\end{eqnarray}
All output values in the range (0,1) and sum up to 1 for softmax function, and this property make it suitable for a probablistic interpretation.
If we have $N_L$ output classes, the softmax can be interpreted as
\begin{eqnarray}
y_n^{L}= \hbox{P}(x \in \hbox{n-th class}) \nonumber 
\end{eqnarray}
\label{mathematic_notation_CNN}
\vskip 0cm
\hskip 1cm
\bigskip
\centering
\begin{tabular}{|p{4.5 cm}|p{3 cm}|}
\hline 
description & symbol \\ \hline
input image size  & $H_0 \times W_0$    \\ \hline
input image pixel & $x(i,j)$ or $y_1^0(i,j)$ \\ \hline
layer index & $\ell$ \\ \hline
number of layers & $L= 2a+2$ \\ \hline
convolution layers & $C^1$, $C^3$, $\cdots$, $C^{2a+1}$ \\ \hline
sub-sampling layers & $S^1$, $S^3$, $\cdots$, $S^{2a}$ \\ \hline
output layer & $F^{2a+2}$ \\ \hline
activation function of layer $\ell$ & $f_\ell$ \\ \hline
number of feature maps in layer $\ell$ & $N_\ell$ \\ \hline
size of convolution mask for layer & $h_\ell \times w_\ell$ \\ \hline
convolution mask from feature map $m$ in layer $S^{\ell-1}$ to feature map $n$ in layer $C^\ell$ & $\{ w^\ell_{m,n}(i,j) \}$ \\ \hline
weight for feature map $n$ in layer $S^\ell$ & $w_n^\ell$ \\ \hline
bias for feature map $n$ in convolution layer $C^\ell$ & $b^\ell_n$ \\ \hline
bias for feature map $n$ in sub-sampling layer $S^\ell$ & $b^\ell_n$ \\ \hline
feature map $n$ in layer $\ell$ & $y_n^\ell(i,j)$ \\ \hline
size of a feature map in layer $\ell$ & $H_\ell \times W_\ell$ \\ \hline
\end{tabular}
\end{table}


Table.\ref{mathematic_notation_CNN} summarizes the notation used to describe the functional aspects of CNN.
The symbol $\ell$ denotes the index of a network layer.
The layer index $\ell$ goes from 1 to $L$, and $L$ is the number of network layers.
In this tutorial, there will be $a+1$ convolution layer, $a$ subsampling layer and one output layer.
$a$ is a positive integer, and $L = 2a +2$. 
$N_\ell$ is the number of feature maps in layer $\ell$, and $f_\ell(\cdot)$ is the activation function of layer $\ell$.
$\bar{\bar{y}}_n^\ell$ is the $n$th feature map (output) of layer $\ell$.


\subsection{Convolution layer}
\begin{figure}[h]
\vskip 6.5 cm
\hskip 0 cm
\special{wmf:a_convolution_layer_in_CNN.jpg x=9 cm y=6.5 cm}
\caption{A convolution layer in a CNN.}
\label{a_convolution_layer_in_CNN}
\end{figure}

Fig.\ref{a_convolution_layer_in_CNN} shows a convolution layer in CNN.
Considering the $n$th feature map in a convolution layer $\ell$, $\ell=1,3,\cdots, 2a+1$, 
$\bar{\bar{w}}_{m,n}^\ell = \{ w_{m,n}^\ell(i,j) \}$ is the convolution mask, which will take in feature map $m$ in layer $(\ell-1)$ and generate  feature map $n$ in layer $\ell$.
$b^\ell_n$ is the bias term assoicated with feature map $n$. 


\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case2.jpg x=9.3 cm y=6. cm}
\caption{The second convolution layer (layer 3) as an illustrative example.}
\label{convolution_layer_in_CNN_case2}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case2} shows an illustrative example for the second convolution layer, layer 3.
The second and fourth feature maps in second layer, $\bar{\bar{y}}_2^2$ and $\bar{\bar{y}}_4^2$, serve as input sources for the fourth feature map in third layer, $\bar{\bar{y}}_4^3$. 




\begin{figure}[h]
\vskip 6.75 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case1.jpg x=9 cm y=6.75 cm}
\caption{First convolution layer as an illustrative example.}
\label{convolution_layer_in_CNN_case1}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case1} shows another illustrative example for the first convolution layer.
The feature maps of 1, 2, 3 and 4 in first layer, $\bar{\bar{y}}_{1}^1$, $\bar{\bar{y}}_{2}^1$, $\bar{\bar{y}}_{3}^1$, $\bar{\bar{y}}_{4}^1$, have idendical input source of the image map $\bar{\bar{y}}_1^0$ in the input layer; however, different convolution masks are applied, $\bar{\bar{w}}_{1,1}^1$, $\bar{\bar{w}}_{1,2}^1$, $\bar{\bar{w}}_{1,3}^1$, $\bar{\bar{w}}_{1,4}^1$. 

\begin{figure}[h]
\vskip 3.5 cm
\hskip 0.3 cm
\special{wmf:schematic_convolution_U_and_V.jpg x=7.5 cm y=3.5 cm}
\caption{Schematic of $U^\ell$ and $V^{\ell+1}$, which are applied to describe the connection between successive layer.}
\label{schematic_convolution_U_and_V}
\end{figure}
Fig.\ref{schematic_convolution_U_and_V} shows the schematic of $U^\ell$ and $V^{\ell+1}$, which are applied to describe the connection between successive layer. 
$U^\ell_n$ stores the next elements connection for $n$th neuron (feature map) in layer $\ell$ to $\ell+1$ layer.
$V^{\ell+1}_m$ stores the preceding elements conection for $m$th neuron (feature map) in layer $\ell+1$ to $\ell$ layer.
In this illustrative example, the first neuron in layer $\ell$ is connected to the first, second and fourth neurons in the layer $\ell+1$ (next layer), and it renders 
\begin{eqnarray}
U_1^\ell=[U_1^\ell(1), U_1^\ell(2), U_1^\ell(3) ] = [1,2,4] \nonumber 
\end{eqnarray}
The second neuron in layer $\ell+1$ is connected to the first, second and third neruons in the $\ell$ layer (preceding neuron), and it renders 
\begin{eqnarray}
V_2^{\ell+1} = [V_2^{\ell+1}(1), V_2^{\ell+1}(2), V_2^{\ell+1}(3) ] =[1,2,3] \nonumber
\end{eqnarray}

Note that the U in $\ell$ layer, $U^\ell$, and V in $\ell+1$ layer, $V^{\ell+1}$ carry equivalent information.
From fig.\ref{schematic_convolution_U_and_V}, it indicates that 
\begin{eqnarray}
&&\hbox{if } m \in U_n^\ell \hspace{0.1 in} \to n \in V_m^{\ell+1} \nonumber \\
&&\hbox{if } n \in V_m^{\ell+1} \to m \in U_n^\ell  \nonumber           
\end{eqnarray}

Typically, fully connections are adopted for CNN, i.e., 
\begin{eqnarray}
&&U_n^\ell=\{ 1,2,\cdots, N_{\ell+1} \} \nonumber \\
&&V_n^\ell=\{ 1,2, \cdots, N_{\ell-1} \} \nonumber
\end{eqnarray}


\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case1_general.jpg x=9 cm y=6 cm}
\caption{Convolution layer one-multiple mapping ($U^\ell_n$).}
\label{convolution_layer_in_CNN_case1_general}
\end{figure}

\begin{figure}[h]
\vskip 5 cm
\hskip 0 cm
\special{wmf:convolution_layer_in_CNN_case2_general.jpg x=9 cm y=5 cm}
\caption{Convolution layer multiple-one mapping ($V^\ell_m$).}
\label{convolution_layer_in_CNN_case2_general}
\end{figure}
Fig.\ref{convolution_layer_in_CNN_case1_general} and Fig.\ref{convolution_layer_in_CNN_case2_general} show the schematic of convolution layer by exploiting the concept of U (one-multiple mapping) and V (multiple-one mapping).

Feature map $n$ of convolution layer $\ell$ is calculated as
\begin{eqnarray}
&&\bar{\bar{y}}_n^\ell = f_\ell \left( \sum_{m \in V_n^\ell} \bar{\bar{y}}^{\ell -1}_m \bigcirc \!\!\!\!\!\!\! \times \hspace{0.1 in} \bar{\bar{w}}_{m,n}^\ell + b_n^\ell \right)
\label{convolution_layer_output} \\
&&=f_\ell \left( \sum_{m \in V_n^\ell} \bar{\bar{C}}_{m,n}^\ell + b_n^\ell \right) = f_\ell ( \bar{\bar{s}}_n^\ell )
\nonumber 
\end{eqnarray}
where 
\begin{eqnarray}
\bar{\bar{C}}_{m,n}^\ell= \bar{\bar{y}}_m^{\ell-1} \bigcirc \!\!\!\!\!\!\! \times  \hspace{0.1 in}  \bar{\bar{w}}_{m,n}^\ell
\nonumber
\end{eqnarray}
where $\bigcirc \!\!\!\!\! \times$ is the 2D convolution operator. 

The computation of convolution operation can be explicitly expressed as
\begin{eqnarray}
&&\bar{\bar{C}}^\ell_{m,n}(i,j) = \sum_{i'=1}^{ h_\ell} \hspace{0.1 in} \sum_{j'=1}^{w_\ell} \nonumber \\
&&\bar{\bar{y}}_m^{\ell-1}(i'-1+i, j'-1+j) \times \bar{\bar{w}}^\ell_{m,n}(i',j')
\nonumber \\
&&1 \leq i \leq H_{\ell-1} - h_\ell+1, \hspace{0.1 in} 1 \leq j \leq W_{\ell-1} - w_\ell+1 \nonumber 
\end{eqnarray}
Or 
\begin{eqnarray}
&&\bar{\bar{C}}^\ell_{m,n}(i,j)=\sum_{i'=i}^{h_\ell + i -1} \sum_{j'=j}^{w_\ell+j-1} \nonumber \\
&&\bar{\bar{y}}^{\ell-1}_m(i',j') \times \bar{\bar{w}}_{m,n}^\ell (i'-i+1,j'-j+1)
\nonumber \\
&&1 \leq i \leq H_{\ell}, \hspace{0.1 in} 1 \leq j \leq W_{\ell} \nonumber \\
&& 1 \leq i' - i +1 \leq h_\ell, \hspace{0.1 in} 1 \leq j' - j +1 \leq w_\ell \nonumber
\end{eqnarray}

\begin{figure}[h]
\vskip 6 cm
\hskip 0 cm
\special{wmf:schematic_convolution_operation_01242017.jpg x=8.5 cm y=6 cm}
\caption{Schematic of convolution operation, $\bar{\bar{C}}_{m,n}^{\ell}=  \bar{\bar{y}}^{\ell -1}_m \bigcirc \!\!\!\!\!\!\! \times \hspace{0.1 in} \bar{\bar{w}}_{m,n}^\ell$.}
\label{schematic_convolution_operation_ell_nth_neuron}
\end{figure}
Fig.\ref{schematic_convolution_operation_ell_nth_neuron} shows the schematic of convolution operation.
If the size of input feature maps $\bar{\bar{y}}^{\ell-1}_m$ is $H_{\ell-1} \times W_{\ell-1}$ pixels and the size of convolution masks $\bar{\bar{w}}_{m,n}^\ell$ is $h_\ell \times w_\ell$, the size of output feature map $\bar{\bar{y}}_n^\ell$ is
\begin{eqnarray}
H_\ell \times W_\ell = \left( H_{\ell-1} - h_\ell+1 \right) \times \left( W_{\ell-1} - w_\ell+1 \right) 
\nonumber
\end{eqnarray}



For the last convolution layer, the output will become a scalar.
From (\ref{convolution_layer_output}), the $n$th neuron for the last convolution layer ($L-1$) can be expressed as
\begin{eqnarray}
&&y_n^{L-1} = f_{L-1} (s_n^{L-1}) \nonumber \\
&&s_n^{L-1} = s_n^{L-1}(1,1) = b_n^{L-1} + \sum_{m \in V_n^{L-1}} \sum_{i'=1}^{h_{L-1}} \sum_{j'=1}^{w_{L-1}} \nonumber \\
&&y_m^{L-2}(i',j') \times w_{m,n}^{L-1}(i',j') \nonumber
\end{eqnarray}
From (\ref{convolution_layer_output}), the $n$th neuron for the other convolutioh layer ($\ell=2a+1, a=0,1,2, \cdots$) can be expressed as
\begin{eqnarray}
&&\bar{\bar{y}}_n^\ell = f_\ell (\bar{\bar{s}}_n^\ell) \nonumber \\
&&s_n^\ell(i,j) =b_n^{\ell} + \sum_{m \in V_n^\ell} \sum_{i'=i}^{h_\ell +i-1} \sum_{j'=j}^{w_\ell + j-1} \nonumber \\
&&y_m^{\ell-1}(i',j') \times w_{m,n}^\ell(i'-i+1,j'-j+1) \nonumber
\end{eqnarray}


\subsection{Sub-sampling layer}
\begin{figure}[h]
\vskip 4.3 cm
\hskip 0 cm
\special{wmf:a_subsampling_layer_CNN_01262017.jpg x=9 cm y=4.3 cm}
\caption{A sub-sampling layer in a CNN, where $W^\ell=W^{\ell-1}/2$ and $H^\ell=H^{\ell-1}/2$.}
\label{a_subsampling_layer_CNN}
\end{figure}

Fig.\ref{a_subsampling_layer_CNN} shows a sub-sampling layer in a CNN.
Considering the feature map $n$ in a sub-sampling layer $\ell$, $\ell=2,4,\cdots,2a$, the feature map $n$ of convolution layer $\ell-1$ is divided into non-overlapping blocks of size $2 \times 2$ pixels.

\subsubsection{Average Pooling}
$\bar{\bar{z}}_n^{\ell}$ is a matrix, and its element is obtained by summing the four pixels in each block, 
\begin{eqnarray}
&&z_n^{\ell}(i,j)= y_n^{\ell-1} (2i-1,2j-1) + y_n^{\ell-1} (2i-1, 2j) \nonumber \\
&&+  y_n^{\ell-1} (2i, 2j-1) +  y_n^{\ell-1} (2i, 2j) 
\label{z_y_relation}
\end{eqnarray}


\subsubsection{Max Pooling}

$\bar{\bar{z}}_n^{\ell}$ is a matrix, and its element is assigned with the maximal entry among the four pixels in each block, 
\begin{eqnarray}
&&z_n^{\ell}(i,j)= \max \{ y_n^{\ell-1} (2i-1,2j-1), y_n^{\ell-1} (2i-1, 2j), \nonumber \\
&&  y_n^{\ell-1} (2i, 2j-1),  y_n^{\ell-1} (2i, 2j) \}
\label{z_y_relation_max_pooling}
\end{eqnarray}



The feature map $n$ of sub-sampling layer $\ell$ is calculated as
\begin{eqnarray}
\bar{\bar{y}}_n^\ell = f_\ell( \bar{\bar{z}}_n^{\ell} \times w^\ell_n + b_n^\ell ) =f_\ell(\bar{\bar{s}}_n^\ell)
\label{subsampling_layer_output}
\end{eqnarray}
where  $w^\ell_n$ and $b_n^\ell$ are the weight and bias term, respectively.
The size of feature map $\bar{\bar{y}}_n^\ell$ in sub-sampling layer $\ell$ is
\begin{eqnarray}
H_\ell=  H_{\ell-1}/2, \hspace{0.1 in} W_\ell = W_{\ell-1}/2 
\nonumber
\end{eqnarray}


\subsection{Output Layer Fully Connected Layer}
The output of sigmoidal neuron $n$ is calculated as
\begin{eqnarray}
&&y_n^L = f_L \left( \sum_{m=1}^{N_{L-1}} y_m^{L-1} w_{m,n}^L + b_n^L \right) \label{outupt_layer_output} \\
&& = f_L ( s_n^L )    \label{outupt_layer_output_reduced}
\end{eqnarray}
where  $N^L$ is the number of output sigmoidal neuron, $w_{m,n}^L$ denotes the weight from feature map $m$ of the last convolutional layer, to neuron $n$ of the output layer. $b_n^L$ be the bias term associated with neuron of layer $L$.
The outputs of all sigmoidal neurons form the network outputs:
\begin{eqnarray}
\bar{y}= \left[ y_1^L, y_2^L, \cdots, y_{N_L}^L \right]
\nonumber
\end{eqnarray}


\subsection{SoftMax function (usually adopted for output layer)}
Softmax function model generalizes logistic regression to clasification problems where the class label can take on multiple values.
\begin{eqnarray}
y_n^{\ell}= \frac{\exp( s_n^\ell) }{\displaystyle \sum_{n_\ell=1}^{N_\ell} \exp( s_{n_\ell}^\ell)}
\label{soft_max_function}
\end{eqnarray}
All output values in the range (0,1) and sum up to 1 for softmax function, and this property make it suitable for a probablistic interpretation.
If we have $N_L$ output classes, the softmax can be interpreted as
\begin{eqnarray}
y_n^{L}= \hbox{P}(x \in \hbox{n-th class}) \nonumber 
\end{eqnarray}
