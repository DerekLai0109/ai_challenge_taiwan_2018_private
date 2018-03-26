# Mathematical Model

Table.1 summarizes the notation used to describe the functional aspects of CNN. The symbol $$\ell$$ denotes the index of a network layer. The layer index $$\ell$$ goes from 1 to $$L$$, and $$L$$ is the number of network layers. In this tutorial, there will be $$a+1$$ convolution layer, $$a$$ subsampling layer and one output layer. $$a$$ is a positive integer, and $$L = 2a +2$$. $$N_\ell$$ is the number of feature maps in layer $$\ell$$, and $$f_\ell(\cdot)$$ is the activation function of layer $$\ell$$. $$\bar{\bar{y}}_n^\ell$$ is the $$n$$th feature map \(output\) of layer $$\ell$$.

### Table.1 MATHMATICAL NOTATION FOR CNN

| description | symbol |
| -- | -- |
| input image size | $$H_0 \times W_0$$ |
| input image pixel | $$x(i,j)$$ or $$y_1^0(i,j)$$ |
| layer index | $$\ell$$ |
| number of layers | $$L= 2a+2$$ |
| convolution layers | $$C^1$$, $$C^3$$, $$\cdots$$, $$C^{2a+1}$$ |
| sub-sampling layers | $$S^1$$, $$S^3$$, $$\cdots$$, $$S^{2a}$$ |
| activation function of layer $$\ell$$ | $$f_\ell$$ |
| number of feature maps in layer $$\ell$$ | $$N_\ell$$ |
| size of convolution mask for layer | $$h_\ell \times w_\ell$$ |
| convolution mask from feature map $$m$$ &lt;br/&gt; in layer $$S^{\ell-1}$$ to feature map $$n$$ in layer $$C^\ell$$ | $${ w^\ell_{m,n}(i,j) }$$ |
| weight for feature map $$n$$ in layer $$S^\ell$$ | $$w_n^\ell$$ |
| bias for feature map $$n$$ in convolution layer $$C^\ell$$ | $$b^\ell_n$$ |
| bias for feature map $$n$$ in sub-sampling layer $$S^\ell$$ | $$b^\ell_n$$ |
| feature map $$n$$ in layer $$\ell$$ | $$y_n^\ell(i,j)$$ |
| size of a feature map in layer $$\ell$$ | $$H_\ell \times W_\ell$$ |

#### Convolution layer

![a\_convolution\_layer\_in\_CNN\](/assets/a_convolution_layer_in_CNN.jpg)

*Fig.1 A convolution layer in a CNN.*

Fig.1 shows a convolution layer in CNN. Considering the $$n$$th feature map in a convolution layer $$\ell$$, $$\ell=1,3,\cdots, 2a+1$$, $$\bar{\bar{w}}_{m,n}^\ell = { w_{m,n}^\ell(i,j) }$$ is the convolution mask, which will take in feature map $m$ in layer $$(\ell-1)$$ and generate feature map $$n$$ in layer $$\ell$$. $$b^\ell_n$$ is the bias term assoicated with feature map $$n$$.

!\[convolution\\_layer\\_in\\_CNN\\_case2\]\(/assets/convolution\_layer\_in\_CNN\_case2.jpg\)

\*Fig.2 The second convolution layer \\(layer 3\\) as an illustrative example.\*

Fig.2 &lt;\ref{convolution\\_layer\\_in\\_CNN\\_case2}&gt; shows an illustrative example for the second convolution layer, layer 3.

The second and fourth feature maps in second layer, $$\bar{\bar{y}}\_2^2$$ and $$\bar{\bar{y}}\_4^2$$, serve as input sources for the fourth feature map in third layer, $$\bar{\bar{y}}\_4^3$$.

!\[convolution\\_layer\\_in\\_CNN\\_case1\]\(/assets/convolution\_layer\_in\_CNN\_case1.jpg\)

\*Fig.3 First convolution layer as an illustrative example.\*

Fig.3 &lt;\ref{convolution\_layer\\_in\\_CNN\\_case1}&gt; shows another illustrative example for the first convolution layer. The feature maps of 1, 2, 3 and 4 in first layer, $$\bar{\bar{y}}\_{1}^1$$, $$\bar{\bar{y}}\_{2}^1$$, $$\bar{\bar{y}}\_{3}^1$$, $$\bar{\bar{y}}\_{4}^1$$, have idendical input source of the image map $$\bar{\bar{y}}\_1^0$$ in the input layer; however, different convolution masks are applied, $$\bar{\bar{w}}\_{1,1}^1$$, $$\bar{\bar{w}}\_{1,2}^1$$, $$\bar{\bar{w}}\_{1,3}^1$$, $$\bar{\bar{w}}\_{1,4}^1$$.

!\[schematic\\_convolution\\_U\\_and\\_V\]\(/assets/schematic\_convolution\_U\_and\_V.jpg\)

\*Fig.4 Schematic of $$U^\ell$$ and $$V^{\ell+1}$$, which are applied to describe the connection between successive layer.\*

Fig.4 &lt;\ref{schematic\\_convolution\\_U\\_and\\_V}&gt; shows the schematic of $$U^\ell$$ and $$V^{\ell+1}$$, which are applied to describe the connection between successive layer. $$U^\ell\_n$$ stores the next elements connection for $$n$$th neuron \\(feature map\\) in layer $$\ell$$ to $$\ell+1$$ layer. $$V^{\ell+1}\\_m$$ stores the preceding elements conection for $$m$$th neuron \\(feature map\\) in layer $$\ell+1$$ to $$\ell$$ layer. In this illustrative example, the first neuron in layer $$\ell$$ is connected to the first, second and fourth neurons in the layer $$\ell+1$$ \\(next layer\\), and it renders


$$
U\_1^\ell = \[U\_1^\ell\(1\), U\_1^\ell\(2\), U\_1^\ell\(3\) \] = \[1,2,4\] \nonumber
$$


The second neuron in layer $$\ell+1$$ is connected to the first, second and third neruons in the $$\ell$$ layer \\(preceding neuron\\), and it renders


$$
V\_2^{\ell+1} = \[V\_2^{\ell+1}\(1\), V\_2^{\ell+1}\(2\), V\_2^{\ell+1}\(3\) \] = \[1,2,3\] \nonumber
$$


Note that the U in $$\ell$$ layer, $$U^\ell$$, and V in $$\ell+1$$ layer, $$V^{\ell+1}$$ carry equivalent information.

From fig.4 &lt;\ref{schematic\\_convolution\\_U\\_and\\_V}&gt;, it indicates that


$$
\hbox{if } m \in U\\_n^\ell \hspace{0.1 in} \to n \in V\\_m^{\ell+1} \nonumber \\

\hbox{if } n \in V\\_m^{\ell+1} \to m \in U\\_n^\ell \nonumber
$$


Typically, fully connections are adopted for CNN, i.e.,


$$
U\_n^\ell={ 1,2,\cdots, N\_{\ell+1} } \\

V\_n^\ell={ 1,2, \cdots, N\_{\ell-1} } \nonumber
$$


\begin{figure}\\[h\\]

\vskip 6 cm

\hskip 0 cm

\special{wmf:convolution\\_layer\\_in\\_CNN\\_case1\\_general.jpg x=9 cm y=6 cm}

\caption{Convolution layer one-multiple mapping \\($U^\ell\\_n$\\).}

\label{convolution\\_layer\\_in\\_CNN\\_case1\\_general}

\end{figure}

\begin{figure}\\[h\\]

\vskip 5 cm

\hskip 0 cm

\special{wmf:convolution\\_layer\\_in\\_CNN\\_case2\\_general.jpg x=9 cm y=5 cm}

\caption{Convolution layer multiple-one mapping \\($V^\ell\\_m$\\).}

\label{convolution\\_layer\\_in\\_CNN\\_case2\\_general}

\end{figure}

Fig.\ref{convolution\\_layer\\_in\\_CNN\\_case1\\_general} and Fig.\ref{convolution\\_layer\\_in\\_CNN\\_case2\\_general} show the schematic of convolution layer by exploiting the concept of U \\(one-multiple mapping\\) and V \\(multiple-one mapping\\).

Feature map $$n$$ of convolution layer $$\ell$$ is calculated as


$$
\bar{\bar{y}}\_n^\ell = f\_\ell \left\(\sum\_{m \in V\_n^\ell} \bar{\bar{y}}^{\ell -1}\_m \otimes

\bar{\bar{w}}\_{m,n}^\ell + b\_n^\ell \right\) \\

=f\_\ell \left\( \sum\_{m \in V\_n^\ell} \bar{\bar{C}}\_{m,n}^\ell + b\_n^\ell \right\) = f\_\ell \( \bar{\bar{s}}\_n^\ell \) \nonumber
$$


is

where

\begin{eqnarray}

\bar{\bar{C}}\_{m,n}^\ell= \bar{\bar{y}}\_m^{\ell-1} \bigcirc !!!!!!! \times \hspace{0.1 in} \bar{\bar{w}}\_{m,n}^\ell

\nonumber

\end{eqnarray}

where $\bigcirc !!!!! \times$ is the 2D convolution operator.

The computation of convolution operation can be explicitly expressed as

\begin{eqnarray}

&&\bar{\bar{C}}^\ell\_{m,n}\\(i,j\\) = \sum\_{i'=1}^{ h\_\ell} \hspace{0.1 in} \sum\_{j'=1}^{w\_\ell} \nonumber \

&&\bar{\bar{y}}\_m^{\ell-1}\\(i'-1+i, j'-1+j\\) \times \bar{\bar{w}}^\ell\_{m,n}\\(i',j'\\)

\nonumber \

&&1 \leq i \leq H\_{\ell-1} - h\_\ell+1, \hspace{0.1 in} 1 \leq j \leq W\_{\ell-1} - w\_\ell+1 \nonumber

\end{eqnarray}

Or

\begin{eqnarray}

&&\bar{\bar{C}}^\ell\_{m,n}\\(i,j\\)=\sum\_{i'=i}^{h\_\ell + i -1} \sum\_{j'=j}^{w\_\ell+j-1} \nonumber \

&&\bar{\bar{y}}^{\ell-1}\_m\\(i',j'\\) \times \bar{\bar{w}}\_{m,n}^\ell \\(i'-i+1,j'-j+1\\)

\nonumber \

&&1 \leq i \leq H\_{\ell}, \hspace{0.1 in} 1 \leq j \leq W\_{\ell} \nonumber \

&& 1 \leq i' - i +1 \leq h\_\ell, \hspace{0.1 in} 1 \leq j' - j +1 \leq w\_\ell \nonumber

\end{eqnarray}

\begin{figure}\\[h\\]

\vskip 6 cm

\hskip 0 cm

\special{wmf:schematic\_convolution\\_operation\\_01242017.jpg x=8.5 cm y=6 cm}

\caption{Schematic of convolution operation, $\bar{\bar{C}}\_{m,n}^{\ell}= \bar{\bar{y}}^{\ell -1}\_m \bigcirc !!!!!!! \times \hspace{0.1 in} \bar{\bar{w}}\_{m,n}^\ell$.}

\label{schematic\_convolution\\_operation\\_ell\\_nth\\_neuron}

\end{figure}

Fig.\ref{schematic\\_convolution\\_operation\\_ell\\_nth\\_neuron} shows the schematic of convolution operation.

If the size of input feature maps $\bar{\bar{y}}^{\ell-1}\_m$ is $H\_{\ell-1} \times W\_{\ell-1}$ pixels and the size of convolution masks $\bar{\bar{w}}\_{m,n}^\ell$ is $h\_\ell \times w\_\ell$, the size of output feature map $\bar{\bar{y}}\_n^\ell$ is

\begin{eqnarray}

H\_\ell \times W\_\ell = \left\\( H\_{\ell-1} - h\_\ell+1 \right\\) \times \left\\( W\_{\ell-1} - w\\_\ell+1 \right\\)

\nonumber

\end{eqnarray}

For the last convolution layer, the output will become a scalar.

From \\(\ref{convolution\_layer\\_output}\\), the $n$th neuron for the last convolution layer \\($L-1$\\) can be expressed as

\begin{eqnarray}

&&y\_n^{L-1} = f\_{L-1} \\(s\_n^{L-1}\\) \nonumber \

&&s\_n^{L-1} = s\\_n^{L-1}\\(1,1\\) = b\\_n^{L-1} + \sum\_{m \in V\_n^{L-1}} \sum\_{i'=1}^{h\_{L-1}} \sum\_{j'=1}^{w\_{L-1}} \nonumber \

&&y\_m^{L-2}\\(i',j'\\) \times w\_{m,n}^{L-1}\\(i',j'\\) \nonumber

\end{eqnarray}

From \\(\ref{convolution\_layer\\_output}\\), the $n$th neuron for the other convolutioh layer \\($\ell=2a+1, a=0,1,2, \cdots$\\) can be expressed as

\begin{eqnarray}

&&\bar{\bar{y}}\_n^\ell = f\_\ell \\(\bar{\bar{s}}\_n^\ell\\) \nonumber \

&&s\_n^\ell\\(i,j\\) =b\\_n^{\ell} + \sum\_{m \in V\_n^\ell} \sum\_{i'=i}^{h\_\ell +i-1} \sum\_{j'=j}^{w\_\ell + j-1} \nonumber \

&&y\_m^{\ell-1}\\(i',j'\\) \times w\_{m,n}^\ell\\(i'-i+1,j'-j+1\\) \nonumber

\end{eqnarray}

\subsection{Sub-sampling layer}

\begin{figure}\\[h\\]

\vskip 4.3 cm

\hskip 0 cm

\special{wmf:a\\_subsampling\\_layer\\_CNN\\_01262017.jpg x=9 cm y=4.3 cm}

\caption{A sub-sampling layer in a CNN, where $W^\ell=W^{\ell-1}/2$ and $H^\ell=H^{\ell-1}/2$.}

\label{a\\_subsampling\\_layer\\_CNN}

\end{figure}

Fig.\ref{a\\_subsampling\\_layer\\_CNN} shows a sub-sampling layer in a CNN.

Considering the feature map $n$ in a sub-sampling layer $\ell$, $\ell=2,4,\cdots,2a$, the feature map $n$ of convolution layer $\ell-1$ is divided into non-overlapping blocks of size $2 \times 2$ pixels.

\subsubsection{Average Pooling}

$\bar{\bar{z}}\\_n^{\ell}$ is a matrix, and its element is obtained by summing the four pixels in each block,

\begin{eqnarray}

&&z\\_n^{\ell}\\(i,j\\)= y\\_n^{\ell-1} \\(2i-1,2j-1\\) + y\\_n^{\ell-1} \\(2i-1, 2j\\) \nonumber \

&&+ y\\_n^{\ell-1} \\(2i, 2j-1\\) + y\\_n^{\ell-1} \\(2i, 2j\\)

\label{z\\_y\\_relation}

\end{eqnarray}

\subsubsection{Max Pooling}

$\bar{\bar{z}}\\_n^{\ell}$ is a matrix, and its element is assigned with the maximal entry among the four pixels in each block,

\begin{eqnarray}

&&z\\_n^{\ell}\\(i,j\\)= \max { y\\_n^{\ell-1} \\(2i-1,2j-1\\), y\\_n^{\ell-1} \\(2i-1, 2j\\), \nonumber \

&& y\\_n^{\ell-1} \\(2i, 2j-1\\), y\\_n^{\ell-1} \\(2i, 2j\\) }

\label{z\\_y\\_relation\\_max\\_pooling}

\end{eqnarray}

The feature map $n$ of sub-sampling layer $\ell$ is calculated as

\begin{eqnarray}

\bar{\bar{y}}\_n^\ell = f\_\ell\\( \bar{\bar{z}}\_n^{\ell} \times w^\ell\_n + b\\_n^\ell \\) =f\_\ell\\(\bar{\bar{s}}\_n^\ell\\)

\label{subsampling\\_layer\\_output}

\end{eqnarray}

where $w^\ell\\_n$ and $b\\_n^\ell$ are the weight and bias term, respectively.

The size of feature map $\bar{\bar{y}}\\_n^\ell$ in sub-sampling layer $\ell$ is

\begin{eqnarray}

H\_\ell= H\_{\ell-1}/2, \hspace{0.1 in} W\_\ell = W\\_{\ell-1}/2

\nonumber

\end{eqnarray}

\subsection{Output Layer Fully Connected Layer}

The output of sigmoidal neuron $n$ is calculated as

\begin{eqnarray}

&&y\_n^L = f\_L \left\\( \sum\_{m=1}^{N\_{L-1}} y\_m^{L-1} w\_{m,n}^L + b\_n^L \right\\) \label{outupt\\_layer\\_output} \

&& = f\\_L \\( s\\_n^L \\) \label{outupt\\_layer\\_output\\_reduced}

\end{eqnarray}

where $N^L$ is the number of output sigmoidal neuron, $w\_{m,n}^L$ denotes the weight from feature map $m$ of the last convolutional layer, to neuron $n$ of the output layer. $b\_n^L$ be the bias term associated with neuron of layer $L$.

The outputs of all sigmoidal neurons form the network outputs:

\begin{eqnarray}

\bar{y}= \left\\[ y\_1^L, y\\_2^L, \cdots, y\_{N\\_L}^L \right\\]

\nonumber

\end{eqnarray}

\subsection{SoftMax function \\(usually adopted for output layer\\)}

Softmax function model generalizes logistic regression to clasification problems where the class label can take on multiple values.

\begin{eqnarray}

y\_n^{\ell}= \frac{\exp\\( s\_n^\ell\\) }{\displaystyle \sum\_{n\_\ell=1}^{N\_\ell} \exp\\( s\_{n\_\ell}^\ell\\)}

\label{soft\\_max\\_function}

\end{eqnarray}

All output values in the range \\(0,1\\) and sum up to 1 for softmax function, and this property make it suitable for a probablistic interpretation.

If we have $N\\_L$ output classes, the softmax can be interpreted as

\begin{eqnarray}

y\\_n^{L}= \hbox{P}\\(x \in \hbox{n-th class}\\) \nonumber

\end{eqnarray}

