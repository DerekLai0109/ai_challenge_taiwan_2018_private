# LSTM Introduction

MM 03/11/2018



Due to insufficient and decaying error back flow, learning to store information over extended time intervals via recurrent backpropagation takes a long time. The Hochreiter's 1991 analysis of this problem is brief reviewed in \[1\]. An efficient, gradient-based method \`\`Long Short-Term Memory" \(LSTM\) is introduced \[1\]. Truncating the gradient where this does not harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing constant error flow through \`\`constant error carrousels"  within special units.Multiplicative gate units learn to open and close access to the constant error flow.LSTM is local in space and time; its computational complexity per time step and weight is O\(1\). 

The experiments with the artificial data involve local, distributed, real-valued, and noisy pattern representation. In comparisons with \`\`Back-Propagation Through Time" \(BPTT\),  \`\`Real-Time Recurrent Learning" \(RTRL\), Recurrent successful runs, and learns much faster. LSTM also solves complex, artificial long time lag tasks that have never been solved by previous recurrent network algorithms.Recurrent networks can  use their feedback connections to store representation of recent input events in form of activations \(\`\`short-term memory", as opposed to \`\`long-term memory" embodied by slowly changing weights\). This is significant for many applications, including speech processing, non-Markovian control, and music composition.The most widely used algorithms for learning what to put in short-term memory, however, take too much to or do not work well at all, especially when minimal time lags between inputs and corresponding teacher signals are long.Although theoretically fascinating, existing methods do not provide clear practical advantages over backprop in feedforward nets with limited time windows. An analysis of the problem and a remedy is suggested in \[1\].

### The problem: 

With conventional \`\`Back-Propagation Through Time" \(BPTT\) or \`\`Real-Time Recurrent Learning" \(RTRL\), error signals \`\`flowing backwards in time" tend to either \(1\) blow up or \(2\) vanish: the temporal evolution of the backpropagated error exponentially depends on the size of the weights. Case \(1\) may lead to oscillating weights, while in case \(2\) learning to bridge long time lags takes a prohibitive amount of time, or does not work at all.

### The remedy: 

\`\`Long Short-Time Memory" \(LSTM\), a novel recurrent network architecture in conjunction with an appropriate gradient-based learning algorithm is present \cite{S.Hochreiter\_1997}. LSTM is designed to overcome these error back-flow problems. It can learn to bridge time intervals in excess of 1000 steps even in case of noisy, incompressible input sequences, without loss of short time lag capabilities.This is achieved by an efficient, gradient-based algorithm for an architecture enforcing constant \(thus neither exploding nor vanishing\) error flow through internal states of special units \(provided the gradient computation is truncated at certain architecture-specific points, this does not affect long term error flow thoughA naive approach to constant error backpro is introduced for didactic purposes, and its problem cnocerning information storage and retrieval is highlighted.The LSTM architecture is described. Numerous experiments and comparisons with competing methods are present. LSTM was shown to outperform other, and it leans to solve complex, artificial tasks no other recurrent net algorithm has solved. The limitations and advantages are discussed. The appendix  contains detailed description of algorithm and explicit flow formulae.





\[1\]

S. Hochreiter, J. Schmidhuber, \`\`Long short-term memory", Neural computation, vol.9, issue 8, pp.1735-1780, 1997.

