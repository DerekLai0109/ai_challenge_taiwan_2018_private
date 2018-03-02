# Recurrent Neural Networks Tutorial, Part 1 - Introduction to RNNs

Recurrent Neural Networks \(RNNs\) are models in many Neuro Linguistic Programming \(NLP\) tasks. This tutorial will cover the following part:

1. Introduction to RNNs \(this post\)
2. Implementing a RNN using Python and Theano
3. Understanding the Backpropagation Throuth Time \(BPTT\) algorithm and the vanishing gradient problem
4. Implementing a GRU / LSTM RNN

A recurrent neural network based language model has two-fold:

First, it can score arbitrary sentences based on how likely they are. The model gives a measure of grammatical and semantic correctness. Such RNN based language models are typically used as part of machine translation systems.

Secondly, a language model allows us to generate new text. Training a language model on Shakespeare allows us to generate Shakespeare-like text. [This post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andref Karpathy demonstrates the capability of character-level language models based on RNNs.

If you aren't familiar with basic Neural Networks, you may have to read [Implementing A Neural Network From Scratch](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/), which guides you the non-recurrent networks ideas and implementation.

### What are RNNs?

RNNs is to make use of sequential information. In a traditional neural network, all inputs and outputs are independent of each other. But if we want to predict the next word in a sentence, we better know which words came before it. RNNs are called _recurrent_ because neural networks perform the same task for every element of a sequence, with the output depended on the previous computations.

RNNs have a "memory" which captures information about what has been calculated so far. In theory RNNs can make use of information in arbitrarily long sequences, but in pratice neural networks are limited to looking back only a few steps \(more on this later\). Here is what a typical RNN looks like:

![](http://www.wildml.com/wp-content/uploads/2015/09/rnn.jpg)Forward computations are involved in the RNN and the unfolding in time of the computation.

Fig. shows a RNN being _unrolled_ \(or unfolded\) into a full network. Unrolling means that we write out the network for the complete sequence. 

---

vanish v. 消失

gradient n. 梯度

