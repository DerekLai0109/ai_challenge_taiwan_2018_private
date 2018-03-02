# Recurrent Neural Networks Tutorial, Part 1 - Introduction to RNNs

Recurrent Neural Networks \(RNNs\) are models in many Neuro Linguistic Programming \(NLP\) tasks. This tutorial will cover the following part:

1. Introduction to RNNs \(this post\)
2. Implementing a RNN using Python and Theano
3. Understanding the Backpropagation Throuth Time \(BPTT\) algorithm and the vanishing gradient problem
4. Implementing a GRU / LSTM RNN

A recurrent neural network based language model has two-fold: 

First, it can score arbitrary sentences based on how likely they are. The model gives a measure of grammatical and semantic correctness. Such RNN based language models are typically used as part of machine translation systems.

Secondly, a language model allows us to generate new text. Training a language model on Shakespeare allows us to generate Shakespeare-like text. [This post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andref Karpathy demonstrates the capability of character-level language models based on RNNs.



---

vanish v. 消失

gradient n. 梯度

