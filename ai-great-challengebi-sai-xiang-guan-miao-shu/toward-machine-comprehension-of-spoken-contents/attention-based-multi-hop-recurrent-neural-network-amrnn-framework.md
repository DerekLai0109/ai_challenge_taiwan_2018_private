# Attention-based Multi-hop Recurrent Neural Network \(AMRNN\) Model

MM 03/12/2018

\(Fig..xxx\)

**The overall structure of the proposed Attention-based Multi-hop Recurrent Neural Network \(AMRNN\) model.**

Fig. shows the overall structure of the AMRNN model. The input of model includes the transcriptions of an audio story, a question and four answer choices, all represented as word sequences. The word sequence of the input question is first represented as a question vector $$\bar{V}_Q$$, the attention mechanism is applied to extract the question-related information from the story. The machine then goes through the story by the attention mechanism several times and obtain an answer selection vector $$\bar{V}_{Q_n}$$is finally used to evaluate the confidence of each choice, and the choice with the highest score is taken as the output. All the model parameters are jointly trained with the target where 1 for the correct choice and 0 otherwise.

## Question Representation

\(Fig. xxx\)

**\(A\) The Question Vector Representation and \(B\) The Attention Mechanism.  
**

Fig. shows the procedure of encoding the input question into a vector representation $$\bar{V}_Q$$. The input question is a sequence of T words, $$w1,w_2, \cdots, w_T$$, every word $$W_i$$ represented in 1-of-N encoding. A bidirectional Gated Recurrent Unit \(GRU\) network takes one word from the input question sequentially at a time.

In Fig., the hidden layer output of the forward GRU \(green rectangle\) at time index t is denoted by $$y_f(t)$$, and that of the backward GRU \(blue rectangle\) is by $$y_b(t)$$. After looking through all the words in the question, the hidden layer output of forward GRU network at the last time index $$y_f(T)$$, and that of backward GRU network at the first time index $$y_b(1)$$, are concantenated to form the question vector representation $$\bar{V}_Q$$, or $$\bar{V}Q=[y_f(T)||y_b(1)]$$.

$$\odot$$

The symbol $$[ \cdot || \cdot ]$$ denotes concatenation of two vectors.

The symbol $$\odot$$ denotes cosine similarity between two vectors.

