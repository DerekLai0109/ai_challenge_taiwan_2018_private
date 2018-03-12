# Attention-based Multi-hop Recurrent Neural Network \(AMRNN\) Model

MM 03/12/2018

\(Fig..xxx\)

**The overall structure of the proposed Attention-based Multi-hop Recurrent Neural Network \(AMRNN\) model.**

Fig. shows the overall structure of the AMRNN model. The input of model includes the transcriptions of an audio story, a question and four answer choices, all represented as word sequences. The word sequence of the input question is first represented as a question vector $$\bar{V}_Q$$, the attention mechanism is applied to extract the question-related information from the story. The machine then goes through the story by the attention mechanism several times and obtain an answer selection vector $$\bar{V}_{Q_n}$$is finally used to evaluate the confidence of each choice, and the choice with the highest score is taken as the output. All the model parameters are jointly trained with the target where 1 for the correct choice and 0 otherwise.

