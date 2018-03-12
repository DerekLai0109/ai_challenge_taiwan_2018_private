# Towards Machine Comprehension of Spoken Content: Initial TOEFL Listening Comprehension Test by Machine

## Abstract

A new task of machine comprehension of spoken content was proposed, and the initial goal is to comprehend TOEFL test, a challenging academic English examination for English learners whose native language is not English.

An Attention-based Multi-hop Recurrent Neural Network \(AM-RNN\) architecture is proposed. Initial results have shon that word-level attentions is more robust than sentence0level attention with ASR errors.

## Introduction

With the popularity of shared videos, social networks, online course, etc, the quantity of multimedia or spoken content is growing much faster beyond what human beings can view or listen to.

Accessing large collections of multimedia or spoden content is difficult and time-consuming for humans.

It will be great if the machine can automatically listen to and understand teh spoken content, and even visualize the key information for humans.

In an initial task, the machine is expected to listen to and understand an audio story, and answer the questions related to that audio content.

The listening comprehension task considered is related to Spoken Question Answering \[1\].

In SQA, when the users enter questions in either text or spoken form, the machine needs to find the answer from some audio files.

SQA usually worked with ASR transcripts of the spoken content, and used information retrieval \(IR\) techniques \[2\] or relied on kowledge bases \[3\] to find the proper answer.

A factoild SQA system used some IR techniques and utilized several levels of linquistic information to deal with the task.

Question Answering in Speech Transcripts \(QAST\) has been a evaluation program of SQA for years \[4\].

However, most previous works on SQA mainly focused on factoid questions like " what is the name of the higest mountain in Taiwan?"

Sometimes this kind of questions may be correctly answered by extracting the key terms from a properly chosen uterance without understanding the given spoken content.

Neural netwroks have successfully applied to speech recognition \[5\] or NLP tasks.

A number of recent efforts have explored various ways to understand multimedia intext form \[6\].

They incorporated attention mechanisms \[7\] with LSTM networks \[8\].

In Question Answering field, most of the works focused on understanding text documents \[9\].

\[0\]

B. H. Tseng, S. S. Shen, H. Y. Lee, L. S. Lee, \`\`Towards machine comprehension of spoken content: Initial TOEFL listening comprehension test by machine," {\em Towards machine comprehension of spoken content: Initial TOEFL listening comprehension test by machine}, 2016.

\[1\]

P. R. C. i Umbert, “Factoid question answering for spoken documents,” Ph.D. dissertation, Universitat Politecnica de Catalunya, \`2012

\[2\] S.-R. Shiang, H.-y. Lee, and L.-s. Lee, “Spoken question answering using tree-structured conditional random fields and two-layer random walk.” in INTERSPEECH, 2014, pp. 263–267.

\[3\] B. Hixon, P. Clark, and H. Hajishirzi, “Learning knowledge graphs for question answering through conversational dialog.”

\[4\] J. Turmo, P. R. Comas, S. Rosset, O. Galibert, N. Moreau, D. Mostefa, P. Rosso, and D. Buscaldi, Multilingual Information

Access Evaluation I. Text Retrieval Experiments: 10th Workshop of the Cross-Language Evaluation Forum, CLEF 2009, Corfu,

Greece, September 30 - October 2, 2009, Revised Selected Papers. Springer Berlin Heidelberg, 2010, ch. Overview of QAST

2009, pp. 197–211.

\[5\] A. Graves, A.-r. Mohamed, and G. Hinton, “Speech recognition with deep recurrent neural networks,” in Acoustics, Speech and

Signal Processing \(ICASSP\), 2013 IEEE International Conference on. IEEE, 2013, pp. 6645–6649.

\[6\]

A. M. Rush, S. Chopra, and J. Weston, “A neural attention model for abstractive sentence summarization,” arXiv preprint

arXiv:1509.00685, 2015

\[7\]

S. Sukhbaatar, J. Weston, R. Fergus et al., “End-to-end memory networks,” in Advances in Neural Information Processing Systems,  
 2015, pp. 2431–2439

\[8\]

S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural computation, vol. 9, no. 8, pp. 1735–1780, 1997.

\[9\]

A. Bordes, S. Chopra, and J. Weston, “Question answering with subgraph embeddings,” arXiv preprint arXiv:1406.3676, 2014.

