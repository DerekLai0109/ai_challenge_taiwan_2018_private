#Brief Introduction to NLP

## Recent Trends in Deep Learning Based Natural Language Processing

MM 03/01 2018

Natural language processing (NLP) are computational techniques for the automatic analysis and representation of human language. NLP enables computers to perform natural language related tasks.

Machine learning approaches targeting NLP problems have been based on shallow models (e.g., SVM and logistic regression) trained on high dimensional and sparse features.

Neural networks based on dense vector representations have been producing superior results on various NLP tasks. This trend is sparked by the success of word embeddings [1] and deep learning methods [2].

Deep learning enables multi-level automatic feature representation learning.
Traditional machine learning based NLP systems relies on hand-craft features which are time-consuming and incomplete.

Deep learning methods employ multiple processing layers to learn hierarchical representations of data, and have produced state-of-the-art results in many domains [0].

The deep learning framework outperforms state-of-the-art approaches in NLP tasks such as named-entity recognition (NER), semantic role labeling (SRL), and POS tagging [3].

![](/assets/HMM_GMM_DNN_0325.png)

**Fig. 1 Schematic for HMM \(hidden Markov model\), GMM \(Gaussian mixture model\) and DNN \(Deep neural network\).**

Fig.1 shows the schematic of hidden Markov models (HMMs), Gaussian mixture models (GMMs) and deep neural network (DNN). 

The hidden Markov models (HMMs) can be used to deal with the temporal variability of speech.

The Gaussian mixture models (GMMs) can be used to determine how well each state of each HMM fits a frame or a short window of frames of coefficients that represents the acoustic input [4].

An alternative way to evaluate the fit is to use a feed-forward neural network that takes several frames of coefficients as input and produces posterior probabilities over HMM states as output.

Deep neural networks (DNNs) have been shown to outperforms GMMS on a variety of speech recognition benchmarks [4].

Deep learning methods and a technical overview of distributional semantics, i.e., word2vec and CNN, for NLP in a tutorial manner is present in [5]. Various models are summarized, compared in [0].

## NLP tasks

![](/assets/NLP_tasks_0325.png)

**Fig. 2 Syntactic tasks and semantic tasks for NLP.**

Fig. 2 shows the schematic of NLP tasks.

Syntax (文法) is about the structure or the grammar of the language. It answers the question: how do I construct a valid sentence?

Semantics(語意) is about the meaning of the sentence. It answers the questions: is this sentence valid? If so, what does the sentence mean?

Syntax is the concept that concerns itself only whether or not the sentence is valid for the grammar of the language.
Semantics is about whether or not the sentence has a valid meaning.

There are six standard NLP tasks.

### Part-of-Sppeach Tagging (POS)

POS aims at labeling each word with a unique tag that indicates its syntactic role, e.g., plural noun, adverb, ...

### Chunking

Chunking, also called shallow parsing, aims at labeling segments of a sentence with syntactic constituents such as noun or verb phrase (NP or VP). Each word is assigned only one unique tag, often encoded as a begin-chunk (e.g. B-NP) or inside-chunk tag (e.g. I-NP).

### Named Entity Recognition (NER)

NER labels atomic elements in the sentence into categories such as "PERSON", "COMPANY", or "LOCATION".

### Semantic Role Labeling (SRL)

SRL aims at given a semantic role to a syntactic constituent of a sentence.
In the PropBank formalism, one assigns roles ARG0-5 to words that are arguments of a predicate in the sentence, e.g. the following sentence might be tagged "[John]ARG0 [ate]REL [the apple]ARG1", where "ate" is the predicate. The precise arguments depend on a verb's frame and if there are multiple verbs in a sentence, some words might have multiple tags. In addition to the ARG0-5 tags, there are 13 modifier tags such as ARGM-LOC (locational) and ARGM-TMP (temporal) that operate in a similar way for all verbs.

### Language Model.

A language model traditionally estimates the probability of the next word being  in a sequence. A different setting can be considered: predict whether the given sequence exists in nature, or not. This is achieved by labeling real texts as positive examples, and generating "fake" negative text [6].

### Semantically related words ("Synonyms").

This is the task of predicting whether two words are semantically related (synonyms, holonyms, hypernyms,...) which is measured using WordNet database [http://wordnet.princeton.edu\] as ground truth.

## Distributed Representation

![](/assets/statistical_NLP_DNM_0325.png)

**Fig.3 statistical NLP  v.s. deep learning model**

Fig.3 shows the mechanism improvement from learning joint probability to distributed representation.

Statistical NLP has emerged as the primary option for modeling natural language tasks.

However, it often used to suffer from the "curse of dimensionality" while learning joint probability functions of language models. This led to the motivation of learning distributed representations of words existing in low-dimensional space.

The concept of distributed representation is the basis of deep learning model.

[0]

Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 \(2017\).

[1]

T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, ¡§Distributed representations of words and phrases and their compositionality,¡¨ in Advances in neural information processing systems, pp. 3111-3119, 2013.

[2]

R. Socher, A. Perelygin, J. Y. Wu, J. Chuang, C. D. Manning,
A. Y. Ng, C. Potts et al., Recursive deep models for semantic compositionality over a sentiment treebank," in Proceedings of the conference on empirical methods in natural language processing \(EMNLP\), vol. 1631, p. 1642, 2013.

[3]

R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. Kuksa, ¡§Natural language processing \(almost\) from scratch," Journal of Machine Learning Research, vol. 12, no. Aug, pp. 2493-2537, 2011.

[4]

G. Hinton, D. Deng, D. Yu, G. E. Dahl, etc. \`\`Deep Neural Networks for Acoustic Modeling in Speech Recognition,"
IEEE Signal Processing Magzine, vol.82, November, 2012.

[5]

Y. Goldberg, \`\`A primer on neural network models for natu- ral language processing," Journal of Artificial Intelligence Research, vol. 57, pp. 345-420, 2016.

[6]

Okanohara, D., & Tsujii, J. \(2007\). A discriminative language model with pseudo-negative samples. Proceedings of the 45th Annual Meeting of the ACL, 73-80.

