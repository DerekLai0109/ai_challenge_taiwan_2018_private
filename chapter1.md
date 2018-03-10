# NLP tasks

MM 03/01 2018

![](/assets/NLP_task.png)Syntactic tasks and semantic tasks for NLP

Fig. shows the scheatic of NLP tasks.

Syntax \(文法\) is about the structure or the grammar of the language. It answers the question: how do I construct a valid sentence?

Semantics\(語意\) is about the meaning of the sentence. It answers the questions: is this sentence valid? If so, what does the sentence mean?

Syntax is the concept that concerns itself only whether or not the sentence is valid for the grammar of the language.  
Semantics is about whether or not the sentence has a valid meaning.

There are six standard NLP tasks.

#### Part-of-Sppeach Tagging \(POS\)

POS aims at labeling each word with a unique tag that indicates its syntactic role, e.g., plural noun, adverb, ...

#### Chunking

Chunking, also called shallow parsing, aims at labeling segments of a sentence with syntactic constituents such as noun or verb phrase \(NP or VP\). Each word is assigned only one unique tag, often encoded as a begin-chunk \(e.g. B-NP\) or inside-chunk tag \(e.g. I-NP\).

#### Named Entity Recognition \(NER\)

NER labels atmoic elements in the sentence into categoreis such as "PERSON", "COMPANY", or "LOCATION".

#### Semantic Role Labeling \(SRL\)

SRL aims at given a semantic role to a syntactic constituent of a sentence.  
In the PropBank formalism, one assigns roles ARG0-5 to words that are arguments of a predicate in the sentence, e.g. the following sentence might be tagged "\[John\]ARG0 \[ate\]REL \[the apple\]ARG1", where "ate" is the predicate. The precise arguments depend on a verb's frame and if there are multiple verbs in a sentence, some words might have multiple tags. In addition to the ARG0-5 tags, there are 13 modifier tags such as ARGM-LOC \(locational\) and ARGM-TMP \(temporal\) that operate in a similar way for all verbs.

#### Language Model.

A language model traditionally estimates the probability of the next word being $w$ in a sequence. A different setting can be considered: predict whether the given sequence exists in nature, or not. This is achieved by labeling real texts as positive examples, and generating "fake" negative text \[1\].

#### Semantically related words \("Synonyms"\).

This is the task of predicting whether two words are semantically related \(synonyms, holonyms, hypernyms,...\) which is measured using WordNet database \([http://wordnet.princeton.edu\](http://wordnet.princeton.edu%29%29\) as ground truth.

\[0\]

Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 \(2017\).

\[1\]

Okanohara, D., & Tsujii, J. \(2007\). A discriminative language model with pseudo-negative samples. Proceedings of the 45th Annual Meeting of the ACL, 73-80.

