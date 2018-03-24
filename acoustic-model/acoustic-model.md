# Acoustic model

AnAn 3/18 2018 unfinished

1. Basic Approach tfor Speech Recognition

As speech recognition system transforms a piece of speech into a sentence. The input is the sound signal in the real world and the output is word sequence in a computer.

To implement a speech recognition process, first, a front-end signal processing is needed to transform the signal from the real world into the digital signals that is easy to compute by computers. Second, the feature extraction process extracts the linguistic features from the digital signals. These features can be represented by vectors. Third, computers output the most appropriate word sequence from features vectors by the trained acoustic models, the lexicon, and the trained language models.

![](/assets/speech recognition.jpg)

The acoustic models are to predict the phonemes or syllables from the feature vectors, and the lexicon is a database to form basic phonetic unit into words, and the language models are to predict the appropriate sentence from the possible word sequences. Both the acoustic models and language models can be trained by machine learning, and the lexicon has done by linguists.

1. Acoustic Model

An acoustic model can be implemented by an HMM\[1\]. It can deal with phrases, words, syllables, or phonemes\[2\] of a speech.

Speech signal is complex. Sound production can be changed because of the neighboring units.

1. Unit Selection Principles of Sound

\[1\] An HMM\(Hidden Markov Model\) is a probabilistic model. The state sequence of an HMM is hidden but generates the observation sequence.

\[2\] Phonemes are the minimum units of speech sound in a language which can serve to distinguish one word from the other. For example, the phonetic alphabets \[p\] and \[b\] are the pronunciation of the first phoneme of the words “pat” and “bat”.

