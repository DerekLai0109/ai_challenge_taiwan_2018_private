# Acoustic model

AnAn 3/18 2018 unfinished

CCL 3/24 modified

AnAn 3/25 Add content: HMM, 2.1, and 2.2\(unfinished\)

1.	Basic Approach to Speech Recognition

A speech recognition system transforms a piece of speech into a sentence. The input is the speech signal and the output is a sequence of words in a computer.![](/assets/speech recognition.jpg)

**Fig.1. Schematic of Speech Recognition**

The figure above shows the schematic to implement a speech recognition process. First, a front-end signal processing is needed to transform the signal from the real world into the digital signals that is easy to compute by computers. Second, the feature extraction process extracts the linguistic features from the digital signals. These features can be represented by vectors. Third, computers output the most appropriate word sequence from features vectors by the trained acoustic models, the lexicon, and the trained language models.

The acoustic models are to predict the phonemes or syllables from the feature vectors, and the lexicon is a database to form basic phonetic unit into words, and the language models are to predict the appropriate sentence from the possible word sequences. Both the acoustic models and language models can be trained by machine learning, and the lexicon has done by linguists.

2.	Acoustic Model

An acoustic model can be implemented by an HMM\[1\], which is shown by the figure below. 

![](/assets/Acoustic HMM.jpg)

**Fig.2. HMM for Acoustic**

An HMM \(Hidden Markov Model\) is a statistical model in which the system being modeled is assumed to be a sequence with unobserved \(i.e. hidden\) states. The states S1, S2, ……, SN can represent a speech. The number of the states is proportional to the sound changes of the speech. 

\(AnAn: the HMM is a big topic that I have not understand it.\)

 2.1.	Unit Selection Principles of Sound

Speech signal is complex for computers. The units of sound signal are hard to distinguish and each of the sound productions by humans can be changed because of the adjacent units.

The typical units of sounds defined by linguists are words, syllables, and phonemes. A syllable, which includes 1 vowel and zero or more consonants, and a phoneme, which stands for 1 vowel or 1 consonant, are more basic units of sounds. For example, he word ‘student’ has 2 syllables, ‘stu’ and ‘dent’, and has 8 phonemes, that is \[s\], \[t\], \[j\],\[ ʊ\],\[d\],\[ ə\],\[n\],\[t\].

The units of sounds selection consider as follow.

1.	Accuracy: the selected unit can descript the phone accurately.

2.	Trainability: the data is enough for model training.

3.	Generalizability: the selected unit can form into most of the words.

![](/assets/IMAG0288.jpg)

**Fig.3. Different Unit Selection of Sound**

Words and phonemes are the extreme cases among these units of sounds. A word usually consists of many phonemes that can carry a lot of information for high accuracy. However, collecting words for training is difficult and the trained words cannot form other new words resulting in poor generalizability. Phonemes are most basic units of sounds and are easy to be collected. The trained phonemes can form most of words or syllables result in good generalizability. However, the pronunciation of a phoneme varies by its adjacent phonemes, both front and back, which result in poor accuracy.

According to current researches, triphones is a better selection of unit of sounds. A triphone is a phoneme that should be considered its adjacent phonemes. The phonemes \[æ\] in the words ‘pat’ and ‘bat’, for example, are considered as the same phoneme but different triphones because of their adjacent phonemes \[p\] and \[b\].

The number of triphones in a language is the third power of the number of phones in that language by definition. There are about 50 phonemes in a language in general, so the number of the triphones should be,

	\(50\)^3=21500.

which is too many for data collection and training. Therefore, a compromise solution, sharing of parameters and training data of triphones, should be adopted.



Some rare triphones can be merged into one triphone to increase trainability.

Some states of triphones can share.

2.2.	Classification and Regression Tree

A decision tree asks a bunch of questions to predict the unknown property of objects.

Decision question should be defined by linguist

100~300 questions

To implement a decision tree, every node splitting should decrease the cross entropy as hard as possible.

\(AnAn: unfinshed\)



---

\[0\] 數位語音處理概論 第五章 Acoustic Model, 李琳山 https://www.youtube.com/watch?time\_continue=226&v=j0gQ8K3QjWU

\[1\] Phonemes are the minimum units of speech sound in a language which can serve to distinguish one word from the other. For example, the phonetic alphabets \[p\] and \[b\] are the pronunciation of the first phoneme of the words “pat” and “bat”.

