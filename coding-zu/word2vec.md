# Distributed Representations of Sentences and Documents

#### Review of Le & Mikolov \(2014\) and introduction into Word2Vec

---

### Word2Vec Introduction

The "learning continuous word embeddings" method and tool has developed by Mikolov et al \(2013\). The model forms the basis for the study of Le & Mikolov \(2014\). To have a more complete understanding of the model, look at this paper [Xin Rong \(2014\)](http://arxiv.org/abs/1411.2738).

Word2Vec attempts to associate words with points in space. The spatial distance between words describes the relation \(similarity\). Words that are spatially close, are similar. Words are represented by continuous vectors over _x_ dimensions. This example shows the relation between words, where each word is represented by a vector of two dimensions:

```py
%matplotlib inline
```

```py
import seaborn as sb
import numpy as np

words = ['queen', 'book', 'king', 'magazine', 'car', 'bike']
vectors = np.array([[0.1,   0.3],  # queen
                    [-0.5, -0.1],  # book
                    [0.2,   0.2],  # king
                    [-0.3, -0.2],  # magazine
                    [-0.5,  0.4],  # car
                    [-0.45, 0.3]]) # bike

sb.plt.plot(vectors[:, 0], vectors[:, 1], 'o')
sb.plt.xlim(-0.6, 0.3)
sb.plt.ylim(-0.3, 0.5)
for word, x, y in zip(words, vectors[:, 0], vectors[:, 1]):
    sb.plt.annotate(word, (x, y), size=12)
```

![](/assets/1.png)



**Reference**

[http://nbviewer.jupyter.org/github/fbkarsdorp/doc2vec/blob/master/doc2vec.ipynb](http://nbviewer.jupyter.org/github/fbkarsdorp/doc2vec/blob/master/doc2vec.ipynb)

