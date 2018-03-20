# Word Embedding

#### 2018/3/20 Tsung-Yu Tsai

Word Embedding 是一個非常好廣為人知Unsupervised Learning: Dimension Reduction的應用。

如果要用一個Vector表現一個word，最typical的作法是1-of-N encoding ，N就表示所有Word的數目，每個word對應到其中的一維。例如：如果我們要辨識10萬個Word的話，N就等於10萬。1-of-N encoding的缺點是很難得到word彼此之間的關係，所以Vector一點都不informative，例如：cat vs dog其實都是動物，但是在1-of-N encoding卻是兩個獨立向量。

另外一種用Vector表示word的方法是word class，就是預先把所有的word分成幾大類\(class\)，用這些類\(class\)的index來表示，這個就等同於Dimension reduction裡面的clustering。不過word class還是無法很好表示word跟word之間的關係。

Word Embedding是把word投影到一個高維的空間，這個投影\(project\)能夠滿足：

1. similar words are closer  
2. each dimension has unique semantic meaning

為什麼word embedding是unsupervised learning?因為我們有明確的input，但是沒有明確的output。那我們能不能用Auto-encoder來做word embedding呢？從前人的研究結果顯示，Auto-encoder並不是一個好的model，如果我們是用1-of-N encoding當作input的話，auto-encoder完全沒辦法學到有用的訊息。如果是用character-level encoding例如：N-gram，可能可以抓到一個字首字根的關係，但是還沒有word之間的關係。

Word Embedding 的基礎是：A word can be understood by its context，例如：馬英九520宣誓就職 vs 蔡英文520宣誓就職，雖然機器不曉得蔡英文跟馬英文是什麼意思，但是從給出的句子，機器可以學到馬英九跟蔡英文應該是相似的詞。裡面又可以更細分為counting-based跟prediction-based

#### Counting-based vs Prediction-based 



\[0\]

H. Y. Lee, ML lecture \#14, Word Embedding, at

[https://www.youtube.com/watch?v=X7PH3NuYW0Q&t=1846s](https://www.youtube.com/watch?v=X7PH3NuYW0Q&t=1846s)

