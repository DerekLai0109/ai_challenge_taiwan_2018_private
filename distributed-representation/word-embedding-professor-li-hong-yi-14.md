# Word Embedding

Word Embedding 是一個非常好廣為人知Unsupervised Learning: Dimension Reduction的應用。

如果要用一個Vector表現一個word，最typical的作法是1-of-N encoding ，N就表示所有Word的數目，每個word對應到其中的一維。例如：如果我們要辨識10萬個Word的話，N就等於10萬。1-of-N encoding的缺點是很難得到word彼此之間的關係，所以Vector一點都不informative，例如：cat vs dog其實都是動物，但是在1-of-N encoding卻是兩個獨立向量。

另外一種用Vector表示word的方法是word class，就是預先把所有的word分成幾大類\(class\)，用這些類\(class\)的index來表示，這個就等同於Dimension reduction裡面的clustering。不過word class還是無法很好表示word跟word之間的關係。



