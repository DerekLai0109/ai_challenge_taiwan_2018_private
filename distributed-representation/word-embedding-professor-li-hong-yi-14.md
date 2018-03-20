# Word Embedding

#### 2018/3/20 Tsung-Yu Tsai

Word Embedding 是一個非常好廣為人知Unsupervised Learning: Dimension Reduction的應用。

如果要用一個Vector表現一個word，最typical的作法是1-of-N encoding ，N就表示所有Word的數目，每個word對應到其中的一維。例如：如果我們要辨識10萬個Word的話，N就等於10萬。1-of-N encoding的缺點是很難得到word彼此之間的關係，所以Vector一點都不informative，例如：cat vs dog其實都是動物，但是在1-of-N encoding卻是兩個獨立向量。

另外一種用Vector表示word的方法是word class，就是預先把所有的word分成幾大類\(class\)，用這些類\(class\)的index來表示，這個就等同於Dimension reduction裡面的clustering。不過word class還是無法很好表示word跟word之間的關係。

Word Embedding是把word投影到一個高維的空間(雖然是高維，但是維度仍然遠小於所有詞的數量)，這個投影\(project\)能夠滿足：

1. similar words are closer  
2. each dimension has unique semantic meaning

為什麼word embedding是unsupervised learning?因為我們有明確的input，但是沒有明確的output。那我們能不能用Auto-encoder來做word embedding呢？從前人的研究結果顯示，Auto-encoder並不是一個好的model，如果我們是用1-of-N encoding當作input的話，auto-encoder完全沒辦法學到有用的訊息。如果是用character-level encoding例如：N-gram，可能可以抓到一個字首字根的關係，但是還沒有word之間的關係。

Word Embedding 的基礎是：A word can be understood by its context，例如：馬英九520宣誓就職 vs 蔡英文520宣誓就職，雖然機器不曉得蔡英文跟馬英文是什麼意思，但是從給出的句子，機器可以學到馬英九跟蔡英文應該是相似的詞。裡面又可以更細分為counting-based跟prediction-based

#### Counting-based vs Prediction-based

Counting-based method 是計算兩個詞在context裡面同時出現的機率來計算兩個詞的相似度。以Glove Vector為例，對於兩個word $$w_
i, w_j$$ ，對word embedding function $$V$$， $$V(w_i) \cdot V(w_j)$$ 跟兩個詞在文章中同時出現的次數 $$N_{ij}$$ 要盡量靠近。

Predicition based approach 如圖一所示，給一個 word $$w_{i-1}$$，它能夠預測下一個word $$w_i$$ 是誰？整個神經網路把$$w_{i-1}$$的1-of-N encoding當作input，經過網路的計算，output代表每個word是 $$w_i$$ 的機率。經過充分訓練之後，我們取該網路第一層neurons的值當作我們的word vector $$V(w)$$，用它來代表word $$w$$。

![Prediction-based Word Embedding](/assets/prediction-based method.png)
*圖一 Prediction-based Word Embedding*

同樣的原理，我們也可以使用N個word $$(w_{i-N},w_{i-N+1},\cdots, w_{i-1})$$ 來預測下一個word $$w_i$$ ，我們用一個N=2的model，如圖二所示：

![N-word Prediction-based Word Embedding](/assets/N-word Prediction-based Word Embedding.png)
*圖二 N-word Prediction-based Word Embedding*

$$w_{i-1}$$ 跟 $$w_{i-2}$$ 的1-of-N encoding $$x_{i-1}$$ 和 $$x_{i-2}$$ 的維度都是 $$|V|$$，word embedding vector $$z$$ 維度是 $$|Z|$$。根據圖二，$$z$$ 是 $$x_{i-1}$$, $$x_{i-2}$$ 的線性組合。
$$
z = W_1 x_{i-2} + W_2 x_{i-1}
$$

通常我們會讓$$W_1=W_2=W$$，代表我們對於各個word的權重是共享的。原因有兩個，一個原因顯而易見，就是降低要訓練的權重數量，第二個的原因是如果每個時間點的權重不一樣的話，代表我們是相同的word在不同時間點會產生不同的word vector。那我們在訓練的時候如何保持權重相同，方式就是在原本back-propagation更新時，除了自己的偏微分之外，也加入其他同層參數的偏微分，如下面的公式：
$$
{w_i} \leftarrow {w_i} + \eta {\textstyle{{\partial C} \over {\partial {w_i}}}} + \eta {\textstyle{{\partial C} \over {\partial {w_j}}}} \\
{w_j} \leftarrow {w_j} + \eta {\textstyle{{\partial C} \over {\partial {w_j}}}} + \eta {\textstyle{{\partial C} \over {\partial {w_i}}}}
$$

![Prediction-based Training](/assets/Prediction-based Training.png)
*圖三 Prediction-based Training*

在Word Embedding的架構裡面，最常使用的架構有兩種：
1. Continous Bag-of-Word (CBOW) model
2. skip-gram model

![CBOW vs skip-gram model](/assets/CBOW vs skip-gram model.png) 
*圖四 CBOW model vs skip-gram model*


\[0\]
H. Y. Lee, ML lecture \#14, Word Embedding, at

[https://www.youtube.com/watch?v=X7PH3NuYW0Q&t=1846s](https://www.youtube.com/watch?v=X7PH3NuYW0Q&t=1846s)

