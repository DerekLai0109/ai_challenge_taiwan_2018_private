# Language Model

安安0309/2018

### 一、語言模型\(Language Model\)

語言模型描述一個句子\(sentence\)、字串\(Word Sequence\)的機率。

一個字串\(word sequence\)由許多字排序組合而成

字串=\[W1, W2, W3, ……, Wn\], 其中W表示一個字\(Word\)。  
一個字串的機率可以表現為：

P\(W1, W2, W3, ……, Wn\)

舉例來說，下列兩個英文字串：

字串一：“recognize speech”

字串二：“wreck a nice beach”

這兩個字串在英文發音上非常相似，在意思上卻完全不一樣。

當機器聽到類似這兩個字串的發音時，應當判斷當下的字串為“recognize speech”的機率高一些。  
因此，如果語言模型的判斷上字串一的機率較字串二高，則輸出的字串將會是字串一，其數學表示如下：

If P\(“recognize speech”\) &gt; P\(“wreck a nice beach”\)

Output = “recognize speech”

語言模型可用於上述判斷聽到的句子應當為何者，也可以用於翻譯和句子產生。當一個聊天機器人必須選擇一個句子作為回應時，語言模型幫助機器人選擇較正確且符合文法的句子。

### 二、N-gram語言模型

在N-gram語言模型的中，評估一個字串的機率，即是計算字之間的條件機率。以Bi-gram語言模型為例，字串W1, W2, W3, ……, Wn的機率可表示如下：

P\(W1, W2, W3, ……, Wn\) = P\(W1\|START\) P\(W2\|W1\)……P\(Wn\|Wn-1\)

上式的條件機率P\(Wi\|Wi-1\)的計算來自語料庫，舉例來說，當出現”nice”時，”beach”緊接在後的機率是，將語料庫中出現”nice beach”的次數除以出現”nice”的次數：

P\(“nice beach”\|“nice”\) = C\(“nice beach”\)÷C\(“nice”\)

這樣的語言模型稱為Bi-gram語言模型，因為這個條件機率使用了兩個字發生的次數。相同的概念下，也可以從語料庫中產生Tri-gram或4-gram等語言模型。如下為Tri-gram語言模型的一個例子。

P\(“a nice beach”\|“a nice”\) = C\(“a nice beach”\)÷C\(“a nice”\)

### 三、神經網絡語言模型\(Neural Network Language Model\)

神經網絡語言模型依然以條件機率的方式呈現一個字串的機率，然而不同於N-gram語言模型從語料庫統計以換算機率，一個神經網絡被大量語料庫的資料訓練之後，能夠做到給定輸入字後，預測下一個字的機率。其訓練方法主要是調整神經網絡的參數降低其的輸出\(output\)與目標\(target\)的交叉熵\[1\]\(Cross Entropy\)。

![](/assets/language_model_1.png)

重複輸入字到神經網絡就可獲得構成字串的條件機率，如下所示。

P\(“wreck a nice beach”\) =

P\(“wreck”\|START\)P\(“a”\|“wreck”\)……P\(“beach”\|“nice”\)

![](/assets/language_model2.png)

在此START表示一個字串將要開始，對神經網絡的輸入來說，視為一個字。

循環神經網絡\(Recurrent Neural Network\)語言模型，簡稱RNN語言模型，可以讓預測某一個字的機率不再是單靠上一個字，而是依靠所有已經出現過的字。

P\(“wreck a nice beach”\) =

=P\(“wreck”\|START\)P\(“a”\|“wreck”\)P\(“nice”\|“wreck a”\)……P\(“beach”\|“wreck a nice”\)

![](/assets/language_model_3.png)

#### 三之一、N-gram語言模型的缺點

N-gram語言模型的機率評估仰賴的是語料庫，語料庫的資料量往往是不夠的：相較於古往今來被說過的語句，目前已有的語料庫顯得微不足道。這產生的問題是，許多詞語的使用是存在的，但所憑藉的語料庫未有這些詞語，因而導致預測其發生機率為零。舉例來說，根據語料庫能找到以下字串：

The dog ran

The cat jumped

但是找不到以下的組合：

The dog jumped

The cat ran

則N-gram語言模型認定該發生的機率為零。

P\(“jumped”\|“the dog”\) = 0

P\(“ran”\|“the cat”\) = 0

然而這些字串確實在統計用的語料庫以外是有使用的。一種解決的方式稱為語言模型平滑\(Language Model Smoothing\)技術，即讓上述條件機率不為零，而是一個極小的機率，例如：

P\(“jumped”\|“the dog”\) = 0.0001

P\(“ran”\|“the cat”\) = 0.0001

然而胡亂給予機率值的方式，很難有好的表現。

#### 三之二、神經網絡語言模型的優點

Bi-gram語言模型可以下表格的方式呈現，表格中每一個數字表示第一欄\(column\)字緊接在第一列\(row\)字後的機率。

![](/assets/language_model_4.png)

例如從表中可以查到：

P\(“jumped”\|“cat”\) = 0.2

上述表格中許多欄位的值為零，表示現有的語料庫並沒有該字串。對於N-gram語言模型來說，發現許多欄位都為零是很常見的。

將各字以向量\(Vector\)的形式表示：第一列以h1, h2, ……, hn表示，其中h表示history；第一欄以v1, v2, ……, vn，其中v表示vocabulary。history與vocabulary之間的機率值則以n11, n12, ……, nij表示。h1, h2, ……, hn和v1, v2, ……, vn的參數由機器學習獲得，盡可能使下式的最小化。

![](/assets/language_model_5.png)

取得h1, h2, ……, hn和v1, v2, ……, vn後，可以重新填入上表而獲得一個新的關係表：

![](/assets/language_model_6.png)

在新的表中，所有的機率值都由向量的內積\(inner product\)取代。其優點為，若兩個向量相似度高，例如dog和cat作為動物名詞都有類似的特性\(兩者緊接ran的機率都高\)，儘管在統計數據上，字串”dog jumped”出現的次數為零，但是代表“cat jumped”的v2·h2值高，代表“dog jumped”v2·h1也因而高些。而v3·h1、v3·h2都會為一極小值，因為dog和cat兩者後緊接cried和laughed的機率原先就為零，而在訓練的過程中，產生的child的向量特性有別於dog和cat兩者的向量特性，儘管child緊接cried和laughed的機率高，但不至於使dog和cat兩者緊接cried和laugh也升高。

上述方法用神經網絡的形式呈現，圖示如下。

![](/assets/language_model_7.png)

輸入為一個1-of-N encoding\[2\]向量，此例中向量中dog的值為1，其餘的的值為0。輸入\(input\)與藍色隱層\(hidden layer\)之間的權重\(weights\)即是hn，此例中由於輸入向量只有dog的值為1，則藍色隱層的輸出為hdog，在此先忽略活化函數\[3\]\(activation function\)。獲得hdog之後，於下一個隱層再乘上向量vn，此例中和為vran和vcried，再經過softmax函數\[5\]，得到各字串的機率此例字串即為P\(“ran”\|“dog”\)和P\(“cried”\|“dog”\)。最後，利用機器學習降低輸出與目標之間的交叉熵，即完成神經網絡的訓練。

若使用N-gram語言模型的方式表示章節二之二的表格，則需要的參數量是欄數乘上列數，而欄數或列數的大小可能至字詞庫\(lexicon\)的量級。然而使用神經網絡語言模型的參數量，只有h和v向量的參數量，由使用者自行設定，通常遠小於字詞庫的量級。神經網絡語言模型使用較少的參數，因而比較不容易發生統計上的過適\(overfitting\)現象。

考慮一個很長的字串W1, W2, ......, Wt，欲估測該字串後緊接Wt+1的機率，若使用N-gram語言模型，則第一列會有V^t的個數，其中V為字詞庫的字數。V^t是個過大而不切實際的數。若使用RNN語言模型，則參數將會大量減少。

ht是整個字串W1, W2, ......, Wt以RNN的方式呈現，無論該字串的字數t有多少，RNN的參數都不會改變。由RNN輸出的ht可以依照之前的程序，乘上vn，以得到下個字出現的機率。

此外，越長字串出現在語料庫的次數會越低，通常導致無法在統計數據上形成有意義的機率分布。RNN語言模型可以避免以上問題，獲得一個長字串之後緊接的字的機率分布。



\[0\] https://www.youtube.com/watch?v=Jvigef51rqk

\[1\]交叉熵簡單來說可以是為兩個機率分布之間的非相關性，若兩者的交叉熵值高，表示兩者的相關性很低。

\[2\]若欲以一向量表達N種可能，1-of-N encoding向量有N個維度

\[3\]活化函數將輸入向量的值轉為介於1到0的數值，可避免數值運算時過大，也方便於表達機率。

\[4\] softmax函數的概念是取出一輸入向量中的維度最大值，但其餘維度並沒有完全捨棄。

\[5\]過適現象發生於統計模型使用過多參數，但擁有的數據過少，反而破壞了模型一般化的能力。

