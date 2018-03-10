# Language Model

AnAn 03/10 2018

### 一、語言模型\(Language Model\)

語言模型描述一個字串\(word sequence\)的機率。對於一個字串：

```
    W1, W2, W3, ……, Wn
```

其中W表示一個字\(word\)，而組合而成的字串即是句子\(sentence\)。

一個字串的機率可以表現如下：

```
    P(W1, W2, W3, ……, Wn)
```

舉例來說，下列兩個英文字串：

```
    字串一：“recognize speech”
```

```
    字串二：“wreck a nice beach”
```

這兩個字串在英文發音上非常相似，在意思上卻完全不一樣。當人或機器聽到類似這兩個字串的發音時，應當判斷該字串為“recognize speech”的機率高一些。

因此，如果語言模型的判斷字串一的機率較字串二高，則輸出的字串將會是字串一，其數學表示如下：

```
    If P(“recognize speech”) > P(“wreck a nice beach”)
```

```
    Output = “recognize speech”
```

語言模型可用於上述判斷聽到的句子應當為何者，也可以用於翻譯和句子產生。當一個聊天機器人必須選擇一個句子作為回應時，語言模型幫助機器人選擇較正確且符合文法的句子。

### 二、N-gram語言模型

在N-gram語言模型的中，評估一個字串的機率，即是計算字之間的條件機率。以Bi-gram語言模型為例，字串W1, W2, W3, ……, Wn的機率可表示如下：

```
    P(W1, W2, W3, ……, Wn) = P(W1|START) P(W2|W1)……P(Wn|Wn-1)
```

上式的條件機率P\(Wi\|Wi-1\)的計算來自語料庫，舉例來說，當出現”nice”時，”beach”緊接在後的機率是，將語料庫中出現”nice beach”的次數除以出現”nice”的次數：

```
    P(“nice beach”|“nice”) = C(“nice beach”) ÷ C(“nice”)
```

這樣的語言模型稱為Bi-gram語言模型，因為這個條件機率使用了兩個字發生的次數。相同的概念下，也可以從語料庫中產生Tri-gram或4-gram等語言模型。如下為Tri-gram語言模型的一個例子。

```
    P(“a nice beach”|“a nice”) = C(“a nice beach”) ÷ C(“a nice”)
```

### 三、神經網絡語言模型\(Neural-network-based Language Model\)

神經網絡語言模型依然以條件機率的方式呈現一個字串的機率，然而不同於N-gram語言模型從語料庫統計以換算機率，一個神經網絡被語料庫的資料訓練之後，能夠做到給定輸入字後，預測下一個字的機率。其訓練方法是調整神經網絡的參數降低其的輸出\(output\)與目標\(target\)的交叉熵\[1\]\(Cross Entropy\)。

![](/assets/language_model_1.png)

Figure 1

重複輸入字到神經網絡就可獲得各個字與字緊接的條件機率，相乘後即為整個字串的機率，如下所示。

```
P(“wreck a nice beach”) =
```

```
P(“wreck”|START)P(“a”|“wreck”)……P(“beach”|“nice”)
```

![](/assets/language_model_2.png)

Figure 2

在此START表示一個字串將要開始，對神經網絡的輸入來說，視為一個字。

循環神經網絡\(Recurrent-neural-network-based\)語言模型，簡稱RNN語言模型，可以讓預測某一個字的機率不再是單靠上一個字，而是依靠所有已經出現過的字。

```
    P(“wreck a nice beach”) =
```

```
    P(“wreck”|START)P(“a”|“wreck”)P(“nice”|“wreck a”) ……P(“beach”|“wreck a nice”)
```

![](/assets/language_model_3.png)

Figure 3

#### 三之一、N-gram語言模型的缺點

N-gram語言模型的機率評估仰賴語料庫，而語料庫的資料量往往是不夠的，尤其N越大時：相較於人類古往今來說過的話，目前已有的語料庫顯得微不足道。這產生的問題是，許多詞語的使用是存在的，但所憑藉的語料庫未有這些詞語，因而導致預測其發生機率為零。舉例來說，根據語料庫能找到以下字串：

```
    The dog ran
```

```
    The cat jumped
```

但是找不到以下的組合：

```
    The dog jumped

    The cat ran
```

則N-gram語言模型認定該發生的機率為零：

```
    P(“jumped”|“the dog”) = 0
```

```
    P(“ran”|“the cat”) = 0
```

然而這些字串確實在統計用的語料庫以外是有被使用過的。一種解決的方式稱為語言模型平滑\(Language Model Smoothing\)技術，即更改上述條件機率，使之不為零而是一個極小的機率，例如：

```
    P(“jumped”|“the dog”) = 0.0001
```

```
    P(“ran”|“the cat”) = 0.0001
```

然而胡亂給予機率值的方式，該語言模型很難有好的表現。

#### 三之二、神經網絡語言模型的優點

欲了解神經網絡語言模型優於N-gram語言模型，將Bi-gram語言模型以下表格的方式呈現，表格中每一個數字表示第一欄\(column\)字緊接在第一列\(row\)字後的機率。\(安安：這方法叫做Matrix Factorization但我的脈絡中很難把這個名詞帶入\)

![](/assets/language_model_4.png)

Table 1

例如從表格可以查到：

```
    P(“jumped”|“cat”) = 0.2
```

上述表格中許多欄位的值為零，表示現有的語料庫並沒有該字串。例如：

```
    P(“jumped”|“dog”) = 0
```

```
    → P(“dog jumped”) = 0
```

對於N-gram語言模型來說，發現許多欄位都為零是很常見的。

接著將表格內各字以向量\(Vector\)的形式表示：第一列以h1, h2, ……, hn表示，其中h表示history；第一欄以v1, v2, ……, vn，其中v表示vocabulary。history與vocabulary之間的機率值則以n11, n12, ……, nij表示。h1, h2, ……, hn和v1, v2, ……, vn的參數由機器學習獲得，盡可能使下式最小化。

![](/assets/language_model_5.png)

取得h1, h2, ……, hn和v1, v2, ……, vn後，可以重新填入上表格：

![](/assets/language_model_6.png)

Table 2

在新的表中，所有的機率值都由向量的內積\(inner product\)取代。其優點為，若兩個向量相似度高，例如dog和cat作為動物名詞都有類似的特性，兩者緊接ran的機率都高，儘管在統計數據上，字串”dog jumped”出現的次數為零，但是代表“cat jumped”的v2·h2值高，代表“dog jumped”的v2·h1也因而遠高於零。而v3·h1和v3·h2都會為一極小值，因為dog和cat兩者後緊接cried和laughed的機率原先就為零，而在訓練的過程中，產生的child的向量特性\(人類名詞\)有別於dog和cat兩者\(動物名詞\)的向量特性，儘管child緊接cried和laughed的機率高，但不至於使dog和cat兩者緊接cried和laugh的機率遠高於零。這方法亦達到語言模型平滑技術，並且是較有根據的。

上述方法用神經網絡的形式呈現，圖示如下。

![](/assets/language_model_7.png)

Figure 4

輸入為一個1-of-N encoding\[2\]向量，此例中向量中dog的值為1，其餘的值為0。輸入\(input\)與藍色隱層\(hidden layer\)之間的權重\(weights\)即是hn，此例中由於輸入向量只有dog的值為1，則藍色隱層的輸出為hdog，在此先忽略活化函數\[3\] \(activation function\)。獲得hdog之後，於下一個隱層再乘上向量vn，此例中和為vran和vcried，再經過softmax函數\[5\]，得到各字串的條件機率，此例即為P\(“ran”\|“dog”\)和P\(“cried”\|“dog”\)。最後，利用機器學習降低輸出與目標之間的交叉熵，即完成神經網絡的訓練。

若使用Table 1表示N-gram語言模型，則需要的參數量是欄數乘上列數，而欄數或列數的大小可能是字詞庫\(lexicon\)的量級。然而使用神經網絡語言模型的參數量，只有h和v向量的參數量，由使用者自行設定，通常遠小於字詞庫的量級。神經網絡語言模型使用較少的參數，因而比較不容易發生統計上的過適\(overfitting\)現象\[5\]。

欲了解N-gram語言模型的參數量，考慮一個很長的字串W1, W2, ......, Wt，估測該字串後緊接Wt+1的機率，使用t-gram語言模型，則第一列的項數會有V^t，其中V為字詞庫的字數。V^t過大而不切實際運算。

若使用RNN語言模型，則參數將會大量減少：ht是整個字串W1, W2, ......, Wt以RNN的方式呈現，無論該字串的字數t有多少，RNN的參數都不會改變。由RNN輸出的ht可以依照之前的程序，乘上vn，以得到下個字出現的機率。

此外，越長字串出現在語料庫的次數會越低，通常導致無法在統計數據上形成有意義的機率分布。RNN語言模型可以避免以上問題，獲得一個長字串之後緊接的字的機率分布。

\(安安：自從有了神經網絡模型，N-gram語言模型就完全被取代嗎？\)

---

\[1\] 交叉熵簡單來說是兩個機率分布之間的非相關性，若兩者的交叉熵值高，表示兩者的相關性很低。

\[2\] 1-of-N encoding向量有N個維度，其規則為：令向量的某一個維度為1，其餘維度為0。該項量能夠表達N種可能。

\[3\] 活化函數將輸入向量各維度的值轉為介於1到0的數值，可避免數值運算時過大，也方便於表達機率。

\[4\] softmax函數的概念是取出一輸入向量各維度中，有最大值的維度，但其餘維度並沒有完全捨棄。

\[5\] 過適現象發生於統計模型使用過多參數，而擁有的數據過少。這樣的統計模型容易破壞模型一般化的能力。

## Reference

\[0\] [https://www.youtube.com/watch?v=xCGidAeyS4M&t=8s](https://www.youtube.com/watch?v=xCGidAeyS4M&t=8s)

