# Recurrent Neural Network \(Professor 李宏毅 ML \#26 45-90 mins\)

如果要做learning的話要做cost function

Recurrent neural network

Training data

了解sentence的label

Cost要怎麼定呢?

Word sequence要當成一個整體來看

Cost就是每個時間點的對象

算他的cross antropogy

Taipei丟進去的話

X2丟進去之前要先知道X1

Cost就是每個時間點的Cross antrop

有了loss function\(L\)之後，要怎麼去做?

計算w對XXX的偏微分

為了計算方便，也有開發一個演算法叫做BPTT這邊不講DBPTT

RNN就是用Gradient decent去train

RNN training is difficult to learn

Loss應該慢慢地會下降，因為參數愈來越多，每次train可以用的資料就越多

但是綠色的線條可能會產生?

第一個想法是程式有bug

發明word vector的人有很長一段時間只有他能train起model

\(講故事的時間\)：如何解決RNN的問題?

RNN的error surface空間上有些地方像是懸崖峭壁

Total loss對參數\(w1, w2, w3…\)的變化，非常的陡峭

踩在懸崖上的的gradient很大，如果learning \* gradient就會跳出去，調整參數之後就飛出去了

用了一招：只有他可以讓他RNN的model可以training

就是Clipping，設定gradient&gt; assigned value，就固定為assigned value

Sigmoid function可以讓gradient非常的小嗎？

ReLU

改變w的時候，對於最後的output的影響會有多大呢?

Gradient&gt;1設定small learning rate

Gradient&lt;1設定large learning rate

這樣會造成我們很難調整learning rate

難training RNN的原因不是在activation，而是在gradient

Solution: Long Short-term Memory

可以解決gradient vanishing的問題

RNN和LSTM在面對memory裡面的資訊，operation是不一樣的

RNN每次memory都會被完全洗掉

LSTM把原來memory\*值+ input放入cell裡面

OH~ LSTM裡面會對memory影響的值只要一產生之後，就會一值存在

LSTM 1997年提出，當初沒有forget gate，總共有3個gate

用gate操控memory的cell: gated recurrent unit\(GRU\)只有兩個gate

精神是：舊的不去，新的不來

forget gate

input gate之間的關係

identity

many vectors to one vector

利用RNN給machine去看一篇文章，他會去找裡面有哪些關鍵詞彙?

Many vector to many vectors

長sequence轉成短sequence

e.g. speech recognition

為了要把好棒跟好棒棒分開

那就用了一招connectionist temporal classification\(CTC\)

多了一個null值

因此，CTC Training

窮舉所有的alignments當成都是正確的

有個巧妙的演算法解決不用全部列完的

英文辨識英文字母+空白

傳說google的語音辨識，現在已經全部都轉換成CTC training了

如果沒有出現過的詞彙第一次被講，也能被正確的辨識出來

不知道英文和中文裡，input & output哪個長？哪個短？

推文接龍

回到machine learning裡面，add a symbol ===讓他斷掉

Google brain：seq 2 seq learning

Input某種語言的聲音 opunt另種語言的文字

英文的聲音訊號轉成中文的文字

目前有做到輸入法文語音，model已經可以做到可辨識成中文

需要有語音和對應的翻譯丟進去

做到beyond sequence

產生syntactic parsing如何讓machine得到一個樹狀結構?

有了seq 2 seq之後，

過去要用structure learning的技術，只需要把樹狀圖描述成一個sequence

Output就是一個syntactic parsing tree，就可training起來

考慮Word sequence的情況之後，來解決字彙量一樣，但組起來語句不同的問題=&gt; seq 2 seq encoder接著可以具有hierarchy

同時也可以用到speech上，

語音的搜尋

不需要聲音辨識，只需要做聲音相似度的辨識即可audio segment to vector

Audio segment =&gt; acoustic features =&gt; RNN encoder =&gt; RNN Decoder

LSTM encoder =&gt; LSTM decoder

除了RNN以外，有用到memory的model是attention-based model

Input =&gt; DNN/RNN =&gt; ouput

多了reading head controller及writing head controller

稱為Neural Turing Machine

Query=&gt;中央處理器\(DNN/RNN\)去控制reading head controller如果讀到相關的information之後，就把header放在上面，再放回去DNN/RNN=&gt; get answer

Keras有example在處理讀一句話回答答案

可以餵機器visual /語音輸入\(TOFEL\)，來回答答案

實驗結果隨意猜的結果是25%  
 shortest或是一選項和另外三個選項語意相關的話，這兩種方法可以到達35%

Memory Network可以達到39.2%

proposed approach:可以到達48.8%

Deep & Structure Learning的關係

| RNN, LSTM | HMM, CRF, Structured Perceptron/SVM |
| :--- | :--- |
| Unidirectional RNN不考慮整個seq | 因為用Viterbi，所以可考慮整個seq |
| Cost和error往往沒有關係 | 可以很explicitly去考慮label間關係?可以 |
| 可以是deep | Cost function是error的上界 |
| Deep Leaning可以佔到比較大的優勢 |  |

Input feature先通過RNN/LSTM，再餵到HMM, CRF, Structured Perceptron/SVM

語音辨認: CNN/LSTM/DNN +HMM

RNN是一個一個frame去看

語意tagging: Bi-directional LSTM+ CRF/Structured SVM

Structure learning要解三個問題

1.Inference

2.解inference的問題，要窮舉所有的可能

3.How to learn how to learn F\(x\)

把GAN中的discriminator看成evaluation function可以解Problem 1

把generator當成可以窮舉所有的可能

**Conditional GAN**

Discriminator會去Check real \(x,y\) pair

Input x =&gt; image y =&gt; discriminator就會去檢查畫和圖是不是一樣?

learning需要有cost function \(也稱做loss function\)。cost function就是計算entropy。是每個時間點的cross entropy。有了loss function後，便將loss function對w \(weight,RNN的權重參數\)微分。BPTT的演算法來計算這些微分運算。RNN不容易訓練，loss很容易沒有收斂。過去以為是有程式bug。後來發現其實是RNN的error surface很不平滑。就像是懸涯峭壁，懸崖上的的gradient很大，若是踩在懸崖上，調整參數\(w\)之後就飛出去了。

Recurrent neural network training data就是訓練RNN的資料。\(這邊再聽一次便再補進文章\)

\[0\]

H. Y. Lee, ML lecture \#26, RNN part II  
, at [https://www.youtube.com/watch?v=rTqmWlnwz\\_0](https://www.youtube.com/watch?v=rTqmWlnwz_0)

