# Pytorch Seq2Seq

MM 03/01/2018

這篇想說明如何使用pytorch框架來寫出一個 seq2seq 的 model。

程式放在

[https://github.com/ywk991112/pytorch-chatbot](https://github.com/ywk991112/pytorch-chatbot)

## ![](/assets/pytorch_seq2seq.png)

## 流程

![](/assets/pytorch_seq2seq_flow.png)

使用深度學習的框架去訓練\(train\)一個model時，通常有三個主要步驟：處理資料Processing、訓練Training、測試Testing

### 1 處理資料：

處理資料包含預處理preprocessing與特徵抽取feature extraction。

preprocessing 是資料預處理，目的是去除除雜訊、或是不適合拿來train的資料。

feature extraction是將各種資料\(文字、圖片\)轉換成張量，此張量擁有原資料的feature。

### 2 訓練\(training\)：

訓練包含三個元素：module模組、Graph、gradient decent。

模組：就seq2seq model為例，module分成encoder、decoder部分。

Graph：pytorch會動態建立graph以計算gradient

### 3 Gradient decent：

模模式的參數是利用gradient decent的方式來更新。

![](/assets/weight_update.png)

Pytorc的troch.optim提供不同演算法的optimizer來做gradient decent，包括SGD、Adam、RMSProp。

## 程式解說：

### 1 module \(model.py\)

seq2seq的model可以拆成兩個module，encoder和decoder。

EncoderRNN是encoder module；

LuongAttnDecoderRNN 是attention mechanism的decoder module。

張量維度的轉換是實現encoder module與decoder module最重要的地方。

若能理解model的架構，可以較直觀的實現model。以下說明三個model的重點：embedding layer、packed sequence、requires\_grad。

#### Embedding layer

Embeding layer是一個lookup table。當輸入字的索引，embedding layer會輸出對應的word vector。word vectors存在於weight 變數variable中，當requires\_grad=True時，word vector會跟著被訓練train的。

如果要將pretrain的word vector放入embedding layer，

![](/assets/pretrained_WV_embeding.png)

#### Packed Sequence

在Recurrent neural network裡，每筆資料的input和output的長度有所不同，無法用batch的方式來訓練，PackedSequence是pytorch的一個class，用來解決這個問題。但是不能直接宣告一個PackedSequence物件，而是用torch.nn.utils.rnn.pack\_padded\_sequence將variable轉換成PackedSequence，如果要再轉換回variable，要用torch.nn.utils.rnn.pad\_packed\_sequence函式。

且input的長度需由長排到短，所以在load的時候，training data需要依照長度排序。

#### requires\_grad

requires\_grad預設為true，i.e.requires\_grad=True。

### 2 訓練 \(train.py\)

訓練包含三個主要函式：batch2TrainData、train、trainEpochs

#### batch2TrainData將load.py整理好的training pairs轉換成input、output variable

![](/assets/batch2_traindata.png)

#### train將input data 順向計算出函數值，再利用back propagation算出model 參數的gradient，並使用optimizer去更新model的參數。![](/assets/train.png)

#### trainEpochs準備好training時所需要的module和optimizer，並重複把training batch餵進model裡做參數的更新

#### ![](/assets/trainEpoch.png)

#### ![](/assets/trainEpoch.png)

\[0\] [https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305df38a7c015e194f22f8015b](https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305df38a7c015e194f22f8015b)

