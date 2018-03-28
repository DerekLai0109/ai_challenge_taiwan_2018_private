# Recurrent Neural Network \(Professor 李宏毅 \#21-1\)

Tsung-Yung 3/20/2018

舉個例子，如果我們要建一個基於語音辨識的售票系統\(ticket booking system\)，它可以輸入一段語音，它可以從裡面挑出抵達日期\(time of arrival\)跟目的地\(Destination\)。

![ticket booking system](/assets/ticket-booking system.png) 
*圖一 ticket booking system*

如圖一所示，輸入一段文字 "I would like to arrive Taipei on November $$2^{nd}$$."，它能自動擷取目的地是Taipei，抵達日期是November $$2^{nd}$$。這樣的技術又稱作Slot Filling。

那我們要怎麼實作這個系統呢？我們可以使用簡單的Feedforward Network如圖二所示：  
![feedforward network](/assets/feedforward network.png)
*圖二 Feedforward Neural Network*

把一個一個單字變成word vector輸入網路，在輸出有 $$y_1$$ 代表該單字是Destination的機率， $$y_2$$是抵達時間的機率。常見的word vector產生的方式有1-of-N encoding，但是詞語庫的字彙量一定遠遠小於現實世界的字彙量。為了更好表達這樣的情況，我們在原本N維再增加一維代表沒有在辭彙庫出現的字，另外還有一種方式叫word hashing，就是把一個英文字拆成字母的組合，以3維word hashing，總共會有26x26x26種組合，以單字apple為例，可以拆成a-p-p, p-p-l, p-l-e，代表這三個組合的維度會加1，其餘為0，這樣編碼的好處是可以用有限的維度表現所有的字彙組合。

用Feedforward網路的缺點是，以兩個句子為例 "I would like to arrive Taipei on November $$2^{nd}$$.", "I would like to leave Taipei on November $$2^{nd}$$." 在第二個句子裡面，Taipei並不是目的地，而是出發地。對於Feedforward網路而言，因為它是單個單字判斷，所以只要輸入相同，輸出的值就一定相同。解決的方式是讓神經網路擁有記憶力，才有辦法解決。

Recurrent Network就是讓網路具備記憶的一種方式。我們以ticket booking system為例，如圖三所示：
![](/assets/rnn_seq2seq.png)
*圖三 ticket booking system using RNN*
我們在不同的時間把單字依序輸入網路，輸出還是跟Feedforward network一樣是每個單字對目的地跟抵達日期的機率，不同的地方是
RNN會把隱藏層(hidden state)儲存起來，跟下個時間點單字同時輸入網路。圖三裡面畫法不是代表有三個獨立的網路，而是同一個RNN在不同時間點被重複使用。因為RNN有記憶功能，所以如圖四所示，對於我們之前舉的例子而言，雖然都是輸入"Taipei"，但是因為網路還有之前輸入是"leave"還是"arrive"，所以對於Probability of "Taipei"會不同。

![](/assets/rnn-ticket-slot-recognition.png)
*圖四 How RNN solve Ticket Slot Problem*

在RNN，我們常見的網路又稱作Elman Network(圖五)，另外還有另外一種Jordan Network，兩者的差異是Jordan Network儲存的不是隱藏層的資料，而是輸出層。根據李宏毅的survey，聽說Jordan network的performance會比Elman Network好一些，原因是Jordan Network直接是從輸出結果來直接影響網路，而不是Elman Network是比較間接。

Q by TY: 如果Jordan Network performance比較好，為什麼現在決大部分的RNN都是用Elman Network?

![](/assets/elman-jordan-network.png)
*圖五 Elman Network vs Jordan Network*

除了單個時間由前到後的RNN之外，我們還可以把兩個RNN如圖六所示並聯起來，一個是正常時間方向，一個是反著時間方向，用順跟逆時間的資訊來預測每個時間點的輸出，這樣的網路叫做Bidirectional RNN，這樣架構的好處，可以讓每個時間的預測是參照全文的內容，而不是原本RNN，只看原本輸入之前的時間點。
![](/assets/bi-direction-rnn.png)
*圖六 Bi-directional RNN*

###LSTM
![](/assets/lstm-overview.png)
*圖七 Block View of LSTM*

![](/assets/lstm-details.png)
*Mathimatical View of LSTM*

#### Illustrative Example of LSTM
![](/assets/lstm-example1.png)
*Example Sequence*

![](/assets/lstm-example2.png)
*Time 1*

![](/assets/lstm-example3.png)
*Time 2*

![](/assets/lstm-example4.png)
*Time 3*

![](/assets/lstm-example5.png)
*Time 4*

![](/assets/lstm-example6.png)
*Time 5*

#### simple RNN vs LSTM
![](/assets/simple-neuron.png)
*simple rnn*

![](/assets/lstm-neuron.png)
*lstm neuron*

### Vector View of LSTM
![](/assets/lstm-vector.png)
*Vector*

![](/assets/lstm-reform.png)
*Reformulated LSTM*

![](/assets/lstm-reform-seq2seq.png)
*Seq2Seq LSTM*

![](/assets/lstm-reform-recurrent.png)
*Real LSTM*

![](/assets/lstm-reform-recurrent-peephole.png)
*peephole LSTM*

![](/assets/lstm-reform-mutlilayer.png)
*Multi-Layer LSTM*



[0] [ML Lecture 21-1: Recurrent Neural Network (Part I)](https://www.youtube.com/watch?v=xCGidAeyS4M)

