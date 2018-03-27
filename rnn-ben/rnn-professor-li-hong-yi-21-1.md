# Recurrent Neural Network \(Professor 李宏毅 \#21-1\)

Tsung-Yung 3/20/2018

舉個例子，如果我們要建一個基於語音辨識的售票系統\(ticket booking system\)，它可以輸入一段語音，它可以從裡面挑出抵達日期\(time of arrival\)跟目的地\(Destination\)。

![ticket booking system](/assets/ticket-booking system.png) _圖一 ticket booking system_

如圖一所示，輸入一段文字 "I would like to arrive Taipei on November $$2^{nd}$$."，它能自動擷取目的地是Taipei，抵達日期是November $$2^{nd}$$。這樣的技術又稱作Slot Filling。

那我們要怎麼實作這個系統呢？我們可以使用簡單的Feedforward Network如圖二所示：  
![feedforward network](/assets/feedforward network.png) _Feedforward Neural Network_

把一個一個單字變成word vector輸入網路，在輸出有 $$y_1$$ 代表該單字是Destination的機率， $$y_2$$是抵達時間的機率。常見的word vector產生的方式有1-of-N encoding，但是詞語庫的字彙量一定遠遠小於現實世界的字彙量。為了更好表達這樣的情況，我們在原本N維再增加一維代表沒有在辭彙庫出現的字，另外還有一種方式叫word hashing，就是把一個英文字拆成字母的組合，以3維word hashing，總共會有26x26x26種組合，以單字apple為例，可以拆成a-p-p, p-p-l, p-l-e，代表這三個組合的維度會加1，其餘為0，這樣編碼的好處是可以用有限的維度表現所有的字彙組合。

用Feedforward網路的缺點是，以兩個句子為例 "I would like to arrive Taipei on November $$2^{nd}$$.", "I would like to leave Taipei on November $$2^{nd}$$." 在第二個句子裡面，Taipei並不是目的地，而是出發地。對於Feedforward網路而言，因為它是單個單字判斷，所以只要輸入相同，輸出的值就一定相同。解決的方式是讓神經網路擁有記憶力，才有辦法解決。

Recurrent Network就是讓網路具備記憶的一種方式。


Elman Network vs Jordan Network

Bi-directional RNN

LSTM

More Complex Multi-Layer LSTM



\[0\] http://youtube..

