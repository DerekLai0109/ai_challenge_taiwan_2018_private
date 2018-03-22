# structure learning 李宏毅教授

&lt;!--  
 /\* Font Definitions \*/  
 @font-face  
	{font-family:PMingLiU;  
	panose-1:2 2 5 0 0 0 0 0 0 0;  
	mso-font-alt:新細明體;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"Cambria Math";  
	panose-1:2 4 5 3 5 4 6 3 2 4;  
	mso-font-charset:1;  
	mso-generic-font-family:roman;  
	mso-font-format:other;  
	mso-font-pitch:variable;  
	mso-font-signature:0 0 0 0 0 0;}  
@font-face  
	{font-family:Calibri;  
	panose-1:2 15 5 2 2 2 4 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:-520092929 1073786111 9 0 415 0;}  
@font-face  
	{font-family:"\@PMingLiU";  
	panose-1:2 2 5 0 0 0 0 0 0 0;  
	mso-font-alt:"\@Arial Unicode MS";  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"Microsoft JhengHei";  
	panose-1:2 11 6 4 3 5 4 4 2 4;  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:135 680476672 22 0 1048585 0;}  
@font-face  
	{font-family:"\@Microsoft JhengHei";  
	panose-1:2 11 6 4 3 5 4 4 2 4;  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:135 680476672 22 0 1048585 0;}  
 /\* Style Definitions \*/  
 p.MsoNormal, li.MsoNormal, div.MsoNormal  
	{mso-style-unhide:no;  
	mso-style-qformat:yes;  
	mso-style-parent:"";  
	margin-top:0cm;  
	margin-right:0cm;  
	margin-bottom:8.0pt;  
	margin-left:0cm;  
	line-height:107%;  
	mso-pagination:widow-orphan;  
	font-size:11.0pt;  
	font-family:"Calibri",sans-serif;  
	mso-ascii-font-family:Calibri;  
	mso-ascii-theme-font:minor-latin;  
	mso-fareast-font-family:PMingLiU;  
	mso-fareast-theme-font:minor-fareast;  
	mso-hansi-font-family:Calibri;  
	mso-hansi-theme-font:minor-latin;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;}  
a:link, span.MsoHyperlink  
	{mso-style-noshow:yes;  
	mso-style-priority:99;  
	color:blue;  
	text-decoration:underline;  
	text-underline:single;}  
a:visited, span.MsoHyperlinkFollowed  
	{mso-style-noshow:yes;  
	mso-style-priority:99;  
	color:\#954F72;  
	mso-themecolor:followedhyperlink;  
	text-decoration:underline;  
	text-underline:single;}  
.MsoChpDefault  
	{mso-style-type:export-only;  
	mso-default-props:yes;  
	font-family:"Calibri",sans-serif;  
	mso-ascii-font-family:Calibri;  
	mso-ascii-theme-font:minor-latin;  
	mso-fareast-font-family:PMingLiU;  
	mso-fareast-theme-font:minor-fareast;  
	mso-hansi-font-family:Calibri;  
	mso-hansi-theme-font:minor-latin;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;}  
.MsoPapDefault  
	{mso-style-type:export-only;  
	margin-bottom:8.0pt;  
	line-height:107%;}  
@page WordSection1  
	{size:612.0pt 792.0pt;  
	margin:72.0pt 72.0pt 72.0pt 72.0pt;  
	mso-header-margin:35.4pt;  
	mso-footer-margin:35.4pt;  
	mso-paper-source:0;}  
div.WordSection1  
	{page:WordSection1;}  
--&gt;  


什麼是structure learning, 到目前為止，我們考慮的問題，他的input其實都是一個vector，output都是另外一個vector。不管我們是在作SVM，還是在作Deep Learning的實後。我們的input或output都只是vector而以。但實際上我們要面對的問題往往比這個更困難。我們可能需要input或是output事一個sequence，我們可能需要output事一個list。Output 事一個tree。

Output 事一個bounding box等等。像你在representation 的final裡面。你可能希望你的output就直接室一個list，而不是一個一個element。

當然在大原則上我們知道怎麼做。我們就事要找一個function他的input就是我們要的object，

比如說他的input就事一個tree，他的output就是另外一種object。

只是我們不知道要怎麼做。

如果我們目前學過的deep learning的network架構。我們要怎麼都個network，他個inpu才會事一個tree structure。Output 才會是另一個tree structure。你可能不知道要怎麼做這件事。









像這種structure learning的task。他有非常多的應用，他的應用比比接是。我們知道一班的machine learning的課程是不會講structure learning的。其實structure learning他的application非常多。

所以，你不知道structure learning的話，你其實很多時後會非常卡這樣。比如說語音辨識，如果你只知道一班的network，你根本無法想像語音辨識怎麼做的。語音辨識，他是input一個sequence，

Output另外一個sequence。或是你根本無法想像translation是怎麼做的。

Translation是input一個sequence，output另外一個sequence。

你要中翻英，中文事一個sequence，英文是另外一個sequence。

或是你要做一個syntaic 的parsing作文法剖析。Input事一個sentence，output事一個文法頗希的另一個tree structure。

或是你要做一個object detection

，

你個input事一張image output 事一個object的位置，你會把那個object的位置用一個bounding box把他框起來。

那個bounding box是你個output他也試一個object。

或者你要做summary你的input事一個document，你的output是你summarize用的結果。你的input output都是sequence。

或者你要做retrival，你的input是搜尋的關鍵字，你的output是搜尋的結果。搜尋的結果事一個list。他也試一個structure的東西。那structure learning怎麼做呢?

雖然這個structure learning呢聽起來好像很困難。

但實際上呢。他有一個unified的framework。

怎麼做呢。在training的時候。我們就是找一個function。

這個function 我們這邊寫作大寫的F，

這個大寫的F他的input根output。他的input是X根Y。

我們知前是找一個小寫的function f

他的input是x

線在不一樣我們找一個大寫的F 他的input就是 X根Y。

他的output就事一個real number 。

這個大寫F他作的事情。就是橫樑說當我的input是X根Y的時候。Structure的Object，這個X根Y。這個X根Y他們有多匹配。OK 越匹沛的話，大寫F他output的值就越大。那testing的時候呢。

Testing的時候我們要怎麼做呢。給一個新的X，我們去窮舉所有的可能的Y 窮舉所有可能的Y一一代進大寫的F這個function看看哪一個Y他可以讓F的值最大

那假設可以讓F的值最大，可以讓F的值的那個Y教作Y delta。

就是你最後辨識的結果。就是你model的output。

那你會說原來小寫的f呢原來想要做的事情。是找一個小寫的f input x output y 那這個小寫的f input x output y就可以把他想成這個小寫的f 其實就是。

Arg max 窮舉所有的y F\(x,y\)這個東西就是小寫的f 這樣講你可能覺得有點抽像，所以我們來舉個十寄的例子。

假設我們現在要做的task是。我們要給一個image 我們現在的任務是找出image裡面。我們要他找一個object。

舉例來說，所以現在在我們的task裡面。 Input 事一張image。

Output事一個bounding box

舉例來說，線在假設我們的task呢事要做一個量工春日的detection。



Input事一張image。

Output的bounding box 就是量工春日的所在位置。這樣子。

不知道的人講一下，有綁黃絲帶的這個是量工春日這樣。這個東西有很多十寄上的

你會說真測量工春日有什麼用途

沒有什麼用

但是號其他的作用

像偵測人臉阿或是無人駕駛偵測有沒有車子。都是在作bounding box的extraction。

有沒有別的方法。用deep learning的方法來作。

有一個network 教作hybrid CNN 就是幫你找出bounding box這樣子。那，是實上deep learning根structure learning是有關係的

這個我還沒有聽其他太多人講過。

這個是我個人的想法，我認為gain呢他就是structure learning非常有關係的。

Gain就個在十作我們剛剛講的framework的那個方法。

這個我們講完strucrure learning之後在講。

Deep learning根structure learning他們並不是independent的。

他們其十是他們急將要被merge在一起。

但是我還是要舉bounding box的例子。我只是要偵測量工春日的圖這樣子。



而如果是object detection是怎麼做的呢?

你的image呢你個input X就事一張image。Y 就是bounding box。



F\(X,Y\)就是說假設這張image，配上這個紅色的bounding box這個位置。



跟這個紅色的bounding box 他們有多匹配。



如果是在object detection的例子就是他有多正確。



你有沒有真的把量工春日框出來。

所以你會期待說你的model可以作到的事情，你的大寫F，他可以做到的事情是這樣。

給這張圖如果框在這邊。他的分數就很高，因為框的很對。

框在這邊，綠色的框框有點不對，



框在，十九留頭上就不對這樣子。



如果是另外一張圖，框在紅色框框很對。

框在這邊後面這個人我也看不清楚。他到底是誰，喔原來也是不對。框在這邊，這個人到底是誰我們想想看。這個是古全不是阿許，這個才是阿許。

好

那接下來testing的時候給一張圖。這個x這個x是從來沒有看過的圖。那麼，你窮舉所有可能的bounding box這個bounding box這個bounding box可以話在這個地方可以話在這個地方。可以話在這個地方這個地方這個地方，可以話在各個不同可能的地方。看說那個得到的分數最高，可能紅色的得到十分，綠色的得兩分。藍色的三分，綠色的一分等等。然後紅色的最高，紅色就是你model的output。



那在別的task裡面呢，其實，也是差不多的假設我們今天要做sumarization。

Summarizton的task就是，input一個document很長的。他有很多句子，output事一個summary。

你的summary就是從這個document取幾個句子出來，取幾個subset出來。

那我們training的時候呢就是，你的這個f\(x,y\)他的document跟這個summary沛成一對的時候。F的值就很大。Document跟不正確的summary沛成一對的時候F的值就很小。對每一個training data都這麼做。

Testing的時候呢，就是窮舉所有可能的summary，看那一個summary可以讓你的f最大。他就事一個正確，他就是你的model的output。



或這是retrival的時候呢，就是，也是一樣，retrival作的task

Input事一個查詢詞output事一個搜尋的結果。Webpage的list



那麼training的時候呢。我們要一些training的data知道說input這個query output是哪一list才是perfect。



Input Obama的時候output是這個list是perfect。分數最高，

Output是這個list是不對的。所以他的分數比較低。



Input trump的時候，output是這個list是對的。所以分數比較高

Output是這個list是不對的，所以分數比較低等等。



作搜尋的時候有人輸入一個量工春日。就全窮舉所有可能的list看看。那一個list分數最高。你可能覺得什麼窮舉所有的例子聽起來是多麼荒謬喔。

這個都是可以作的

這個都是可以作的。

然後你只要想一個好的演算法去解。

解這個問題。

就找看那一個list他可以讓分數最高

他就的f\(x\)的解

好這個unified的framework或許你聽的很陌生。



覺得很怪這樣子第一次聽到的人可能都覺得你搞什麼東西阿真的。



怎麼出現一個f這樣。那我們換一個說法看你有沒有比較接受。這個說法是這樣的。

我們在training的時候事要estimate X根Y的joint probability。Estimate X根Y一起出現的機率。

這個機率其實也試一個function這個機率的input就事一個X一個Y output就事一個機率。零到伊之間的值。

那我在作testing的時候我在作testing的時候。就是給我一個object X我去計算。

P\(y\|x\)的機率。那麼那一個y積率最高，那麼他就是我的答案。那p\(y\|x\)的機率可以寫成P\(x,y\)/P\(x\)

P\(x\)對於你最後找出來的y沒有影響

那麼就是找P\(x,y\)在哪一個y的probability最高，那個y就是最後你的output。



而這個training就是這個training

這個inference就是這個inference

我們剛剛講的F\(X,Y\)你可能會覺得說evaluate F\(X,Y\)會有多相容這個道理是在講什麼。不太懂。

如果我把他換成我是要evaluate X根Y的joint Probability. X根Y一起出現的機率。然後在testing的時候，根據這個機率，我要找最有可能的Y，這樣你會不會覺得比較能夠接受一點呢?這樣會不會覺得能夠接受一點呢?

我們來做一下民意調查，你比較喜歡。你覺得直接用F\(X,Y\)必較容易理解的你舉手一下。沒有?有舉手一下請手放下。如果你覺得說這個機率P\(X,Y\)你比較容易理解的你舉手一下。手放下，稍微多一點。那其實這兩個東西都是可以的。如果你今天是在讀grafical model的文獻的話，假設，只是graphical model可能自己看都是看得一頭霧水。

其實，graphical model就是structure learning的其中一種。所以，你可以之後把我的structure learning的可聽完，你可以MAP到graphical model的部分，你可以發現其實我講的其實就是其實graphical model就是一種structured learning。只是在graphical model的時候我們把F\(X,Y\)換成機率。那其實講的是一樣的事情。那個什麼believed network Michel randon field啦。他們講的其實是一樣的事情。

他們都是去找一個evaluation的function，只是他們找的evaluation的function是一個機率

那用機率有什麼壞處呢?我個人覺得，我比較喜歡用F\(X,Y\)勝過機率。

機率的壞處就是有時後東西說機率很怪。

你說，我們做搜尋，那X是ㄧ個查詢詞，那Y是ㄧ個搜尋的結果。那個要衡量這個查詢詞跟這個搜尋結果共同出現的機率。我覺得很怪，有時後不太能夠接受。

那麼再來機率會有constraint就是summation is 1

那麼你現在是ㄧ個有structure的東西X Y都是一個很大的space。很大的space這個要怎麼做summaiton阿很難。所以就像用石頭砸自己的腳這樣子。你把機率的東西引進來，然後要normalize變成機率，然後結果你會發現大部分的時間你在想說要怎麼把他做normalization。那麼何不想說，不要去做normalization呢?



那作機率有一個好處啦

就是機率是meaningful的。你比較容易了解想像他是什麼樣的一個東西這樣。

那麼其實還有另外一個東西叫做energy model你可能有聽過。

這個energy model，這個是央朗課題出來的我在下面有附上央朗課的energy model的說明給大家參考。其實，energy model講的也是structure learning，那個其實在差不多的時間，在世界上有很多人在差不多的時間點都提出的很多類似的framework，那麼合起來就是我們這邊這個unified的structure learning的framework

他們講的其實是一樣的東西，什麼graphical model阿structured learning structured SDM energy model他們的framework都是一樣。就好像同樣的東西，在獵人裡面叫作獵人在海賊王裡面叫做霸氣，在火影裡面叫做查克拉，他們其實都是一樣的東西這樣子。

那麼這個framework聽起來好像很厲害。

那其實要做這個framework其實你要解三個問題。

我知道快下課了，所以我就很快帶過這三個問題。



第一個問題是F\(X,Y\)長什麼樣子，你很難想像F\(X,Y\)應該是長什麼樣子，現在input是，input是ㄧ個image，

Input是ㄧ個image加上一個bounding box，這個F\(X,Y\)應該是長什麼樣子



Input是ㄧ個keyword根list這個F\(X,Y\)應該長什麼樣子。



再來就是那個荒唐的inference的問題，怎麼解arg max的這個問題

這個Y阿可是很大的。

比如說，你要解object detection你就窮舉所有的可能的bounding box，這件事情做的到嗎?

第三個問題是training，training的時候的priciple又是我們正確的X根Y的pair可以大過其他的正確XY pair可以大過其他的這個training是可以完成的嗎?指要你，指要你解出這三個問題，你就可以解出structure learning的problem，或著三張神之卡就可以成為法老王。ㄟ地震警報，現在有地震嗎?沒有喔…好那其實是這樣子啦。

我覺得gain其實就可以解出這三個問題的solution，你可能看不出來gain跟這個問題有什麼關係，他們是有關係的。Gain好像就是我看到就是解這三個問題的曙光這樣子



好，那其實這三個問題，你在別的地方是有聽過的喔你如果有修過數位影音處理的時候，里寧三老師就有說過他也說過HAN有三個問題，其實這三個問題general的structure learning的三個問題這樣子。他不只用在HAN上，他可以用在任何的strucrue learning的problem上，

是實上這個東西，我們可以把他跟DNN Link在一起。我們之前講的feedforward network你可能覺得跟現在講的strucrured learning沒有關係，是有關係的。之前講的case其實就是structure learning的一個special case。

現在說我要做一個手寫數值辨識

Input一個image把他分成十類，我們在training的時候F長什麼樣子呢

F長這樣子。F是ㄧ個DNN經過DNN

我在input Y這個y是ㄧ個Y是ㄧ個十維的vector他只有一維是1其他是0

他分別代表十個不同的數字。

把這個y呢根N\(X\)算cross entrophy

Negtive的cross entrophy就是F\(X,Y\)

那你的大F就是input X根Y output就是這個值。

然後呢接下來在testing的時候

在inference的時候，就是說我現在要做手寫數字辨識

我窮舉十個所以可能辨識的結果，其實說窮舉就是十個可能辨識的結果。每一個都帶進去這個function裡面，看那一個辨識節果可以讓F\(X,Y\)最大。那個Y就是我們的辨識結果。

好吧，那一個F\(X,Y\)可以讓我們的辨識結果最大呢?其實呢，如果你是用cross entrophy來定義這兩個vector之間的差距的話。你就看說現在哪個digit他對應的dimention他的值最大。他就是那個辨識的結果，這就是跟我們之前在train neural network用cross entrophy作loss function的時候做的事情其實是一模一樣的。就我們之前講的東西是strucrure learning的一個special case

你可以定出事實上我們做的是f\(x\) output Y其實那個問題我們也可以想成我們找一個大F，input X Y output一個number去evaluate X Y有多compatible。

有多相容，這個arg max這個問題，因為在trasfication裡面我們的y太少了，才看了幾個case就有幾個Y有可能窮舉的

那找max其實就是那個窮舉的那個行，可你輕易做到。

  
















Input is a vector

Output is a vector

Deep Learning,

Output: List, tree, bonding box



Structure learning:

Speech recognition input sequence output sequence

Translation:



Unified Framework- Object Detection



figure, ...



Enumerate all possible bounding box y



Unified framework

Training is to estimate the joint probability of X and Y

Find the y that makes the joint probability of X and Y be maximum

[https://www.youtube.com/watch?v=5OYu0vxXEv8](https://www.youtube.com/watch?v=5OYu0vxXEv8)





