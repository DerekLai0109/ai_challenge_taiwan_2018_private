# RNN

```asciidoc
[#img-sunset] 
.A mountain sunset 
[link=https://www.flickr.com/photos/javh/5448336655] 
image::sunset.jpg[Sunset,300,200]
```


接下來討論case2如圖十二，當C對z’與C對z”該層的neural還不是最後一層，此時需要再往下一層反推上來，即利用如上述的反向公式反推回來，如果下層還不是最後一層就一直往下層找到最後一層反推上來。

而我們可利用Backpropagation來讓訓練Neural Network變得更有效率。而Backpropagation就是一種計算Gradient Descent會更有效率的演算法。Backpropagation主要的數學是Chain rule，如下圖二所示: