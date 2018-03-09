## Linear SVC Machine learning SVM example with Python

180302 CCL

The objective of a Linear SVC \(Support Vector Classifier\) isto obtain a "best fit" hyperplane to categorizes the data. After getting the hyperplane, the classifier can predict the label \(or class，類別\) based on the input features. A simple supervised learning example working with Linear SVC isdescribedin this article \(Fig. 01\).

![](/assets/procedure_of_conducting_linear_SVC.png)

**Import packages**![](/assets/SVM_CODE_01.png)  
Numpy is for array conversion.

Maitipzlotlib is for data visualization that can show how linear SVC works.



**Define features**

Considering the data points with \(x,y\) coordinate values of \(1,2\), \(5,8\), \(1.5,1.8\) \(8,8\), \(1,0.6\) and \(9,11\). These data has two dimension feature, i.e., x and y coordinate values.

Therefore, these data serves as two-feature data example, and they can be expressed as.![](/assets/SVM_CODE_02.png)

##### Graph the data.

![](/assets/distribution_data_fig2.png)F**ig. 02 The distribution of data features**

From Fig. 02, two groups can be easily divided. But, to draw the exact dividing line need further calculation.

**Compile an array**

The array formats are required when running the machine learning algorithm. Thus, the features \(x and y\) are stored in an array of two elements. \(A variable\)

In the supervised learning, data sets are labeled \(or classed\) for training purposes. The labeling rule is: 0 are assigned to lower feature pairs and 1 to the higher ones. \(L variable\)





.![](/assets/SVM_CODE_04.png)

Two features derive a 2D graph. Thus, the problem occurs when there are thousands of more features.

**\*\* 再把CODE&RESULTS貼上來!**

**Environment: Anoconda**

**REF**

[https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/](https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/)

