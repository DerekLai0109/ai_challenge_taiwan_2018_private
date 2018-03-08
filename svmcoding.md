## Linear SVC Machine learning SVM example with Python

180302 CCL

The objective of a Linear SVC \(support Vector Classifier\) is to obtain a best fit hyperplane \(classifier\) to categorize the data.

After getting the hyperplane, the classifier can predict the class \(類別\) based on the input features.

A simple supervised learning example working with Linear SVC is descirbed in this article.

![](/assets/SVM_Procedure.jpg)

**Import packages**![](/assets/SVM_CODE_01.png)Maitipzlotlib is for data visualization that can show how linear SVC works. And Numpy is for array conversion.

**Define features**

These features will be visualized as axis on the graph.![](/assets/SVM_CODE_02.png)Graph the data.![](/assets/SVM_CODE_03.png)Result 01 \(ADD FIG\)

**FIG!!  **

**Compile an array**

Two groups can be divided without further calculation. But, to draw the exact dividing line need further calculation.

To feed data into the machine learning algorithm, compiling an array of the features is required, rather than having them as x and y coordinate values.

The features \(x and y\) is stored in X variable.![](/assets/SVM_CODE_04.png)

Two features derive a 2D graph. Thus, the problem occurs when there are thousands of more features.

**\*\* 再把CODE&RESULTS貼上來!**

**Environment: Anoconda**

**REF**

[https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/](https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/)

