## Linear SVC Machine learning SVM example with Python

180302 CCL

The objective of a Linear SVC \(Support Vector Classifier\) isto obtain a "best fit" hyperplane to categorizes the data. After getting the hyperplane, the classifier can predict the label \(or class，類別\) based on the input features. A simple supervised learning example working with Linear SVC isdescribedin this article \(Fig. 01\).

![](/assets/procedure_of_conducting_linear_SVC.png)

**Import packages**![](/assets/CODE02import.png)

Numpy is for array conversion.

Maitipzlotlib is for data visualization that can show how linear SVC works.







**Define features**

Considering the data points with \(x,y\) coordinate values of \(1,2\), \(5,8\), \(1.5,1.8\) \(8,8\), \(1,0.6\) and \(9,11\). These data has two dimension feature, i.e., x and y coordinate values.

Therefore, these data serves as two-feature data example, and they can be expressed as.![](/assets/SVM_CODE03.png)

##### Graph the data![](/assets/SVM_CODE04.png)

![](/assets/SVM_FIG02import.png)

F**ig. 02 The distribution of data features**

From Fig. 02, two groups can be easily divided. But, to draw the exact dividing line need further calculation.

**Compile an array**

The array formats are required when running the machine learning algorithm. Thus, the features \(x and y\) are stored in an array of two elements. \(A variable\)

In the supervised learning, data sets are labeled \(or classed\) for training purposes. The labeling rule is: 0 are assigned to lower feature pairs and 1 to the higher ones. \(L variable\)![](/assets/a_is_np.png)**D**

**efine the classifier**

The SVC in SVM \(support vector machine\) module is used. The kernel is defined as linear and C, a valuation value, is defined as 1.0.

The classifier does learning with “clf.fit\(A, L\)”.

Two features derive a 2D graph. Thus, the problem occurs when there are thousands of more features.![](/assets/clf_is_svm.png)



##### Draw the exact dividing line \(Fig. 03\)![](/assets/draw_fig_3.png)

![](/assets/a_is_minus.png)

![](/assets/fig_3_distribution_data_feature.png)

**Fig. 03 The distribution of data features and the dividing line**



The learned classifier can be test as follows:

![](/assets/learnt_svm.png)![](/assets/SVM_FIG04.png)











Fig. 04 The distribution of the test points \(the blue and red points\)



&lt;!--  
 /\* Font Definitions \*/  
 @font-face  
	{font-family:新細明體;  
	panose-1:2 2 5 0 0 0 0 0 0 0;  
	mso-font-alt:PMingLiU;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"Cambria Math";  
	panose-1:2 4 5 3 5 4 6 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-536870145 1107305727 0 0 415 0;}  
@font-face  
	{font-family:Calibri;  
	panose-1:2 15 5 2 2 2 4 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:-536859905 -1073732485 9 0 511 0;}  
@font-face  
	{font-family:微軟正黑體;  
	panose-1:2 11 6 4 3 5 4 4 2 4;  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:679 684672000 22 0 1048585 0;}  
@font-face  
	{font-family:"\@新細明體";  
	panose-1:2 1 6 1 0 1 1 1 1 1;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"\@微軟正黑體";  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:679 684672000 22 0 1048585 0;}  
 /\* Style Definitions \*/  
 p.MsoNormal, li.MsoNormal, div.MsoNormal  
	{mso-style-unhide:no;  
	mso-style-qformat:yes;  
	mso-style-parent:"";  
	margin:0cm;  
	margin-bottom:.0001pt;  
	mso-pagination:none;  
	font-size:12.0pt;  
	mso-bidi-font-size:11.0pt;  
	font-family:"Calibri",sans-serif;  
	mso-ascii-font-family:Calibri;  
	mso-ascii-theme-font:minor-latin;  
	mso-fareast-font-family:新細明體;  
	mso-fareast-theme-font:minor-fareast;  
	mso-hansi-font-family:Calibri;  
	mso-hansi-theme-font:minor-latin;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;  
	mso-font-kerning:1.0pt;}  
.MsoChpDefault  
	{mso-style-type:export-only;  
	mso-default-props:yes;  
	font-family:"Calibri",sans-serif;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;}  
 /\* Page Definitions \*/  
 @page  
	{mso-page-border-surround-header:no;  
	mso-page-border-surround-footer:no;}  
@page WordSection1  
	{size:612.0pt 792.0pt;  
	margin:72.0pt 90.0pt 72.0pt 90.0pt;  
	mso-header-margin:36.0pt;  
	mso-footer-margin:36.0pt;  
	mso-paper-source:0;}  
div.WordSection1  
	{page:WordSection1;}  
--&gt;  


**Fig. 04 The distribution of the test points \(the blue and red points\)**

 Fig. 04 The distribution of the test points \(the blue and red points\)Fig. 04 The distribution of the test points \(the blue and red points\)

&lt;!--  
 /\* Font Definitions \*/  
 @font-face  
	{font-family:新細明體;  
	panose-1:2 2 5 0 0 0 0 0 0 0;  
	mso-font-alt:PMingLiU;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"Cambria Math";  
	panose-1:2 4 5 3 5 4 6 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-536870145 1107305727 0 0 415 0;}  
@font-face  
	{font-family:Calibri;  
	panose-1:2 15 5 2 2 2 4 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:-536859905 -1073732485 9 0 511 0;}  
@font-face  
	{font-family:微軟正黑體;  
	panose-1:2 11 6 4 3 5 4 4 2 4;  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:679 684672000 22 0 1048585 0;}  
@font-face  
	{font-family:"\@新細明體";  
	panose-1:2 1 6 1 0 1 1 1 1 1;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"\@微軟正黑體";  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:679 684672000 22 0 1048585 0;}  
 /\* Style Definitions \*/  
 p.MsoNormal, li.MsoNormal, div.MsoNormal  
	{mso-style-unhide:no;  
	mso-style-qformat:yes;  
	mso-style-parent:"";  
	margin:0cm;  
	margin-bottom:.0001pt;  
	mso-pagination:none;  
	font-size:12.0pt;  
	mso-bidi-font-size:11.0pt;  
	font-family:"Calibri",sans-serif;  
	mso-ascii-font-family:Calibri;  
	mso-ascii-theme-font:minor-latin;  
	mso-fareast-font-family:新細明體;  
	mso-fareast-theme-font:minor-fareast;  
	mso-hansi-font-family:Calibri;  
	mso-hansi-theme-font:minor-latin;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;  
	mso-font-kerning:1.0pt;}  
.MsoChpDefault  
	{mso-style-type:export-only;  
	mso-default-props:yes;  
	font-family:"Calibri",sans-serif;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;}  
 /\* Page Definitions \*/  
 @page  
	{mso-page-border-surround-header:no;  
	mso-page-border-surround-footer:no;}  
@page WordSection1  
	{size:612.0pt 792.0pt;  
	margin:72.0pt 90.0pt 72.0pt 90.0pt;  
	mso-header-margin:36.0pt;  
	mso-footer-margin:36.0pt;  
	mso-paper-source:0;}  
div.WordSection1  
	{page:WordSection1;}  
--&gt;  


**Fig. 04 The distribution of the test points \(the blue and red points\)**



&lt;!--  
 /\* Font Definitions \*/  
 @font-face  
	{font-family:新細明體;  
	panose-1:2 2 5 0 0 0 0 0 0 0;  
	mso-font-alt:PMingLiU;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"Cambria Math";  
	panose-1:2 4 5 3 5 4 6 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-536870145 1107305727 0 0 415 0;}  
@font-face  
	{font-family:Calibri;  
	panose-1:2 15 5 2 2 2 4 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:-536859905 -1073732485 9 0 511 0;}  
@font-face  
	{font-family:微軟正黑體;  
	panose-1:2 11 6 4 3 5 4 4 2 4;  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:679 684672000 22 0 1048585 0;}  
@font-face  
	{font-family:"\@新細明體";  
	panose-1:2 1 6 1 0 1 1 1 1 1;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"\@微軟正黑體";  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:679 684672000 22 0 1048585 0;}  
 /\* Style Definitions \*/  
 p.MsoNormal, li.MsoNormal, div.MsoNormal  
	{mso-style-unhide:no;  
	mso-style-qformat:yes;  
	mso-style-parent:"";  
	margin:0cm;  
	margin-bottom:.0001pt;  
	mso-pagination:none;  
	font-size:12.0pt;  
	mso-bidi-font-size:11.0pt;  
	font-family:"Calibri",sans-serif;  
	mso-ascii-font-family:Calibri;  
	mso-ascii-theme-font:minor-latin;  
	mso-fareast-font-family:新細明體;  
	mso-fareast-theme-font:minor-fareast;  
	mso-hansi-font-family:Calibri;  
	mso-hansi-theme-font:minor-latin;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;  
	mso-font-kerning:1.0pt;}  
.MsoChpDefault  
	{mso-style-type:export-only;  
	mso-default-props:yes;  
	font-family:"Calibri",sans-serif;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;}  
 /\* Page Definitions \*/  
 @page  
	{mso-page-border-surround-header:no;  
	mso-page-border-surround-footer:no;}  
@page WordSection1  
	{size:612.0pt 792.0pt;  
	margin:72.0pt 90.0pt 72.0pt 90.0pt;  
	mso-header-margin:36.0pt;  
	mso-footer-margin:36.0pt;  
	mso-paper-source:0;}  
div.WordSection1  
	{page:WordSection1;}  
--&gt;  


**Fig. 04 The distribution of the test points \(the blue and red points\)**&lt;!--  
 /\* Font Definitions \*/  
 @font-face  
	{font-family:新細明體;  
	panose-1:2 2 5 0 0 0 0 0 0 0;  
	mso-font-alt:PMingLiU;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"Cambria Math";  
	panose-1:2 4 5 3 5 4 6 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-536870145 1107305727 0 0 415 0;}  
@font-face  
	{font-family:Calibri;  
	panose-1:2 15 5 2 2 2 4 3 2 4;  
	mso-font-charset:0;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:-536859905 -1073732485 9 0 511 0;}  
@font-face  
	{font-family:微軟正黑體;  
	panose-1:2 11 6 4 3 5 4 4 2 4;  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:679 684672000 22 0 1048585 0;}  
@font-face  
	{font-family:"\@新細明體";  
	panose-1:2 1 6 1 0 1 1 1 1 1;  
	mso-font-charset:136;  
	mso-generic-font-family:roman;  
	mso-font-pitch:variable;  
	mso-font-signature:-1610611969 684719354 22 0 1048577 0;}  
@font-face  
	{font-family:"\@微軟正黑體";  
	mso-font-charset:136;  
	mso-generic-font-family:swiss;  
	mso-font-pitch:variable;  
	mso-font-signature:679 684672000 22 0 1048585 0;}  
 /\* Style Definitions \*/  
 p.MsoNormal, li.MsoNormal, div.MsoNormal  
	{mso-style-unhide:no;  
	mso-style-qformat:yes;  
	mso-style-parent:"";  
	margin:0cm;  
	margin-bottom:.0001pt;  
	mso-pagination:none;  
	font-size:12.0pt;  
	mso-bidi-font-size:11.0pt;  
	font-family:"Calibri",sans-serif;  
	mso-ascii-font-family:Calibri;  
	mso-ascii-theme-font:minor-latin;  
	mso-fareast-font-family:新細明體;  
	mso-fareast-theme-font:minor-fareast;  
	mso-hansi-font-family:Calibri;  
	mso-hansi-theme-font:minor-latin;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;  
	mso-font-kerning:1.0pt;}  
.MsoChpDefault  
	{mso-style-type:export-only;  
	mso-default-props:yes;  
	font-family:"Calibri",sans-serif;  
	mso-bidi-font-family:"Times New Roman";  
	mso-bidi-theme-font:minor-bidi;}  
 /\* Page Definitions \*/  
 @page  
	{mso-page-border-surround-header:no;  
	mso-page-border-surround-footer:no;}  
@page WordSection1  
	{size:612.0pt 792.0pt;  
	margin:72.0pt 90.0pt 72.0pt 90.0pt;  
	mso-header-margin:36.0pt;  
	mso-footer-margin:36.0pt;  
	mso-paper-source:0;}  
div.WordSection1  
	{page:WordSection1;}  
--&gt;  


**Fig. 04 The distribution of the test points \(the blue and red points\)**



**Environment: KaggleFig. 04 The distribution of the test points \(the blue and red points\)**

**REF:**

\[0\] Main contents

[https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/](https://www.gitbook.com/book/mmchiou/ai_challenge_taiwan_2018-private/edit)

\[1\] Errors: “Expected 2D array got 1D array instead”

[https://stackoverflow.com/questions/45554008/error-in-python-script-expected-2d-array-got-1d-array-instead](https://stackoverflow.com/questions/45554008/error-in-python-script-expected-2d-array-got-1d-array-instead)

