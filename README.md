# ImageAnalysis_IrisRecognition-JianaFeng

This is to implement project based on LiMa essay. Iris Recognition is useful for person identification. So it is important to have a deep study on Iris image. 


Based on the LiMa epaper, we implement the project about Iris Recognition. For the identification mode, we got pretty good CRR for all three measures(all higher than 85%). And for the verification mode, we got the FMR/FNMR table and the ROC curve. 

Tables are below following: 

(1):CRR table

|Similartiy measure|Reduced feature set(%)|
|--|--|
|L1 distance measure|85.41|
|L2 distance measure|86.80|
|Cosine similarity measure|89.12|


(2):FMR/FNMR table

 False Match and False Nonmatch Rates with Different Threshold Values
 
 |Threshold|False match rate|False non-match rate|
 |--|--|--|
 |0.446|0.0022387 |0.165295|
 |0.472|0.00122111|0.184757|
 |0.502|0.0007043|0.219736|



Explain the whole logic of design(The whole procedure)
------

1.1 ‘IrisLocalization.py’

(a):First, we try to estimate the rough position of the center of pupil by means of summing the projection of a subimage(120*120) and find the minimum
(b):Then,take rough estimate from step1 to be the center, make a new subimage(120*120).Do the same proccessing as step1 and find the new estimate of center of pupil and also estimate the radius of pupil by means of  area calculation
(c):Finally, take the position of pupil center from step2 to get two different size subimage(120*120,230*270).Then using HoughCircle to find the circles of pupil and iris


1.2 IrisNormalization.py

(a): Based on the unwrapping formula from LiMa essay, we use the inner pupil and outter circle to  unwrap the iris ring and get the rectangle block with a fixed size. This is called normalized iris procedure. 
(b): Then we rotate the normalized iris image for different degrees

1.3 ImageEnhancement.py

(a):Take the normalized image and take the mean of subregion(16*16) and do interpolation to estimate the background
(b):Subtract background from normalized image
(c): For a subregion(32*32), perform histogram equalization
(d): Since we found that ImageEnhancement would do harm to our performance,we just comment out this part 

1.4 FeatureExtraction.py

(a):Since the upper portion of a normalized iris image can provides the most useful texture information for iris recognition, we extract features near pupils. So we use ROI function to get the region of interest. 
(b):We create helper functions M1, G1 and kernal for filtering images
(c): We used the two channels values provided by essay to get the filterd images and then computed the means and standard deviations to get the feature vectors  

1.5 IrisMatching.py

(a): We use the Fisher Linear discriminant(LDA) to reduce dimension on feature vectors and then use the NearestCentroid function to train the model and get the scores 
(b)We collected CRR on three measures and CRR for different dimensions 


1.6 PerformanceEvaluation.py

(a): After collecting CRR data from IrisMatching.py, We created the CRR table and figure for CRR on different dimensions. 
(b): We use test data to compute the cosine similarity and calculate the fp,fn,tp,tn, then return the FMR(false match rate) and FNMR(false non-match rate) 
(c): After collecting FMR and FNMR data on different thresholds, we create FMR and FNMR table and ROC curve 

1.7 IrisRecognition.py

(a):First we implement all the preprocessing on image and also take the best rotation degree from experiment
(b):Impement preprocessing on all images
(c):Save the preprocessed dataframe
(d):Train the model and ouput the results 

Limitation and improvement 
-------

2.1 The highest CRR of our project is 89.12%, which is less than 90%. By adjusting the Irislocalization and IrisNOrmalization, we may get the higher CRR. Or we can try to take a different dimension on feature vector for better performance.
2.2 The method we use is out of date.Nowadays, most iris recognition problems could be soved by DL,especially CNN.Like EfficientNet from Google could achieve much higher CRR in much larger dataset. 


---------------------------------------------------------



