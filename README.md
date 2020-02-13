# Logistic Regression with example : Diabetes Detection

Logistic Regression is a Machine Learning algorithm which is used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability.


<img src="/flask_integration/static/images/pass_fail.png" alt="" width="250"/>

As shown in the picture if you are trying to figure out the test will pass or fail; that is known as Logistic regression. Lets see some real world example
 <ul>
      <li>To predict whether an email is spam (1) or (0)</li>
      <li>Whether the tumour is malignant (1) or not (0)</li>
      <li>If you are diabetic  (1) or not (0)</li>
 </ul>


Consider a scenario where we need to classify whether an email is spam or not. If we use linear regression for this problem, there is a need for setting up a threshold based on which classification can be done. Say if the actual class is malignant, predicted continuous value 0.4 and the threshold value is 0.5, the data point will be classified as not malignant which can lead to serious consequence in real time.
From this example, it can be inferred that linear regression is not suitable for classification problem. Linear regression is unbounded, and this brings logistic regression into picture. Their value strictly ranges from 0 to 1.

# Project:
To make the learning project easier and productive, we are going to learn Logistic Regression by an example of creating a logistic regression model to predict a user is Diabetic or not.
First we are going to collect the dataset and clean it. Then we will create a Logistic regression model with fitting the dataset. 

**FileName: diabetic_analysis_logistic_regression.ipynb**

Finally after a successive training it’s our main goal to integrate our machine learning model with the GUIs. You can use tkinter for making a desktop application like we did in Linear Regression example, but here we are going to integrate it with a web framework i.e Flask, to create a website that will take the input parameters from the user and predict if they are Diabetic or not.

**Command to run the script: python flask_integration\app.py**


<img src="/flask_integration/static/images/diabetics diagnosis.PNG" alt=""/>


Hope you have enjoyed learning this, if so share this with others and for more such contents you can connect with me on 

YouTube: https://www.youtube.com/channel/UCmF8qppe02J1ot4Jfwl_lFg

LinkedIn: https://www.linkedin.com/in/jagwithyou/

Medium: https://medium.com/@jagwithyou


