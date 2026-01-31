created the repo for the fraud model 


Diar:
working on ai data validation layer
to address things like NaN, missing values

General Approach and Instinctual Thoughts:
have a data validation layer to filter the data which is known to be imperfect

use geocomply as a model to base our ideas from
use the logic from leetcode where certain things like if 
sally's amount withdrawn >1000 || sally withdraws from city a and city b within 60mins then flag it

things to consider may be its all about the quality of the data
what we do with things like false flags can be another thing to go above and beyond
ie say our model misses, do we just take it or learn from it and do sm with it

one thing is how good our model is but how can we squeeze out everything we can from the data

is it better to have a false flag or is it better to have a miss

what is a kind of todo list of things to raelly go above and neyond:
ai data validation, 
kfold cv?
make a model
make it be reinforced and learn from mistakes 
potentially give feedback based on results on anticipated or likely shortcomings and improvements








PROBLEM DESCRIPTION:
Overview
Welcome to HackML 2026! We are pleased to have you all attend SFU's first Kaggle-style ML competition, and we look forward to the rest of the day.


Description
Fraud Detection: Multi-Class Classification Challenge
Financial institutions process millions of transactions every day, yet only a small fraction are fraudulent. In practice, fraud teams do not simply ask “Is this fraud?” — instead, they must decide how urgently a transaction should be investigated given limited resources.

In this competition, you will build a multi-class classification model to predict the urgency level of investigating a transaction, ranging from no action required to immediate intervention. The dataset reflects realistic transaction patterns and extreme class imbalance commonly found in financial systems.

Objective
Your goal is to predict the fraud category of each transaction based on historical transaction features.

This is a supervised multi-class classification problem.

Label	Description	Business Context
0	No Action	Transaction appears legitimate
1	Monitor	Low-risk suspicious activity
2	Review	Likely fraud requiring analyst review
3	Immediate Action	High-risk fraud requiring urgent response
The dataset consists of anonymized transaction-level data collected from a simulated payment system.

Each row represents one transaction.

Features include:

Transaction details (amount, time, frequency)
User behavior indicators
Merchant characteristics
Device and channel information
Target variable:
urgency_level — integer class label (0–3)

The dataset is intentionally imbalanced, reflecting real-world fraud distributions.

Evaluation
Primary Metric: Macro F1-score
Submissions are evaluated using Macro F1-score, which computes the F1-score independently for each class and then averages them.

This metric:

Treats all urgency levels equally
Prevents models from ignoring rare but critical fraud cases
Reflects real-world fraud prioritization needs
Higher Macro F1-scores indicate better performance.
Submission File
For each ID in the test set, you must predict a class for the TARGET variable. The file should contain a header and have the following format:

id,urgency_level
1,0
2,0
3,0
etc.
Dataset Structure
Each row represents a single transaction. The dataset contains the following columns:

step
A unit of time in the simulation.
One step corresponds to one hour since the start of the dataset.

type
The type of transaction. Possible values include:
CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER.

amount
The transaction amount in local currency.

oldbalanceOrg
Balance of the origin account before the transaction.

newbalanceOrig
Balance of the origin account after the transaction.

oldbalanceDest
Balance of the destination account before the transaction.
Not applicable for merchant accounts.

newbalanceDest
Balance of the destination account after the transaction.
Not applicable for merchant accounts.

nameOrig
Random generated name origin

nameDest
Random generated name destination

urgency_level (target variable)
A derived categorical label indicating the recommended level of investigation urgency for a transaction.

Citation
daniel06smith and StanleyS https://www.kaggle.com/code/adityashakya2454/fraud-detection/notebook. FRAUD | HackML 2026. https://kaggle.com/competitions/fraud-hack-ml-2026, 2026. Kaggle.










ORIGINAL PROBLEM DESCRIPTION (Comment on problem recommended this as a potential improvement, maybe we should try both: why u use the label encoding?
indted of these u have to use the one hot encoding as they belongs to nominal category.)

Financial Fraud Detection
Introduction
Financial Fraud Detection is the process of monitoring transactions and customer behavior to identify and stop fraudulent activity. According to Juniper Research's 2022 study about online payment fraud, globally payment fraud are noted to exceed $343 billion between 2023 and 2027.
Traditionally, firms have used fraud detection and prevention to curb company financial losses and maintain positive customer relations.
Common Types of Fraud:
There are different type of fraut and constanlty emerging. Some fraud typologies persist because they exploit weaknesses in a company's processes and systems. Here are some common type of frauds.

Payement Fraud: Happens when a criminal acquires another individual's payement iformation and makes unauthorized transactions.
Return Fraud: takes advantage of a retailer's return policy to receive refunts that aren't legitimate. Fraudulent returns may consist of stolen goods, conterfit products, old and worn-out goods, or items bought from a different retailer.
ACH Fraud: Automated Clearing House is a means of transferring money between bank accounts, usually those of businesses and institutions. ACH is carried out using a bank account number and bank routing number.
Chargeback fraud: contains an individual requesting chargebacks for transactions that were fulfilled by the company they purchased from.
Account takeover fraud (ATO): happens when a criminal acquires the authenication of an account, such as bank account, online payment service, mobile account, or e-commerce site.
Methods to detect fraud:
To protect businesses and counsumers from evolving fraud risks, employing the most effective fraud detection techniques is very important. There are following techniques that can be used to detect frauds.

Machine Learning and AI: Mahcine learning algorithm and AI is enhancing fraud detection capabilities. These techniques analyze large amount of data in real-time, identify patterns and anomalies that might show fraudulent activities.
Behavioral Analytics: By analysing users behaviors, businesses can detect deviations from normal patterns.
Anomaly Detection: Anomaly Detection helps in creating a baseline of normal behavior and flagging any data points that deviate significantly from it.
Identity clustering: Making a group of user identities based on common attributes and behaviors helps in identify patterns of fraudulent behavior.
Data analytics: Advanced data analytics tools can shift thgough a large datasets and identify potential fraud indicators.
Main Challenges of Fraud Detection
Management: Relying soley on rule-based transaction monitoring and fraud detection can be a challenge as sca, techniques change.
Remote transactions: While this is convenient and cost-effective, it also opens the door for fraudsters to impersonate genuine customers or intercept their details.
Speed of transactions: Now-a-days transaction ecosystem is built for speed and convenience. This high-speed, low-friction environement can make it easy for fraudsters to complete their crimes and disappear before they can be detected.
False positives: A fraud detection system that is over-zealous can lead to higher false positives. This is inconvenient for customers, who may become less loyal as a result, and expensive for businesses, who must expend time and resources following up the alert.
Range of transaction types: Large number of tools and services are used to make transactions such as payment apps and cryptocurrency, loans, credit cards and savings accounts. # About the dataset
The dataset is a synthetic representation of mobile money transactions, usually to carried out real-world financial activities while integrating fraudulent behaviors for research purposes.
The dataset encompasses a variety of transaction types including CASH-IN, CASH-OUT, DEBIT, PAYEMENT, and TRANSFER over a simulated period of 30 days.
Importing Libraries
# importing the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
Reading the data
# reading the dataset
df = pd.read_csv("/kaggle/input/financial-fraud-detection-dataset/Synthetic_Financial_datasets_log.csv")
Printing the data
# printing the data
df.head()
step	type	amount	nameOrig	oldbalanceOrg	newbalanceOrig	nameDest	oldbalanceDest	newbalanceDest	isFraud	isFlaggedFraud
0	1	PAYMENT	9839.64	C1231006815	170136.0	160296.36	M1979787155	0.0	0.0	0	0
1	1	PAYMENT	1864.28	C1666544295	21249.0	19384.72	M2044282225	0.0	0.0	0	0
2	1	TRANSFER	181.00	C1305486145	181.0	0.00	C553264065	0.0	0.0	1	0
3	1	CASH_OUT	181.00	C840083671	181.0	0.00	C38997010	21182.0	0.0	1	0
4	1	PAYMENT	11668.14	C2048537720	41554.0	29885.86	M1230701703	0.0	0.0	0	0
Statistical analysis
# print the shape of the data
df.shape
(6362620, 11)
There are 6362620 rows and 11 columns in the data.
# printing the information of the data
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6362620 entries, 0 to 6362619
Data columns (total 11 columns):
 #   Column          Dtype  
---  ------          -----  
 0   step            int64  
 1   type            object 
 2   amount          float64
 3   nameOrig        object 
 4   oldbalanceOrg   float64
 5   newbalanceOrig  float64
 6   nameDest        object 
 7   oldbalanceDest  float64
 8   newbalanceDest  float64
 9   isFraud         int64  
 10  isFlaggedFraud  int64  
dtypes: float64(5), int64(3), object(3)
memory usage: 534.0+ MB
About the dataset The dataset is a synthetic representation of mobile money transactions, usually to carried out real-world financial activities while integrating fraudulent behaviors for research purposes. The dataset encompasses a variety of transaction types including CASH-IN, CASH-OUT, DEBIT, PAYEMENT, and TRANSFER over a simulated period of 30 days.- The dataset contains 6,362,620 entries.

It has 11 columns.
The columns include:
step: An integer representing the time step of the transaction.
type: Categorical variable indicating the type of transaction.
amount: Float value representing the amount of the transaction.
nameOrig: Object type representing the name of the origin account.
oldbalanceOrg: Float value indicating the old balance of the origin account before the transaction.
newbalanceOrig: Float value indicating the new balance of the origin account after the transaction.
nameDest: Object type representing the name of the destination account.
oldbalanceDest: Float value indicating the old balance of the destination account before the transaction.
newbalanceDest: Float value indicating the new balance of the destination account after the transaction.
isFraud: Binary integer indicating whether the transaction is fraudulent (1) or not (0).
isFlaggedFraud: Binary integer indicating whether the transaction was flagged as fraudulent (1) or not (0). The data types are as follows:
5 columns are of type float64.
3 columns are of type int64.
3 columns are of type object.
The memory usage of the DataFrame is approximately 534.0+ MB.

# describing the data
df.describe()
step	amount	oldbalanceOrg	newbalanceOrig	oldbalanceDest	newbalanceDest	isFraud	isFlaggedFraud
count	6.362620e+06	6.362620e+06	6.362620e+06	6.362620e+06	6.362620e+06	6.362620e+06	6.362620e+06	6.362620e+06
mean	2.433972e+02	1.798619e+05	8.338831e+05	8.551137e+05	1.100702e+06	1.224996e+06	1.290820e-03	2.514687e-06
std	1.423320e+02	6.038582e+05	2.888243e+06	2.924049e+06	3.399180e+06	3.674129e+06	3.590480e-02	1.585775e-03
min	1.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00
25%	1.560000e+02	1.338957e+04	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00
50%	2.390000e+02	7.487194e+04	1.420800e+04	0.000000e+00	1.327057e+05	2.146614e+05	0.000000e+00	0.000000e+00
75%	3.350000e+02	2.087215e+05	1.073152e+05	1.442584e+05	9.430367e+05	1.111909e+06	0.000000e+00	0.000000e+00
max	7.430000e+02	9.244552e+07	5.958504e+07	4.958504e+07	3.560159e+08	3.561793e+08	1.000000e+00	1.000000e+00
# checking the missing values in the data
df.isna().sum()
step              0
type              0
amount            0
nameOrig          0
oldbalanceOrg     0
newbalanceOrig    0
nameDest          0
oldbalanceDest    0
newbalanceDest    0
isFraud           0
isFlaggedFraud    0
dtype: int64
The dataset contains no null values

# printing unique values counts for each column
for col in df.columns:
    print(f"Unique values for column {col}")
    print(df[col].value_counts())
    print("======================================================")
Unique values for column step
step
19     51352
18     49579
187    49083
235    47491
307    46968
       ...  
245        4
28         4
655        4
112        2
662        2
Name: count, Length: 743, dtype: int64
======================================================
Unique values for column type
type
CASH_OUT    2237500
PAYMENT     2151495
CASH_IN     1399284
TRANSFER     532909
DEBIT         41432
Name: count, dtype: int64
======================================================
Unique values for column amount
amount
10000000.00    3207
10000.00         88
5000.00          79
15000.00         68
500.00           65
               ... 
20464.65          1
26299.05          1
401295.63         1
499807.64         1
1136700.07        1
Name: count, Length: 5316900, dtype: int64
======================================================
Unique values for column nameOrig
nameOrig
C1530544995    3
C545315117     3
C724452879     3
C1784010646    3
C1677795071    3
              ..
C1567523029    1
C644777639     1
C1256645416    1
C1231536757    1
C1971151096    1
Name: count, Length: 6353307, dtype: int64
======================================================
Unique values for column oldbalanceOrg
oldbalanceOrg
0.00         2102449
184.00           918
133.00           914
195.00           912
164.00           909
              ...   
87528.25           1
42686.88           1
32023.32           1
173464.47          1
5737.49            1
Name: count, Length: 1845844, dtype: int64
======================================================
Unique values for column newbalanceOrig
newbalanceOrig
0.00          3609566
26099.09            4
3684.32             4
18672.58            4
38767.21            4
               ...   
1156489.06          1
44707.62            1
29850.29            1
2251.93             1
165200.06           1
Name: count, Length: 2682586, dtype: int64
======================================================
Unique values for column nameDest
nameDest
C1286084959    113
C985934102     109
C665576141     105
C2083562754    102
C1590550415    101
              ... 
M295304806       1
M33419717        1
M1940055334      1
M335107734       1
M1757317128      1
Name: count, Length: 2722362, dtype: int64
======================================================
Unique values for column oldbalanceDest
oldbalanceDest
0.00           2704388
10000000.00        615
20000000.00        219
30000000.00         86
40000000.00         31
                ...   
2039554.04           1
587552.25            1
1326910.11           1
230693.29            1
851586.36            1
Name: count, Length: 3614697, dtype: int64
======================================================
Unique values for column newbalanceDest
newbalanceDest
0.00           2439433
10000000.00         53
971418.91           32
19169204.93         29
16532032.16         25
                ...   
1347758.15           1
3878719.83           1
1605826.83           1
592930.77            1
2580880.68           1
Name: count, Length: 3555499, dtype: int64
======================================================
Unique values for column isFraud
isFraud
0    6354407
1       8213
Name: count, dtype: int64
======================================================
Unique values for column isFlaggedFraud
isFlaggedFraud
0    6362604
1         16
Name: count, dtype: int64
======================================================
EDA
df.head()
step	type	amount	nameOrig	oldbalanceOrg	newbalanceOrig	nameDest	oldbalanceDest	newbalanceDest	isFraud	isFlaggedFraud
0	1	PAYMENT	9839.64	C1231006815	170136.0	160296.36	M1979787155	0.0	0.0	0	0
1	1	PAYMENT	1864.28	C1666544295	21249.0	19384.72	M2044282225	0.0	0.0	0	0
2	1	TRANSFER	181.00	C1305486145	181.0	0.00	C553264065	0.0	0.0	1	0
3	1	CASH_OUT	181.00	C840083671	181.0	0.00	C38997010	21182.0	0.0	1	0
4	1	PAYMENT	11668.14	C2048537720	41554.0	29885.86	M1230701703	0.0	0.0	0	0
sns.set_style("dark") # set the style of the plot as dark grid
sns.set_palette("pastel")
plt.figure(figsize = (8,6))
df['type'].value_counts().plot(kind = 'bar', color = '#F47F10')
plt.title('Type of transaction', color = '#F41010', fontsize = 20)
plt.xticks(rotation = 45, color = '#F41010')
plt.xlabel('Type', fontsize = 18, color = '#F41010')
plt.ylabel('count', fontsize = 18, color = '#F41010')
plt.show()

# Transaction amount
sns.set_style("dark") # set the style of the plot as dark grid
sns.set_palette("pastel")
plt.figure(figsize = (10,5))
df['amount'].value_counts().sort_values(ascending = False).head().plot(kind = 'bar',  color = '#F47F10')
plt.title("Amount of the transaction", fontsize = 20, color = "#F41010")
plt.xticks(rotation = 0, fontsize = 12, color = '#F41010')
plt.xlabel('Amount', fontsize = 16, color = '#F41010')
plt.ylabel('Count', fontsize = 16, color = '#F41010')
plt.show()

The most frequent transaction amount is 10,000,000,dollar occurring 3207 times. This suggests that there may be a common transaction size or a default value used for certain types of transactions.
The distribution of transaction amounts appears to be heavily skewed, with a significant number of transactions being of the dominant amount. This could indicate a specific type of transaction or a system-generated value.
While 10,000,000 dollar is the most common transaction amount, there are also other amounts occurring with lesser frequency. This indicates some variability in the transaction sizes, although they are less common compared to the dominant amount.
It might be worth investigating transactions that are significantly different from the most common amount. For instance, the presence of transactions with amounts much smaller than 10,000,000 dollar (e.g., 500 dollar or 5,000 dollar) could be outliers or represent a different category of transactions.
df.groupby('type').count()['amount']
type
CASH_IN     1399284
CASH_OUT    2237500
DEBIT         41432
PAYMENT     2151495
TRANSFER     532909
Name: amount, dtype: int64
counts = df.groupby('type').count()['amount']
plt.figure(figsize = (6,6))
plt.pie(counts, labels = counts.index, autopct = "%1.1f%%", colors=['#F47F10', '#F41010', '#F47810', '#F4C010','#F4D510'], shadow = True,explode = (0.1, 0, 0, 0, 0),textprops={'fontsize': 15})
plt.title('Count of each type of transaction', fontweight = 'bold', fontsize = 18, fontfamily = 'times new roman')
plt.show()

Upon checking plotting the distribution of type of amounts it can be seen that:
There are five types of transaction that includes CASH-OUT, CASH IN, DEBIT, TRANSFER, PAYMENT
CASH-OUT has the hight count, followed by CASH-IN and PAYMENT. This shows that these types of payement are most common in making fraud.
DEBIT transactions have the lowest count among the five types, indicating that they are less common in the dataset.
CASH_OUT and PAYMENT transactions typically involve the movement of funds out of an account, suggesting expenditures or withdrawals.
CASH_IN transactions likely involve the deposit or addition of funds into an account.
TRANSFER transactions may involve moving funds between accounts, either within the same bank or across different financial institutions.
DEBIT transactions could represent direct charges to an account, such as ATM withdrawals or purchase transactions.
Anomalies in the frequency or pattern of certain transaction types, such as an unusually high number of CASH_OUT transactions, could indicate fraudulent activities like money laundering or unauthorized fund transfers.
Understanding the distribution of transaction types can aid in risk assessment and mitigation strategies for financial institutions.
Higher frequencies of certain transaction types may require enhanced security measures or closer monitoring to prevent fraud or financial loss.
Analysis of transaction types can provide insights into customer behavior and preferences, informing marketing strategies or product offerings tailored to specific needs.
df.groupby(['type','isFraud']).count()
step	amount	nameOrig	oldbalanceOrg	newbalanceOrig	nameDest	oldbalanceDest	newbalanceDest	isFlaggedFraud
type	isFraud									
CASH_IN	0	1399284	1399284	1399284	1399284	1399284	1399284	1399284	1399284	1399284
CASH_OUT	0	2233384	2233384	2233384	2233384	2233384	2233384	2233384	2233384	2233384
1	4116	4116	4116	4116	4116	4116	4116	4116	4116
DEBIT	0	41432	41432	41432	41432	41432	41432	41432	41432	41432
PAYMENT	0	2151495	2151495	2151495	2151495	2151495	2151495	2151495	2151495	2151495
TRANSFER	0	528812	528812	528812	528812	528812	528812	528812	528812	528812
1	4097	4097	4097	4097	4097	4097	4097	4097	4097
There is a fraud transaction present in CAHS-OUT and TRANSFER
CASH_OUT and TRANSFER types have relatively higher counts of fraudulent transactions compared to others, which could indicate that these types are more vulnerable to fraudulent activities.
The presence of fraudulent transactions in certain types highlights the challenges in fraud detection and prevention, especially in high-risk transaction types like CASH_OUT and TRANSFER.
Finding the correlation between the attributes (Pearson correlation matrix)
# selecting the columns of numerical type
numeric_columns = df.select_dtypes(include=['int', 'float']).columns
numeric_data = df[numeric_columns]

# pearson corrleation matrix of the numerical data
correlation = numeric_data.corr()
# visulaising the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation,vmin = -1, vmax = 1,cmap = "Greys",annot = True, fmt = '.2f')
plt.title('Pearson Correlation Matrix', fontsize=16)
plt.xticks(rotation = 45)
plt.show()

Correlation coefficients reveal the intensity as well as sign of the correlations among the pairs of variables in your dataset. Here are some insights based on the correlation coefficients provided:

Amount and New Balance in Destination Account:.

The correlation between 'amount' and 'newcomerVolumeDest' is relatively strong and the relation is positive and linear (0.459).

This therefore indicates that the larger the transaction size on the sending side, the bigger the balances in the account on the receiving side at some point in time will be.

Amount and Old Balance in Destination Account:.

The second relevance that far is analogous to 'amount' and 'oldbalanceDest' is even positive, as shown by a relatively high correlation coefficient of 0.294.

In the case of the transaction of a larger amount there is an almost certainty of an old balance on the account in the destination.

Old Balance in Origin Account and New Balance in Origin Account:.

The degree of association between 'oldbalanceOrg' and 'newbalanceOrig' assessed by the univariate correlation coefficient is as close to 1( 0.998 ), thereby indicating a very strong and positive linear relationship.

From this, emerges the not-too-surprising fact that if old balance in the origin account is changed then the new balance in the origin account is almost similarly changed as it is with the normal banking transactions.

Is Fraud and Transaction Amount:

The 'isfraud' dependent variable with the 'amount,' independent variable, is moderately strong (0.077), displaying a positive linear relationship.

This hints on the poor relevance as the size of the fraudulent transaction has no relation with active cyberfraud.

Is Fraud and Flagged Fraud:

The metric 'isFraud' and 'isFlaggedFraud' are rather weak (0.044), a positive linear association exists.

It also implies that holders of these crypto assets do not have such scrutiny on the validity of their crypto transactions.

Step and Fraudulent Activity:

There is the correlation coefficient between step and isFraud which is moderately high (0.032), that is to say, these metrics only correlate with each other minimally.

Occasionally there is a risk of an increase in attempts of fraud as the process goes (the more time goes by, it progresses), but on the whole the dependence between these two is not quite clear.

Implemention of machine learning algorithm:
Important Note: Transactions identified as fraudulent are annulled. Hence, for fraud detection analysis, the following columns should not be utilized: oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.

# Removing the columns that are not necessary for the data modeling
# the columns that are not necessary are oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
df.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'], axis = 1, inplace = True)
# nameDest and nameOrig can also be removed
df.drop(['nameOrig', 'nameDest'], axis = 1, inplace = True)
# printing the data frame after removing the columns
df.head()
step	type	amount	isFraud	isFlaggedFraud
0	1	PAYMENT	9839.64	0	0
1	1	PAYMENT	1864.28	0	0
2	1	TRANSFER	181.00	1	0
3	1	CASH_OUT	181.00	1	0
4	1	PAYMENT	11668.14	0	0
# encoding the categorical column into numerical data
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
# separating feature variables and class variables
X = df.drop('isFraud', axis = 1)
y = df['isFraud']
# standardizing the data
sc = StandardScaler()
X = sc.fit_transform(X)
# splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
Logistic Regression
# make an object of logistic regression
lr = LogisticRegression()

#fitting the trainig data into lr model
lr.fit(X_train, y_train)

LogisticRegression
LogisticRegression()
# testing the model on test data
y_pred = lr.predict(X_test)
# calculating the performance matrix

#accuracy of the logistic regression
accuracy_lr = accuracy_score(y_test, y_pred)

# precision of the logistic regression
precision_lr = precision_score(y_test, y_pred)

# recall of the logistic regression
recall_lr = recall_score(y_test, y_pred)

# classification report
classification_lr = classification_report(y_test, y_pred)

# print the performance matrix
print(f"Accuracy of logistic regression {accuracy_lr}")
print(f"Precision of logistic regression {precision_lr}")
print(f"Recall of logistic regression {recall_lr}")
print(f"Classification Report of logistic regression\n {classification_lr}")
Accuracy of logistic regression 0.9987086032693031
Precision of logistic regression 0.1590909090909091
Recall of logistic regression 0.002874743326488706
Classification Report of logistic regression
               precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906351
           1       0.16      0.00      0.01      2435

    accuracy                           1.00   1908786
   macro avg       0.58      0.50      0.50   1908786
weighted avg       1.00      1.00      1.00   1908786

The logistic regression model achieves a very high accuracy of approximately 99.87%. However, its precision and recall for the positive class are notably low, around 15.91% and 0.29%, respectively. This indicates that while the model performs well in predicting negative instances, it struggles to identify positive instances accurately. This suggests the model's limited effectiveness in scenarios where correctly identifying positive cases is critical.

Decision Tree
# make an object of logistic regression
sv = DecisionTreeClassifier(max_depth = 20)

#fitting the trainig data into lr model
sv.fit(X_train, y_train)

# testing the model on test data
y_pred = sv.predict(X_test)
# calculating the performance matrix

#accuracy of the logistic regression
accuracy_sv = accuracy_score(y_test, y_pred)

# precision of the logistic regression
precision_sv = precision_score(y_test, y_pred)

# recall of the logistic regression
recall_sv = recall_score(y_test, y_pred)

# classification report
classification_sv = classification_report(y_test, y_pred)

# print the performance matrix
print(f"Accuracy of Decision Tree {accuracy_sv}")
print(f"Precision of Decision Tree {precision_sv}")
print(f"Recall of Decision Tree {recall_sv}")
print(f"Classification Report of Decision Tree\n {classification_sv}")
Accuracy of Decision Tree 0.9989726454406099
Precision of Decision Tree 0.659919028340081
Recall of Decision Tree 0.4016427104722793
Classification Report of Decision Tree
               precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906351
           1       0.66      0.40      0.50      2435

    accuracy                           1.00   1908786
   macro avg       0.83      0.70      0.75   1908786
weighted avg       1.00      1.00      1.00   1908786

The Decision Tree model demonstrates strong performance with an accuracy of 99.90%. It achieves a precision of 66.21% and recall of 40.08% for the positive class, similar to the logistic regression model. This suggests the Decision Tree model effectively identifies true positives among its positive predictions while capturing 40.08% of actual positive instances. The model maintains a high F1-score of 0.50 for the positive class, indicating balanced precision and recall.

MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes= 10, batch_size= 32, learning_rate= 'adaptive', learning_rate_init=0.001)
mlp.fit(X_train, y_train)

MLPClassifier
MLPClassifier(batch_size=32, hidden_layer_sizes=10, learning_rate='adaptive')
y_pred = mlp.predict(X_test)
# calculating the performance matrix

#accuracy of the logistic regression
accuracy_mlp = accuracy_score(y_test, y_pred)

# precision of the logistic regression
precision_mlp = precision_score(y_test, y_pred)

# recall of the logistic regression
recall_mlp = recall_score(y_test, y_pred)

# classification report
classification_mlp = classification_report(y_test, y_pred)

# print the performance matrix
print(f"Accuracy of MLP Classifier {accuracy_mlp}")
print(f"Precision of MLP Classifier {precision_mlp}")
print(f"Recall of MLP Classifier {recall_mlp}")
print(f"Classification Report of MLP Classifier\n {classification_mlp}")
Accuracy of MLP Classifier 0.998888822529084
Precision of MLP Classifier 0.9131578947368421
Recall of MLP Classifier 0.14250513347022586
Classification Report of MLP Classifier
               precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906351
           1       0.91      0.14      0.25      2435

    accuracy                           1.00   1908786
   macro avg       0.96      0.57      0.62   1908786
weighted avg       1.00      1.00      1.00   1908786

The MLP Classifier exhibits an impressive accuracy of 99.89%. However, its precision for the positive class is notably high at 92.66%, while its recall is much lower at 14.00%. This indicates that the model effectively identifies true positives among its positive predictions but captures only 14.00% of actual positive instances. Consequently, the F1-score for the positive class is relatively low at 0.24. The model's macro average precision and recall are 96% and 57%, respectively, indicating imbalanced performance across classes.

Comparing the model
performance_df = pd.DataFrame({
    'models' : ['Multi Layer Perceptron', 'Logistic Regression', 'Decision Tree'],
    'accuracy' : [accuracy_mlp, accuracy_lr, accuracy_sv],
    'precision' : [precision_mlp, precision_lr, precision_sv],
    'recall' : [recall_mlp, recall_lr, recall_sv]
})
performance_df
models	accuracy	precision	recall
0	Multi Layer Perceptron	0.998889	0.913158	0.142505
1	Logistic Regression	0.998709	0.159091	0.002875
2	Decision Tree	0.998973	0.659919	0.401643
# Create a figure and multiple axis objects
fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

metrics = ['accuracy', 'precision', 'recall']

for i, metric in enumerate(metrics):
    performance_df[metric].plot(kind='bar', ax=ax[i], color = '#F47F10')
    
    # Set the tick labels and rotation
    ax[i].set_xticklabels(performance_df['models'], rotation=45, fontweight = 'bold')
    
    # Adding labels
    ax[i].set_xlabel('Models', fontsize = 14)
    ax[i].set_ylabel(metric.capitalize(), fontsize = 14)  # Use the metric name as ylabel
    ax[i].set_title(f'{metric.capitalize()} by Model', fontsize = 20, fontweight = 'bold')  # Set dynamic title

plt.tight_layout()
plt.show()

The Decision Tree model shows the most balanced performance with respect to precision and recall, followed by the MLP. Logistic Regression performs poorly in correctly identifying positive instances despite its high accuracy.