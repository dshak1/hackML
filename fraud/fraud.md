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




DATASET DESCRIPTION:
FRAUD | HackML 2026
Please check www.sfudsss.com for more information!

FRAUD | HackML 2026

Submit Prediction
Dataset Description
Each row represents a single mobile money transaction generated from the PaySim simulator. All identifiers are synthetic. The target variable for this competition is urgency_level.

Files
train.csv - the training set
test.csv - the test set
sample_submission.csv - a sample submission file in the correct format
Columns
id
Type: Integer
Description: Unique transaction identifier assigned during dataset preparation. Used to align predictions with the correct rows in 'test.csv'
step
Type: Integer
Description: Time step of the transaction, where 1 step = 1 hour since the start of the simulation.
type
Type: Categorical (string)
Description: Type of transaction. Common values include: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
amount
Type: Numeric (float)
Description: Transaction amount in local currency.
oldbalanceOrg
Type: Numeric (float)
Description: Balance of the origin account before the transaction.
newbalanceOrg
Type: Numeric (float)
Description: Balance of the origin account after the transaction.
oldbalanceDest
Type: Numeric (float)
Description: Balance of the destination account before the transaction. Some destination accounts may represent merchants (in the original PaySim formulation), for which certain balance fields may not apply.
newbalanceDest
Type: Numeric (float)
Description: Balance of the destination account after the transaction.
nameOrig
Type: String
Description: The customer initiating the transaction.
nameDest
Type: String
Description: The transaction's recipient customer.
Target Variable
urgency_level
Type: Integer (categorical)
Valid values: {0, 1, 2, 3}
Description: A derived categorical label indicating the recommended urgency of fraud investigation for a transaction. Higher values represent higher risk and require faster response.
Class meanings:
0 — No Action: Transaction appears legitimate; no investigation required.
1 — Monitor: Low-risk suspicious behavior; monitor for patterns or repeated activity.
2 — Review: Likely fraudulent; should be reviewed by an analyst.
3 — Immediate Action: High-risk fraud; requires urgent investigation or intervention.
Notes:

urgency_level is organizer-defined for this competition and is not part of the original PaySim dataset.
The exact derivation procedure is intentionally not disclosed to participants.
The dataset is highly imbalanced, reflecting real-world fraud rarity.
Files
2 files

Size
501.71 MB

Type
csv

License
Subject to Competition Rules

test.csv(9.12 MB)

10 of 10 columns


step

type

amount

nameOrig

oldbalanceOrg

newbalanceOrig

nameDest

oldbalanceDest

newbalanceDest

id
Label	Count
596.00 - 598.94	12,739
598.94 - 601.88	2,699
601.88 - 604.82	28
604.82 - 607.76	38
607.76 - 610.70	2,289
610.70 - 613.64	2,736
613.64 - 616.58	1,292
616.58 - 619.52	5,488
619.52 - 622.46	1,133
622.46 - 625.40	40
625.40 - 628.34	16
628.34 - 631.28	182
631.28 - 634.22	26
634.22 - 637.16	3,418
637.16 - 640.10	1,426
640.10 - 643.04	1,165
643.04 - 645.98	2,277
645.98 - 648.92	44
648.92 - 651.86	289
651.86 - 654.80	30
654.80 - 657.74	1,979
657.74 - 660.68	2,526
660.68 - 663.62	4,889
663.62 - 666.56	1,141
666.56 - 669.50	1,730
669.50 - 672.44	2,077
672.44 - 675.38	489
675.38 - 678.32	22
678.32 - 681.26	4,797
681.26 - 684.20	6,760
684.20 - 687.14	12,743
687.14 - 690.08	12,871
690.08 - 693.02	11,711
693.02 - 695.96	5,491
695.96 - 698.90	83
698.90 - 701.84	79
701.84 - 704.78	38
704.78 - 707.72	2,675
707.72 - 710.66	4,105
710.66 - 713.60	39
713.60 - 716.54	2,700
716.54 - 719.48	1,564
719.48 - 722.42	24
722.42 - 725.36	32
725.36 - 728.30	44
728.30 - 731.24	48
731.24 - 734.18	28
734.18 - 737.12	36
737.12 - 740.06	26
740.06 - 743.00	44
596
743
PAYMENT
35%
CASH_OUT
31%
Other (39731)
34%
Label	Count
0.00 - 304457.62	102,224
304457.62 - 608915.23	11,684
608915.23 - 913372.85	1,875
913372.85 - 1217830.46	775
1217830.46 - 1522288.08	387
1522288.08 - 1826745.69	227
1826745.69 - 2131203.31	140
2131203.31 - 2435660.92	96
2435660.92 - 2740118.54	91
2740118.54 - 3044576.15	54
3044576.15 - 3349033.77	48
3349033.77 - 3653491.38	42
3653491.38 - 3957949.00	25
3957949.00 - 4262406.61	32
4262406.61 - 4566864.23	19
4566864.23 - 4871321.84	18
4871321.84 - 5175779.46	19
5175779.46 - 5480237.07	20
5480237.07 - 5784694.69	20
5784694.69 - 6089152.30	16
6089152.30 - 6393609.92	33
6393609.92 - 6698067.53	24
6698067.53 - 7002525.15	20
7002525.15 - 7306982.76	13
7306982.76 - 7611440.38	29
7611440.38 - 7915898.00	16
7915898.00 - 8220355.61	11
8220355.61 - 8524813.23	13
8524813.23 - 8829270.84	15
8829270.84 - 9133728.46	12
9133728.46 - 9438186.07	10
9438186.07 - 9742643.69	6
9742643.69 - 10047101.30	119
10047101.30 - 10351558.92	2
10351558.92 - 10656016.53	1
10960474.15 - 11264931.76	1
11264931.76 - 11569389.38	2
11873846.99 - 12178304.61	3
12482762.22 - 12787219.84	1
14613965.53 - 14918423.14	1
14918423.14 - 15222880.76	2
0
15.2m
118144

unique values
Label	Count
0.00 - 1146325.10	104,538
1146325.10 - 2292650.20	2,917
2292650.20 - 3438975.30	2,400
3438975.30 - 4585300.40	1,998
4585300.40 - 5731625.50	1,606
5731625.50 - 6877950.61	1,403
6877950.61 - 8024275.71	1,122
8024275.71 - 9170600.81	809
9170600.81 - 10316925.91	576
10316925.91 - 11463251.01	321
11463251.01 - 12609576.11	151
12609576.11 - 13755901.21	71
13755901.21 - 14902226.31	38
14902226.31 - 16048551.41	21
16048551.41 - 17194876.52	29
17194876.52 - 18341201.62	30
18341201.62 - 19487526.72	26
19487526.72 - 20633851.82	27
20633851.82 - 21780176.92	18
21780176.92 - 22926502.02	13
22926502.02 - 24072827.12	9
24072827.12 - 25219152.22	7
25219152.22 - 26365477.32	1
26365477.32 - 27511802.42	2
29804452.63 - 30950777.73	1
32097102.83 - 33243427.93	2
33243427.93 - 34389753.03	1
35536078.13 - 36682403.23	1
36682403.23 - 37828728.33	2
40121378.54 - 41267703.64	1
42414028.74 - 43560353.84	1
44706678.94 - 45853004.04	1
46999329.14 - 48145654.24	1
49291979.34 - 50438304.44	1
56169929.95 - 57316255.05	1
0
57.3m
Label	Count
0.00 - 946325.10	104,012
946325.10 - 1892650.20	2,378
1892650.20 - 2838975.30	2,210
2838975.30 - 3785300.40	1,776
3785300.40 - 4731625.50	1,607
4731625.50 - 5677950.61	1,349
5677950.61 - 6624275.71	1,163
6624275.71 - 7570600.81	1,040
7570600.81 - 8516925.91	772
8516925.91 - 9463251.01	633
9463251.01 - 10409576.11	429
10409576.11 - 11355901.21	281
11355901.21 - 12302226.31	162
12302226.31 - 13248551.41	76
13248551.41 - 14194876.51	46
14194876.51 - 15141201.62	28
15141201.62 - 16087526.72	17
16087526.72 - 17033851.82	25
17033851.82 - 17980176.92	23
17980176.92 - 18926502.02	24
18926502.02 - 19872827.12	19
19872827.12 - 20819152.22	21
20819152.22 - 21765477.32	15
21765477.32 - 22711802.42	13
22711802.42 - 23658127.52	9
23658127.52 - 24604452.63	5
24604452.63 - 25550777.73	4
25550777.73 - 26497102.83	1
26497102.83 - 27443427.93	2
30282403.23 - 31228728.33	1
32175053.43 - 33121378.53	1
35014028.74 - 35960353.84	1
36906678.94 - 37853004.04	1
39745654.24 - 40691979.34	1
46369929.95 - 47316255.05	1
0
47.3m
113488

unique values
Label	Count
0.00 - 6559961.48	113,548
6559961.48 - 13119922.97	3,229
13119922.97 - 19679884.45	743
19679884.45 - 26239845.94	242
26239845.94 - 32799807.42	133
32799807.42 - 39359768.91	77
39359768.91 - 45919730.39	46
45919730.39 - 52479691.88	33
52479691.88 - 59039653.36	14
59039653.36 - 65599614.84	11
65599614.84 - 72159576.33	17
72159576.33 - 78719537.81	9
78719537.81 - 85279499.30	14
85279499.30 - 91839460.78	2
91839460.78 - 98399422.27	7
98399422.27 - 104959383.75	3
104959383.75 - 111519345.23	2
111519345.23 - 118079306.72	1
118079306.72 - 124639268.20	3
137759191.17 - 144319152.66	2
144319152.66 - 150879114.14	2
150879114.14 - 157439075.63	1
163999037.11 - 170558998.59	1
170558998.59 - 177118960.08	1
190238883.05 - 196798844.53	1
216478728.99 - 223038690.47	1
229598651.95 - 236158613.44	2
321438112.74 - 327998074.22	1
0
328m
Label	Count
0.00 - 6568633.96	113,205
6568633.96 - 13137267.93	3,470
13137267.93 - 19705901.89	799
19705901.89 - 26274535.86	273
26274535.86 - 32843169.82	135
32843169.82 - 39411803.79	83
39411803.79 - 45980437.75	50
45980437.75 - 52549071.72	33
52549071.72 - 59117705.68	15
59117705.68 - 65686339.65	13
65686339.65 - 72254973.61	17
72254973.61 - 78823607.58	8
78823607.58 - 85392241.54	15
85392241.54 - 91960875.50	2
91960875.50 - 98529509.47	7
98529509.47 - 105098143.43	3
105098143.43 - 111666777.40	2
118235411.36 - 124804045.33	4
137941313.26 - 144509947.22	2
144509947.22 - 151078581.19	2
151078581.19 - 157647215.15	1
164215849.12 - 170784483.08	1
170784483.08 - 177353117.04	1
190490384.97 - 197059018.94	1
229902188.76 - 236470822.73	3
321863064.27 - 328431698.23	1
0
328m
Label	Count
6244475.00 - 6246837.90	2,363
6246837.90 - 6249200.80	2,363
6249200.80 - 6251563.70	2,363
6251563.70 - 6253926.60	2,363
6253926.60 - 6256289.50	2,363
6256289.50 - 6258652.40	2,363
6258652.40 - 6261015.30	2,363
6261015.30 - 6263378.20	2,363
6263378.20 - 6265741.10	2,363
6265741.10 - 6268104.00	2,362
6268104.00 - 6270466.90	2,363
6270466.90 - 6272829.80	2,363
6272829.80 - 6275192.70	2,363
6275192.70 - 6277555.60	2,363
6277555.60 - 6279918.50	2,363
6279918.50 - 6282281.40	2,363
6282281.40 - 6284644.30	2,363
6284644.30 - 6287007.20	2,363
6287007.20 - 6289370.10	2,363
6289370.10 - 6291733.00	2,362
6291733.00 - 6294095.90	2,363
6294095.90 - 6296458.80	2,363
6296458.80 - 6298821.70	2,363
6298821.70 - 6301184.60	2,363
6301184.60 - 6303547.50	2,363
6303547.50 - 6305910.40	2,363
6305910.40 - 6308273.30	2,363
6308273.30 - 6310636.20	2,363
6310636.20 - 6312999.10	2,363
6312999.10 - 6315362.00	2,362
6315362.00 - 6317724.90	2,363
6317724.90 - 6320087.80	2,363
6320087.80 - 6322450.70	2,363
6322450.70 - 6324813.60	2,363
6324813.60 - 6327176.50	2,363
6327176.50 - 6329539.40	2,363
6329539.40 - 6331902.30	2,363
6331902.30 - 6334265.20	2,363
6334265.20 - 6336628.10	2,363
6336628.10 - 6338991.00	2,362
6338991.00 - 6341353.90	2,363
6341353.90 - 6343716.80	2,363
6343716.80 - 6346079.70	2,363
6346079.70 - 6348442.60	2,363
6348442.60 - 6350805.50	2,363
6350805.50 - 6353168.40	2,363
6353168.40 - 6355531.30	2,363
6355531.30 - 6357894.20	2,363
6357894.20 - 6360257.10	2,363
6360257.10 - 6362620.00	2,363
6.24m
6.36m
596
CASH_IN
145988.55
C587825574
29478
175466.55
C388319905
8495018.14
8349029.59
6244475
596
CASH_OUT
11185.94
C845520261
29823
18637.06
C297085439
0
11185.94
6244476
596
CASH_OUT
285264.27
C704086280
72313
0
C844556604
0
285264.27
6244477
596
PAYMENT
5962.45
C749354151
51625
45662.55
M1136761893
0
0
6244478
596
PAYMENT
12209.43
C1574353005
45662.55
33453.12
M1220316446
0
0
6244479
596
CASH_OUT
124530.66
C890167210
629773
505242.34
C1870027905
2996509.47
3121040.12
6244480
596
PAYMENT
4749.24
C2061585818
505242.34
500493.1
M303863675
0
0
6244481
596
CASH_OUT
80615.61
C1055145051
28035
0
C278408496
305206.45
385822.06
6244482
596
PAYMENT
3327.25
C2102675362
0
0
M220813327
0
0
6244483
596
CASH_IN
204207.7
C2071503262
22762
226969.7
C1213633945
117865.11
0
6244484
596
PAYMENT
8117.57
C1684098126
10356
2238.43
M1003978295
0
0
6244485
596
CASH_OUT
160494.25
C2096734306
5071
0
C1826897336
1822125.57
1982619.82
6244486
596
CASH_OUT
58061.52
C1110673970
23370
0
C415511180
0
58061.52
6244487
596
PAYMENT
5038.4
C744402182
0
0
M746084476
0
0
6244488
596
CASH_IN
248161.06
C489883425
39784
287945.06
C131340193
119716.59
0
6244489
596
PAYMENT
3558.89
C1125088431
287945.06
284386.17
M442556245
0
0
6244490
596
CASH_OUT
99734.09
C1958076381
10415
0
C78185977
1049101.78
1148835.87
6244491
596
CASH_OUT
31678.52
C1899680869
5097
0
C1987013036
0
31678.52
6244492
596
CASH_IN
5971.83
C1239950569
360
6331.83
C1243350517
0
0
6244493
596
PAYMENT
9836.05
C1433325814
21838
12001.95
M268416689
0
0
6244494
596
TRANSFER
41515.84
C675402455
22306
0
C996751485
0
41515.84
6244495
596
PAYMENT
804.03
C537293914
72962
72157.97
M55372236
0
0
6244496
596
PAYMENT
13869.8
C1911856426
72157.97
58288.16
M1459961513
0
0
6244497
596
CASH_OUT
148015.38
C1144136887
5006
0
C4549098
128949.83
276965.22
6244498
596
PAYMENT
3174.46
C542418804
5775
2600.54
M1964798873
0
0
6244499
596
PAYMENT
6134.28
C185793509
2600.54
0
M1590714796
0
0
6244500
596
PAYMENT
2255.05
C1530528261
5560
3304.95
M1294737984
0
0
6244501
596
CASH_OUT
24407.24
C1375548883
3304.95
0
C1118290215
713882.33
738289.57
6244502
596
CASH_OUT
388350.54
C217965377
0
0
C1742598626
1107759.7
1496110.24
6244503
596
PAYMENT
657.63
C131920625
104421
103763.37
M734985832
0
0
6244504
596
PAYMENT
7717.41
C321197423
103763.37
96045.96
M1889606561
0
0
6244505
596
PAYMENT
15348.76
C277488702
162648
147299.24
M1282123257
0
0
6244506
596
TRANSFER
66221.7
C2011062134
54430
0
C855086770
152060.41
218282.11
6244507
596
CASH_IN
90934.45
C265956927
64
90998.45
C2121459771
435118.14
344183.69
6244508
596
CASH_OUT
31814.24
C1311785006
110581
78766.76
C116750305
181318.59
213132.84
6244509
596
PAYMENT
4657.6
C1608872060
59809
55151.4
M1522905369
0
0
6244510
596
CASH_OUT
236275.04
C70305704
55151.4
0
C1695722646
6142521.39
6378796.43
6244511
596
CASH_OUT
229703.99
C1213056560
15836
0
C1202273798
343172.26
572876.25
6244512
596
PAYMENT
3442.55
C1012873685
0
0
M1264912958
0
0
6244513
596
PAYMENT
1069.08
C1240107389
0
0
M1873848886
0
0
6244514
596
PAYMENT
5332.32
C719632260
216581
211248.68
M1236499484
0
0
6244515
596
CASH_IN
200488.16
C1418704182
112688
313176.16
C854712710
6241541.78
6041053.62
6244516
596
PAYMENT
4147.23
C1040471179
80600
76452.77
M827559000
0
0
6244517
596
CASH_OUT
285245.89
C1674728946
1257
0
C2112137818
1084914.83
1370160.73
6244518
596
CASH_OUT
178286.01
C1247640251
299485
121198.99
C1042395265
258391.43
436677.44
6244519
596
CASH_OUT
346634.35
C1449250679
207
0
C1928221309
196209.12
542843.48
6244520
596
PAYMENT
1800.84
C2071836847
0
0
M183570423
0
0
6244521
596
PAYMENT
3014.89
C727200944
0
0
M1931103451
0
0
6244522
596
CASH_IN
131206.59
C1697917655
2085
133291.59
C1309565315
7296.96
0
6244523
596
PAYMENT
6623.91
C1740119857
29874
23250.09
M221207234
0
0
6244524
596
PAYMENT
2176.67
C745714472
21355
19178.33
M1758742243
0
0
6244525
596
TRANSFER
314843.62
C1328713055
20646
0
C1009036787
2064026.77
2378870.39
6244526
596
PAYMENT
8280.63
C951858993
0
0
M32385277
0
0
6244527
596
PAYMENT
35469.71
C1262978429
0
0
M281325887
0
0
6244528
596
PAYMENT
10572.3
C1226312105
0
0
M960592311
0
0
6244529
596
PAYMENT
47092.29
C1322942351
0
0
M2074239522
0
0
6244530
596
PAYMENT
10684.14
C1929275838
0
0
M1077951127
0
0
6244531
596
PAYMENT
21144.24
C1568342807
0
0
M891345384
0
0
6244532
596
PAYMENT
6531.65
C882046315
0
0
M1976034796
0
0
6244533
596
PAYMENT
15333.93
C1889608696
0
0
M1132284829
0
0
6244534
596
PAYMENT
9238.94
C1625802548
0
0
M717596225
0
0
6244535
596
PAYMENT
2457.44
C1401089538
0
0
M2055890508
0
0
6244536
596
PAYMENT
7194.62
C604688462
0
0
M595234080
0
0
6244537
596
CASH_IN
99724.44
C1741460915
283671
383395.44
C1967141358
0
0
6244538
596
TRANSFER
441093.48
C1314731776
43
0
C918690588
0
441093.48
6244539
596
PAYMENT
9731.2
C1017579045
0
0
M1271107356
0
0
6244540
596
CASH_IN
125795.72
C1210186458
29358
155153.72
C1204905072
99426.79
0
6244541
596
PAYMENT
47643.63
C1877156329
155153.72
107510.09
M1810994206
0
0
6244542
596
CASH_IN
12227.06
C1016945939
1513
13740.06
C222849426
160279.16
148052.11
6244543
596
PAYMENT
7922.83
C2066581360
13740.06
5817.22
M230100373
0
0
6244544
596
CASH_OUT
533076.27
C1786853721
875728
342651.73
C2043583949
119529.87
652606.14
6244545
596
TRANSFER
355367.55
C1008895199
5035
0
C918886801
0
355367.55
6244546
596
PAYMENT
4159.63
C723747372
90941
86781.37
M508422520
0
0
6244547
596
CASH_OUT
74649.02
C2103791641
204102
129452.98
C254899021
0
74649.02
6244548
596
CASH_IN
69063.9
C1439246102
59665
128728.9
C124517408
1255346.03
1186282.13
6244549
596
CASH_IN
179867.67
C1649208673
9510
189377.67
C1972596608
279644.06
99776.39
6244550
596
CASH_OUT
98962.5
C322686585
189377.67
90415.17
C1045930008
145008.32
243970.82
6244551
596
CASH_OUT
154156.31
C1149874591
90415.17
0
C1700070777
532583.24
686739.55
6244552
596
PAYMENT
7846.19
C1491394447
50077
42230.81
M1573258775
0
0
6244553
596
CASH_OUT
159597.74
C272847824
42230.81
0
C1393408143
1321852.38
1481450.12
6244554
596
PAYMENT
3721.5
C424615182
211236
207514.5
M1739170063
0
0
6244555
596
PAYMENT
5892.99
C179500371
81070
75177.01
M6683142
0
0
6244556
596
TRANSFER
24000.11
C1638084713
75177.01
51176.9
C411451611
2603343.83
2627343.94
6244557
596
TRANSFER
1581016.28
C1269318795
51176.9
0
C1436420884
2007257.58
3588273.86
6244558
596
PAYMENT
9981.64
C1445800128
401828
391846.36
M2042060233
0
0
6244559
596
CASH_OUT
191559.06
C32443118
13322
0
C1112603234
0
191559.06
6244560
596
PAYMENT
5897.06
C149592644
1401
0
M935250097
0
0
6244561
596
PAYMENT
7642.08
C1948553882
0
0
M233135754
0
0
6244562
596
PAYMENT
2371.07
C1359076808
54963
52591.93
M280722457
0
0
6244563
596
PAYMENT
4664.02
C1079745750
52591.93
47927.91
M1735236441
0
0
6244564
596
PAYMENT
6884.9
C945681889
47927.91
41043
M1934433607
0
0
6244565
596
PAYMENT
14392.19
C1877770964
41043
26650.82
M1691380515
0
0
6244566
596
PAYMENT
11800.26
C871725221
26650.82
14850.56
M675937465
0
0
6244567
596
PAYMENT
41459.49
C599269702
14850.56
0
M1146543969
0
0
6244568
596
PAYMENT
12346.39
C1560196894
0
0
M993342476
0
0
6244569
596
PAYMENT
12599.49
C1499824189
0
0
M725283239
0
0
6244570
596
PAYMENT
11637.19
C39183769
0
0
M282211467
0
0
6244571
596
PAYMENT
16796.46
C1707289284
0
0
M419973138
0
0
6244572
596
PAYMENT
18720.8
C878448519
0
0
M892011499
0
0
6244573
596
PAYMENT
51.43
C1229097600
0
0
M315691874
0
0
6244574
kaggle competitions download -c fraud-hack-ml-2026
Download data

Metadata
License
Subject to Competition Rules