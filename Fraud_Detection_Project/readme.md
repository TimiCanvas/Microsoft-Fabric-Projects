This project aimed to build a credit card fraud detection model, using ML algorithms trained on historical Nigerian bank data and then using the model to detect future fraudulent transactions.

The main steps in this notebook are:
1. Installing custom libraries
2. Loading the data 
3. Understanding and processing the data through exploratory data analysis
4. Training a machine learning model using scikit-learn and tracking experiments using MLflow and Fabric Autologging feature
5. Saving and register the best performing machine learning model
6. Loading the machine learning model for scoring and making predictions


A "lakehouse" was added to this notebook, because I downloaded the dataset from an SQL server and stored the data in the lakehouse. 

Some of the features present in the dataset include:
Customer details;
1.	AcountNumber	
2.	Cvv
3.	Card information
Transaction details;
4.	Domain
5.	TransactionType
6.	Average_Income_expendicture	
7.	New_Balance	
8.	Old_Balance
9.	Amount
10.	ATM 
11.	POS&WEB
12.	Credit_Limit

And the conditions to flag transactions as Fraudulent are:
A.	If 7 is not equal to the sum of 8 +9 then fraud
B.	If 5= credit and 9 is greater than 12 then fraud
C.	If 5 =debit and amount is greater than 6 then fraud
D.	If customers information is wrong and entered more that 3 times trials then wrong
E.	If 9 is more than 12 and tried more that three times then fraud