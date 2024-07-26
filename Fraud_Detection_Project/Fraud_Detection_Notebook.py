#!/usr/bin/env python
# coding: utf-8

# ## Fraud_Detection_Notebook
# 
# New notebook

# # Fraud detection model

# ## Introduction
# 
# In this notebook, I aim to build a credit card fraud detection model, using ML algorithms trained on historical Nigerian bank data and then using the model to detect future fraudulent transactions.
# 
# The main steps in this notebook are:
# 
# 1. Installing custom libraries
# 2. Loading the data 
# 3. Understanding and processing the data through exploratory data analysis
# 4. Training a machine learning model using scikit-learn and tracking experiments using MLflow and Fabric Autologging feature
# 5. Saving and register the best performing machine learning model
# 6. Loading the machine learning model for scoring and making predictions
# 
# #### Prerequisites
# - A "lakehouse" was added to this notebook, because I downloaded the dataset from an SQL server and stored the data in the lakehouse. 

# ## Step 1: Install custom libraries
# When developing a machine learning model or doing ad-hoc data analysis, one may need to install a custom library (such as `imblearn`) for the Apache Spark session.

# For this notebook, I installed the `imblearn` library, using `%pip install`.

# In[4]:


# Use pip to install imblearn
get_ipython().run_line_magic('pip', 'install imblearn')


# ## Step 2: Load the data

# ### Dataset
# 
# This research made used of some samples of Credit/Debit Cards data from Access Bank with strict and entrusted supervision so as to protect customer’s transaction details. The data set is available on request and have undergone proper organization ethics approval processes and available freely for research purposes.
# 
# - The features in the dataset includes `AcountNumber`, `CVV`, `CardInformation`, `CustomerAge`, `Gender`, `Marital Status`, `Cards`, `CardColour`, `CardType`, `TransactionType`, `Domain`, `ATM`, `POSWEBLimit`, `CreditLimit`, `Amount`, `AverageIncomeExpendicture`, `NewBalance`, `OldBalance` 
# 
# - The column `Outcome` is the target variable and takes the value `1` for fraud and `0` otherwise.
# 
# The Credit/Debit card data set consists of 37,097 observations and 19 attributes. 
# 
# The table below shows a preview of the _creditcard.csv_ data:

# In[5]:


df = spark.sql("SELECT * FROM Fraud_Detection_LakeHouse.creditcard LIMIT 5")
display(df)


# ### SMOTE
# During the course of this project I'd be applying the SMOTE approach to address the problem of imbalanced classification. Imbalanced classification happens when there are too few examples of the minority class for a model to effectively learn the decision boundary.

# ### Set up MLflow experiment tracking

# In[6]:


# Set up MLflow for experiment tracking
import mlflow

mlflow.set_experiment("fraud_detection")
mlflow.autolog(disable=True)  # Disable MLflow autologging


# ### Read raw data from the lakehouse
# 
# This code reads raw data from the lakehouse.

# In[7]:


df = spark.sql("SELECT * FROM Fraud_Detection_LakeHouse.creditcard")


# ## Step 3: Perform exploratory data analysis

# In this section, you'll begin by exploring the raw data and high-level statistics. You'll then transform the data by casting the columns into the correct types and converting from a Spark DataFrame into a pandas DataFrame for easier visualization. Finally, you explore and visualize the class distributions in the data.
# 
# ### Display the raw data
# 
# 1. Explore the raw data and view high-level statistics by using the `display` command. For more information, see [Notebook visualization in Microsoft Fabric](https://aka.ms/fabric/visualization).

# In[8]:


display(df)


# 2. Print some basic information about the dataset.

# In[9]:


# Print dataset basic information
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()


# ### Transform the data
# 
# 1. Cast the dataset's columns into the correct types.

# In[10]:


from pyspark.ml.feature import Imputer
from pyspark.sql import functions as F
from pyspark.sql.functions import col


# In[11]:


def clean_data(df):
    # Replace missing values with the mean of each column in: 'CustomerAge'
    cols = ['CustomerAge']
    imputer = Imputer(inputCols=cols, outputCols=cols, strategy='mean')
    df = imputer.fit(df).transform(df)
    # Round column 'CustomerAge' (Number of decimals: 0)
    df = df.withColumn('CustomerAge', F.round(F.col('CustomerAge'), 0))
    return df

df_clean = clean_data(df)
display(df_clean)


# In[12]:


df_clean.describe()


# In[13]:


df_clean.groupBy('Outcome').count().show()


# 2. Convert Spark DataFrame to Pandas DataFrame for easier visualization and processing.

# In[14]:


df_pd = df_clean.toPandas()


# ### Explore the  class distribution in the dataset
# 
# 1. Displaying the class distribution in the dataset.

# In[15]:


# Check the distribution of the target variable
outcome_counts = df_clean.groupBy('Outcome').count().toPandas()
total_count = outcome_counts['count'].sum()

# Calculate percentages
no_frauds_percentage = round(outcome_counts.loc[outcome_counts['Outcome'] == False, 'count'].values[0] / total_count * 100, 2)
frauds_percentage = round(outcome_counts.loc[outcome_counts['Outcome'] == True, 'count'].values[0] / total_count * 100, 2)

print('No Frauds:', no_frauds_percentage, '% of the dataset')
print('Frauds:', frauds_percentage, '% of the dataset')


# The class distribution shows that most of the transactions are fraudulent. Therefore, data preprocessing is required before model training in order to avoid overfitting.

# 2. Using a plot to show the class imbalance in the dataset, by viewing the distribution of fraudulent versus nonfraudulent transactions.

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

colors = ["#0101DF", "#DF0101"]
sns.countplot(x='Outcome', data=df_pd, palette=colors) 
plt.title('Outcome Distributions \n (False: No Fraud || True: Fraud)', fontsize=10)


# The distribution plot clearly shows the class imbalance in the dataset.

# 3. Showing the five-number summary (the minimum score, first quartile, median, third quartile, the maximum score) for the transaction amount, using Box plots.

# In[17]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
s1 = sns.boxplot(ax = ax1, x="Outcome", y="Amount", hue="Outcome",data=df_pd, palette="PRGn", showfliers=True) # Remove outliers from the plot
s2 = sns.boxplot(ax = ax2, x="Outcome", y="Amount", hue="Outcome",data=df_pd, palette="PRGn", showfliers=False) # Kepp outliers from the plot
plt.show()


# When the data is highly imbalanced, these Box plots may not demonstrate accurate insights. Alternatively, we can address the class imbalance problem first and then create the same plots for more accurate insights.

# ## Step 4: Train and evaluate the model

# In this section, I will train a LightGBM model to classify the fraud transactions. I'd train the LightGBM model on the imbalanced dataset and also on the balanced dataset (via SMOTE) and compare the performance of both models.

# ### Prepare training and test datasets
# 
# Before training, split the data into the training and test datasets.

# In[18]:


import pandas as pd

def clean_dataa(df_pd):
    # One-hot encode columns: 'Gender', 'Marital_Status' and 5 other columns
    for column in ['Gender', 'Marital_Status', 'Cards', 'CardColour', 'CardType', 'TransactionType', 'Domain']:
        insert_loc = df_pd.columns.get_loc(column)
        df_pd = pd.concat([df_pd.iloc[:,:insert_loc], pd.get_dummies(df_pd.loc[:, [column]]), df_pd.iloc[:,insert_loc+1:]], axis=1)
    return df_pd

df_pd_clean = clean_dataa(df_pd.copy())
df_pd_clean.head()


# In[19]:


# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_pd_clean, test_size=0.20)
feature_cols = [c for c in df_pd_clean.columns.tolist() if c not in ["Outcome"]]


# ### Applying SMOTE to the training data to synthesize new samples for the minority class

# In[20]:


# Apply SMOTE to the training data
from collections import Counter
from imblearn.over_sampling import SMOTE

X = train[feature_cols]
y = train["Outcome"]
print("Original dataset shape %s" % Counter(y))

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("Resampled dataset shape %s" % Counter(y_res))

new_train = pd.concat([X_res, y_res], axis=1)


# ### Training machine learning models and running experiments

# 1. Updating the MLflow autologging configuration to track additional metrics, parameters, and files, by setting `exclusive=False`.

# In[21]:


mlflow.autolog(exclusive=False)


# 2. Training two models using **LightGBM**: one model on the imbalanced dataset and the other on the balanced (via SMOTE) dataset. Then we'd compare the performance of the two models.

# In[22]:


import lightgbm as lgb

model = lgb.LGBMClassifier(objective="binary") # imbalanced dataset
smote_model = lgb.LGBMClassifier(objective="binary") # balanced dataset


# In[23]:


# Train LightGBM for both imbalanced and balanced datasets and define the evaluation metrics
print("Start training with imbalanced data:\n")
with mlflow.start_run(run_name="raw_data") as raw_run:
    model = model.fit(
        train[feature_cols],
        train["Outcome"],
        eval_set=[(test[feature_cols], test["Outcome"])],
        eval_metric="auc",
        callbacks=[
            lgb.log_evaluation(10),
        ],
    )

print(f"\n\nStart training with balanced data:\n")
with mlflow.start_run(run_name="smote_data") as smote_run:
    smote_model = smote_model.fit(
        new_train[feature_cols],
        new_train["Outcome"],
        eval_set=[(test[feature_cols], test["Outcome"])],
        eval_metric="auc",
        callbacks=[
            lgb.log_evaluation(10),
        ],
    )


# ### Determine feature importance for training

# Determining the feature importance for the model trained on the imbalanced dataset.

# In[24]:


with mlflow.start_run(run_id=raw_run.info.run_id):
    importance = lgb.plot_importance(
        model, title="Feature importance for imbalanced data"
    )
    importance.figure.savefig("feauture_importance.png")
    mlflow.log_figure(importance.figure, "feature_importance.png")


# Determining feature importance for the model you trained on balanced (via SMOTE) dataset.

# In[25]:


with mlflow.start_run(run_id=smote_run.info.run_id):
    smote_importance = lgb.plot_importance(
        smote_model, title="Feature importance for balanced (via SMOTE) data"
    )
    smote_importance.figure.savefig("feauture_importance_smote.png")
    mlflow.log_figure(smote_importance.figure, "feauture_importance_smote.png")


# A comparison of the feature importance plots shows that the important features are drastically different when you train a model with the imbalanced dataset versus the balanced dataset.

# ### Evaluate the models

# In this section, I'd evaluate the two trained models:
# 
# - `model` trained on raw, __imbalanced data__
# - `smote_model` trained on __balanced data__

# #### Compute model metrics
# 
# 1. First defining the function `prediction_to_spark` that performs predictions and converts the prediction results into a Spark DataFrame.

# In[26]:


from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType

def prediction_to_spark(model, test):
    predictions = model.predict(test[feature_cols], num_iteration=model.best_iteration_)
    predictions = tuple(zip(test["Outcome"].tolist(), predictions.tolist()))
    dataColumns = ["Outcome", "prediction"]
    predictions = (
        spark.createDataFrame(data=predictions, schema=dataColumns)
        .withColumn("Outcome", col("Outcome").cast(IntegerType()))
        .withColumn("prediction", col("prediction").cast(DoubleType()))
    )

    return predictions


# 2. Using the `prediction_to_spark` function to perform predictions with the two models `model` and `smote_model`.
# 

# In[27]:


predictions = prediction_to_spark(model, test)
smote_predictions = prediction_to_spark(smote_model, test)
predictions.limit(10).toPandas()


# In[28]:


smote_predictions.limit(10).toPandas()


# 3. Computing metrics for the two models.

# In[29]:


from synapse.ml.train import ComputeModelStatistics

metrics = ComputeModelStatistics(
    evaluationMetric="classification", labelCol="Outcome", scoredLabelsCol="prediction"
).transform(predictions)

smote_metrics = ComputeModelStatistics(
    evaluationMetric="classification", labelCol="Outcome", scoredLabelsCol="prediction"
).transform(smote_predictions)
display(metrics)


# In[30]:


display(smote_metrics)


# #### Evaluate model performance with a confusion matrix
# 
# 1. Using a confusion matrix to summarize the performances of the trained machine learning models on the test data. 

# In[31]:


# Collect confusion matrix value
cm = metrics.select("confusion_matrix").collect()[0][0].toArray()
smote_cm = smote_metrics.select("confusion_matrix").collect()[0][0].toArray()
print(cm)


# In[32]:


print(smote_cm)


# 2. Plotting the confusion matrix for the predictions of `smote_model` (trained on __balanced data__).

# In[33]:


# Plot Confusion Matrix
import seaborn as sns

def plot(cm):
    """
    Plot the confusion matrix.
    """
    sns.set(rc={"figure.figsize": (5, 3.5)})
    ax = sns.heatmap(cm, annot=True, fmt=".20g")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    return ax

with mlflow.start_run(run_id=smote_run.info.run_id):
    ax = plot(smote_cm)
    mlflow.log_figure(ax.figure, "ConfusionMatrix.png")


# 3. Plotting the confusion matrix for the predictions of `model` (trained on raw, __imbalanced data__).

# In[34]:


with mlflow.start_run(run_id=raw_run.info.run_id):
    ax = plot(cm)
    mlflow.log_figure(ax.figure, "ConfusionMatrix.png")


# #### Evaluate model performance with AUC-ROC and AUPRC measures

# 1. Defining a function that returns the AUC-ROC and AUPRC measures.

# In[35]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate(predictions):
    """
    Evaluate the model by computing AUROC and AUPRC with the predictions.
    """

    # Initialize the binary evaluator
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="Outcome")

    _evaluator = lambda metric: evaluator.setMetricName(metric).evaluate(predictions)

    # Calculate AUROC, baseline 0.5
    auroc = _evaluator("areaUnderROC")
    print(f"The AUROC is: {auroc:.4f}")

    # Calculate AUPRC, baseline positive rate (0.172% in the data)
    auprc = _evaluator("areaUnderPR")
    print(f"The AUPRC is: {auprc:.4f}")

    return auroc, auprc


# 2. Logging the AUC-ROC and AUPRC metrics for the model trained on __imbalanced data__.

# In[36]:


with mlflow.start_run(run_id=raw_run.info.run_id):
    auroc, auprc = evaluate(predictions)
    mlflow.log_metrics({"AUPRC": auprc, "AUROC": auroc})
    mlflow.log_params({"Data_Enhancement": "None", "DATA_FILE": "creditcard.csv"})


# 3. Logging the AUC-ROC and AUPRC metrics for the model trained on __balanced data__.

# In[37]:


with mlflow.start_run(run_id=smote_run.info.run_id):
    auroc, auprc = evaluate(smote_predictions)
    mlflow.log_metrics({"AUPRC": auprc, "AUROC": auroc})
    mlflow.log_params({"Data_Enhancement": "SMOTE", "DATA_FILE": "creditcard.csv"})


# The model trained on balanced data returns slightly higher AUC-ROC and AUPRC values compared to the model trained on imbalanced data. Based on these measures, SMOTE appears to be an effective technique for enhancing model performance when working with imbalanced data.

# ## Step 5: Register the models

# Using MLflow to register the two models.

# In[38]:


# Register the model
registered_model_name = "fraud_detection-lightgbm"

raw_model_uri = "runs:/{}/model".format(raw_run.info.run_id)
mlflow.register_model(raw_model_uri, registered_model_name)

smote_model_uri = "runs:/{}/model".format(smote_run.info.run_id)
mlflow.register_model(smote_model_uri, registered_model_name)


# ## Step 6: Save the prediction results

# 1. Loading the better-performing model for batch scoring and generating the prediction results.

# In[40]:


from synapse.ml.predict import MLFlowTransformer

spark.conf.set("spark.synapse.ml.predict.enabled", "true")

model = MLFlowTransformer(
    inputCols=feature_cols,
    outputCol="prediction",
    modelName="fraud_detection-lightgbm",
    modelVersion=2,
)

test_spark = spark.createDataFrame(data=test, schema=test.columns.to_list())

batch_predictions = model.transform(test_spark)


# 2. Save predictions into the lakehouse.

# In[41]:


# Save the predictions into the lakehouse
batch_predictions.write.format("delta").mode("overwrite").save("Files/fraud-detection/predictions/batch_predictions")

