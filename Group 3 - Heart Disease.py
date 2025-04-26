# Databricks notebook source
# MAGIC %md
# MAGIC # Heart Disease Prediction

# COMMAND ----------

# File location and type       
file_location = "/FileStore/tables/heart_disease_uci.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
HD_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(HD_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### # Column Descriptions:
# MAGIC **id ** :  Unique id for each patient ** age** : (Age of the patient in years)      **sex** (Male/Female)
# MAGIC **cp** : chest pain type        **trestbps** resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
# MAGIC **chol** : (serum cholesterol in mg/dl)     **fbs ** : (if fasting blood sugar > 120 mg/dl)       **restecg** (resting electrocardiographic results)
# MAGIC **thalach** : maximum heart rate achieved        **exang** : exercise-induced angina (True/ False)          **oldpeak** : ST depression induced by exercise relative to rest
# MAGIC **slope** : the slope of the peak exercise ST segment        **ca**: number of major vessels (0-3) colored by fluoroscopy          **thal** : [normal; fixed defect; reversible defect]     **num** : the predicted attribute
# MAGIC

# COMMAND ----------

#Imports the libraries ;  needed for machine learning (pyspark.ml) and data manipulation (pyspark.sql).
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnull, sum, round
from pyspark.ml.feature import StringIndexer, OneHotEncoder, PCA, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# COMMAND ----------

# # Display the first 5 rows of the DataFrame
HD_df.show(5)



# COMMAND ----------

# MAGIC %md
# MAGIC T**here are 920 patient records** . cp, fbs, sex and examg are categorical and are stored as objects, these will be catgorically encoded there are columns that have null values. columns with more than 50% missing values will be dropped

# COMMAND ----------

# Dropping 'id' and 'dataset' columns as they are not useful for prediction
HD_df = HD_df.drop("id", "dataset")
HD_df.show(5)

# COMMAND ----------

# Replacing invalid physiological values (0) in 'trestbps', 'chol' with NaN

# Replace 0 with NaN in 'trestbps' column
HD_df = HD_df.withColumn("trestbps", when(col("trestbps") == 0, None).otherwise(col("trestbps")))

# Replace 0 with NaN in 'chol' column
HD_df = HD_df.withColumn("chol", when(col("chol") == 0, None).otherwise(col("chol")))

# Display the updated DataFrame
HD_df.show(5)


# COMMAND ----------

# Calculate the number of missing values (nulls) for each column
missing_values = HD_df.select([sum(col(column).isNull().cast("int")).alias(column) for column in HD_df.columns])

# Show the result
missing_values.show()


# COMMAND ----------


# Calculate percentage of missing values 

# Total number of rows in the DataFrame
total_rows = HD_df.count()

# Calculate % for each column with two decimals
missing_percentage = HD_df.select([
    round((sum(col(column).isNull().cast("int")) / total_rows) * 100, 2).alias(column) 
    for column in HD_df.columns
])

# Show the result
missing_percentage.show()


# COMMAND ----------

# Handling missing values by dropping columns with more than 50% missing values

# Total number of rows in the DataFrame
total_rows = HD_df.count()

# Identify columns with more than 50% missing values
columns_to_keep = [
    column for column in HD_df.columns
    if HD_df.select((sum(col(column).isNull().cast("int")) / total_rows * 100).alias(column)).collect()[0][0] <= 50
]

# Select only the columns to keep
HD_df = HD_df.select(*columns_to_keep)

# Display the updated DataFrame schema
HD_df.printSchema()


# COMMAND ----------

# Count missing values in each column 
missing_counts = HD_df.select([
    sum(col(column).isNull().cast("int")).alias(column)
    for column in HD_df.columns
])

# Display the result
missing_counts.show()

# COMMAND ----------

# For remaining missing values: use median for numeric and mode for categorical

from pyspark.sql.functions import col, lit, expr

# Handle missing values column by column
for column in HD_df.columns:
    column_type = HD_df.select(column).dtypes[0][1]  # Get column data type

    if column_type == 'string':  # Handle categorical columns
        # Calculate mode for categorical column
        mode_value = HD_df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
        # Fill missing values with mode
        HD_df = HD_df.fillna({column: mode_value})
    elif column_type in ['int', 'double', 'float']:  # Handle numeric columns
        # Calculate median for numeric column
        median_value = HD_df.approxQuantile(column, [0.5], 0.01)[0]
        # Fill missing values with median
        HD_df = HD_df.fillna({column: median_value})
    elif column_type == 'boolean':  # Handle boolean columns
        # Fill missing values in boolean columns with the most frequent value (mode)
        mode_value = HD_df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
        HD_df = HD_df.fillna({column: mode_value})


# COMMAND ----------

# Encoding categorical variables

from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

# Step 1: Identify all categorical columns (string and boolean)
cat_cols = [field.name for field in HD_df.schema.fields if field.dataType.simpleString() in ['string', 'boolean']]

# Step 2: Cast Boolean columns to String
for col_name in cat_cols:
    if HD_df.schema[col_name].dataType.simpleString() == 'boolean':
        HD_df = HD_df.withColumn(col_name, col(col_name).cast("string"))

# Step 3: Apply StringIndexer to all categorical columns and replace them, (e.g., "male" → 0, "female" → 1).
for col_name in cat_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_idx")
    HD_df = indexer.fit(HD_df).transform(HD_df).drop(col_name).withColumnRenamed(col_name + "_idx", col_name)

# Step 4: View result
HD_df.show(5)



# COMMAND ----------

# Normalizing numerical features

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

# List of numerical columns to normalize
num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']

for col_name in num_cols:
    # Assemble column into vector
    assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vec")
    HD_df = assembler.transform(HD_df)

    # Apply StandardScaler to scale the data
    scaler = StandardScaler(inputCol=f"{col_name}_vec", outputCol=f"{col_name}_scaled", withMean=True, withStd=True)
    scaler_model = scaler.fit(HD_df)
    HD_df = scaler_model.transform(HD_df)

    # Extract scalar value from scaled vector and overwrite original column.Converts the scaled vector back into a single numeric value, replacing the original column with its normalized version.
    HD_df = HD_df.withColumn(col_name, vector_to_array(col(f"{col_name}_scaled"))[0])

    # Drop intermediate columns
    HD_df = HD_df.drop(f"{col_name}_vec", f"{col_name}_scaled")

# Display result
HD_df.show(5)



# COMMAND ----------

# Displaying cleaned dataset
display(HD_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC **Histograms**

# COMMAND ----------

#Data Visualization

import pandas as pd
import seaborn as sns   # sns, .set(), .histplot()
import matplotlib.pyplot as plt     # plt 

# Convert PySpark DataFrame to pandas DataFrame
HD_df_pd = HD_df.toPandas()

# Set visualization style
sns.set(style="whitegrid")

# Plot distributions of numerical features
num_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(num_features):
    sns.histplot(HD_df_pd[col], kde=True, ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Distribution of {col}', fontsize=12)

# Hide unused subplot
axes[-1].axis('off')
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC The plots show that most features are well normalized, with age, trestbps, chol and thalch appearing normally distributed. However, oldpeak is right-skewed, may need transformation.

# COMMAND ----------

# MAGIC %md
# MAGIC **Correlation Analysis of Features**

# COMMAND ----------


# List of all numerical columns
num_cols = HD_df.columns  # Assuming all columns are now numerical

# Calculate correlation matrix using PySpark
correlation_matrix = {}
for col1 in num_cols:
    correlation_matrix[col1] = []
    for col2 in num_cols:
        corr_value = HD_df.stat.corr(col1, col2)  # Compute pairwise correlation
        correlation_matrix[col1].append(corr_value)

# Convert correlation matrix to pandas DataFrame for visualization
correlation_df = pd.DataFrame(correlation_matrix, index=num_cols, columns=num_cols)

# Visualize correlation matrix using seaborn - Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()



# COMMAND ----------

# Create a view or table - for sql 

temp_table_name = "heartdisease"
HD_df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /*To view the table - testing */
# MAGIC
# MAGIC select * from `heartdisease` limit 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

from pyspark.sql.functions import when, col  # MJ 

#Binarizing the Target Variable , (1 = disease present) 
HD_df = HD_df.withColumn("target", when(col("num") > 0, 1).otherwise(0))

#Create age group from scaled Z-scores
HD_df = HD_df.withColumn(
    "age_group",
    when(col("age") <= -1.2, "<45")
    .when((col("age") > -1.2) & (col("age") <= -0.6), "45-50")
    .when((col("age") > -0.6) & (col("age") <= 0.0), "51-55")
    .when((col("age") > 0.0) & (col("age") <= 0.6), "56-60")
    .otherwise("60+")
)

# Register for SQL use
HD_df.createOrReplaceTempView("heartdisease")

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Target variable was binarized. Any value of "num" greater than 0 was labed as 1 (disease present) and otherwise 0.
# MAGIC 2. Created a new categorical group "age_group". This was to capture potential nonlinear relationships between age and heart disease risk. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL Queries

# COMMAND ----------

# MAGIC %md
# MAGIC **1. Group Patients by Age and Calculate the Ratio of Heart Disease Diagnosis**

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC ---Heart Disease Summary by Age Group
# MAGIC SELECT 
# MAGIC   age_group,
# MAGIC   CASE 
# MAGIC     WHEN age_group = '<45' THEN 1
# MAGIC     WHEN age_group = '45-50' THEN 2
# MAGIC     WHEN age_group = '51-55' THEN 3
# MAGIC     WHEN age_group = '56-60' THEN 4
# MAGIC     WHEN age_group = '60+' THEN 5
# MAGIC   END AS age_group_order,
# MAGIC   COUNT(*) AS total_patients,
# MAGIC   SUM(target) AS heart_disease_cases,
# MAGIC   ROUND(SUM(target) / COUNT(*) * 100, 2) AS disease_ratio_percent,
# MAGIC   ROUND(AVG(chol), 2) AS avg_cholesterol_scaled,
# MAGIC   ROUND(AVG(trestbps), 2) AS avg_resting_bp_scaled
# MAGIC FROM heartdisease
# MAGIC WHERE age_group IS NOT NULL
# MAGIC GROUP BY age_group
# MAGIC ORDER BY age_group_order
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **2. Relationship Between Cholesterol Levels and Heart Disease**

# COMMAND ----------

# MAGIC %sql
# MAGIC --Relationship Between Cholesterol Levels and Heart Disease 
# MAGIC SELECT 
# MAGIC   CASE 
# MAGIC     WHEN chol <= -1.0 THEN 'Very Low'
# MAGIC     WHEN chol > -1.0 AND chol <= -0.3 THEN 'Low'
# MAGIC     WHEN chol > -0.3 AND chol <= 0.3 THEN 'Medium'
# MAGIC     WHEN chol > 0.3 AND chol <= 1.0 THEN 'High'
# MAGIC     ELSE 'Very High'
# MAGIC   END AS chol_group,
# MAGIC
# MAGIC   COUNT(*) AS total_patients,
# MAGIC   SUM(target) AS heart_disease_cases,
# MAGIC   ROUND(SUM(target) / COUNT(*) * 100, 2) AS disease_ratio_percent,
# MAGIC   ROUND(AVG(age), 2) AS avg_scaled_age,
# MAGIC   ROUND(AVG(trestbps), 2) AS avg_resting_bp_scaled,
# MAGIC   ROUND(AVG(thalch), 2) AS avg_max_hr_scaled
# MAGIC
# MAGIC FROM heartdisease
# MAGIC WHERE chol IS NOT NULL
# MAGIC GROUP BY 
# MAGIC   CASE 
# MAGIC     WHEN chol <= -1.0 THEN 'Very Low'
# MAGIC     WHEN chol > -1.0 AND chol <= -0.3 THEN 'Low'
# MAGIC     WHEN chol > -0.3 AND chol <= 0.3 THEN 'Medium'
# MAGIC     WHEN chol > 0.3 AND chol <= 1.0 THEN 'High'
# MAGIC     ELSE 'Very High'
# MAGIC   END
# MAGIC ORDER BY disease_ratio_percent DESC

# COMMAND ----------

# MAGIC %md
# MAGIC **3. Top 10 Predictive Risk Combinations**

# COMMAND ----------

# MAGIC %sql
# MAGIC --Top 10 Predictive Risk Combinations 
# MAGIC
# MAGIC SELECT 
# MAGIC   sex, 
# MAGIC   cp, 
# MAGIC   fbs, 
# MAGIC   exang,
# MAGIC   COUNT(*) AS patients,
# MAGIC   SUM(target) AS disease_cases,
# MAGIC   ROUND(SUM(target) / COUNT(*) * 100, 2) AS likelihood
# MAGIC FROM heartdisease
# MAGIC GROUP BY sex, cp, fbs, exang
# MAGIC ORDER BY likelihood DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC **4.  Gender-Based Comparison of Heart Disease Prevalence**
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Gender-Based Comparison of Heart Disease Prevalence
# MAGIC SELECT 
# MAGIC   CASE WHEN sex = 0 THEN 'Male' ELSE 'Female' END AS gender,
# MAGIC   COUNT(*) AS total, 
# MAGIC   SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) AS with_disease,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS percentage
# MAGIC FROM heartdisease
# MAGIC GROUP BY sex;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **5. Heart Disease Rate by Chest Pain Type**

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Heart Disease Rate by Chest Pain Type
# MAGIC SELECT 
# MAGIC   CASE 
# MAGIC     WHEN cp = 0 THEN 'Asymptomatic'
# MAGIC     WHEN cp = 1 THEN 'Non-Anginal Pain'
# MAGIC     WHEN cp = 2 THEN 'Atypical Angina'
# MAGIC     WHEN cp = 3 THEN 'Typical Angina'
# MAGIC   END AS chest_pain_type,
# MAGIC   COUNT(*) AS total,
# MAGIC   SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) AS with_disease,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS percent_with_disease
# MAGIC FROM heartdisease
# MAGIC GROUP BY cp
# MAGIC ORDER BY percent_with_disease DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC **6. Chest Pain Type Distribution by Gender**

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Chest Pain Type Distribution by Gender
# MAGIC SELECT 
# MAGIC   CASE WHEN sex = 0 THEN 'Male' ELSE 'Female' END AS gender,
# MAGIC   CASE 
# MAGIC     WHEN cp = 0 THEN 'Asymptomatic'
# MAGIC     WHEN cp = 1 THEN 'Non-Anginal Pain'
# MAGIC     WHEN cp = 2 THEN 'Atypical Angina'
# MAGIC     WHEN cp = 3 THEN 'Typical Angina'
# MAGIC   END AS chest_pain_type,
# MAGIC   COUNT(*) AS total
# MAGIC FROM heartdisease
# MAGIC GROUP BY sex, cp
# MAGIC ORDER BY gender, chest_pain_type;

# COMMAND ----------

# MAGIC %md
# MAGIC **7. Top 10 Patients with Highest Cholesterol**

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top 10 Patients with highest cholesterol
# MAGIC SELECT  
# MAGIC   CASE WHEN sex = 0 THEN 'Male' ELSE 'Female' END AS gender,
# MAGIC   chol
# MAGIC FROM heartdisease
# MAGIC ORDER BY chol DESC
# MAGIC LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **8. Average Resting BP per Chest Pain Type**

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Average Resting BP per Chest Pain Type
# MAGIC SELECT 
# MAGIC   CASE 
# MAGIC     WHEN cp = 0 THEN 'Asymptomatic'
# MAGIC     WHEN cp = 1 THEN 'Non-Anginal Pain'
# MAGIC     WHEN cp = 2 THEN 'Atypical Angina'
# MAGIC     ELSE 'Typical Angina'
# MAGIC   END AS chest_pain_type,
# MAGIC   ROUND(AVG(trestbps), 2) AS avg_resting_bp
# MAGIC FROM heartdisease
# MAGIC GROUP BY cp;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Machine Learning Models

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# 1. Define input features
valid_features = [c for c in HD_df.columns if c not in ['target', 'num', 'age_group', 'chol_group']]
assembler = VectorAssembler(inputCols=valid_features, outputCol="assembled_features")
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")

# 2. Split into train and test sets
train_data, test_data = HD_df.randomSplit([0.8, 0.2], seed=42)

# 3. Define models
lr = LogisticRegression(labelCol="target", featuresCol="features")
rf = RandomForestClassifier(labelCol="target", featuresCol="features")
mlp = MultilayerPerceptronClassifier(labelCol="target", featuresCol="features", layers=[len(valid_features), 10, 5, 2], maxIter=100)
gbt = GBTClassifier(labelCol="target", featuresCol="features")

# 4. Pipelines
def build_pipeline(model):
    return Pipeline(stages=[assembler, scaler, model])

lr_pipeline = build_pipeline(lr)
rf_pipeline = build_pipeline(rf)
mlp_pipeline = build_pipeline(mlp)
gbt_pipeline = build_pipeline(gbt)

# 5. Train models
lr_model = lr_pipeline.fit(train_data)
rf_model = rf_pipeline.fit(train_data)
mlp_model = mlp_pipeline.fit(train_data)
gbt_model = gbt_pipeline.fit(train_data)

# 6. Predict on test set
lr_preds = lr_model.transform(test_data)
rf_preds = rf_model.transform(test_data)
mlp_preds = mlp_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)

# 7. Evaluate AUC
evaluator = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
lr_auc = evaluator.evaluate(lr_preds)
rf_auc = evaluator.evaluate(rf_preds)
mlp_auc = evaluator.evaluate(mlp_preds)
gbt_auc = evaluator.evaluate(gbt_preds)

# 8. Print results
print("🔍 Model Evaluation on Test Set:")
print(f"Logistic Regression AUC:        {lr_auc:.4f}")
print(f"Random Forest (Binary) AUC:     {rf_auc:.4f}")
print(f"Multilayer Perceptron AUC:      {mlp_auc:.4f}")
print(f"Gradient Boosted Trees AUC:     {gbt_auc:.4f}")


# COMMAND ----------

# Decision tree clasifier model, Random forest classifier model. These models will predict Heart disease severity (0-4 score)
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Use the same features and assembler from before
valid_features = [c for c in HD_df.columns if c not in ['target', 'num', 'age_group', 'chol_group']]
assembler = VectorAssembler(inputCols=valid_features, outputCol="features")

# Define multiclass models (target = 'num' 0 to 4)
dt = DecisionTreeClassifier(labelCol="num", featuresCol="features")
rf_multi = RandomForestClassifier(labelCol="num", featuresCol="features")

# Pipelines
dt_pipeline = Pipeline(stages=[assembler, dt])
rf_multi_pipeline = Pipeline(stages=[assembler, rf_multi])

# Train
dt_model = dt_pipeline.fit(train_data)
rf_multi_model = rf_multi_pipeline.fit(train_data)

# Predict
dt_preds = dt_model.transform(test_data)
rf_multi_preds = rf_multi_model.transform(test_data)

# Evaluate using accuracy
multi_eval = MulticlassClassificationEvaluator(labelCol="num", metricName="accuracy")
dt_acc = multi_eval.evaluate(dt_preds)
rf_multi_acc = multi_eval.evaluate(rf_multi_preds)

# Print results
print(f" Decision Tree Accuracy (multiclass): {dt_acc:.4f}")
print(f" Random Forest Accuracy (multiclass): {rf_multi_acc:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC **Model Comparison**

# COMMAND ----------

# Comparison of models
import matplotlib.pyplot as plt

# Scores from all 6 models (binary and multiclass)
models = [
    "Logistic Regression", 
    "Random Forest (binary)", 
    "Multilayer Perceptron", 
    "Gradient-Boosted Trees", 
    "Decision Tree (multiclass)", 
    "Random Forest (multiclass)"
]

# Replace the scores with your actual AUC values from evaluation
scores = [0.8435, 0.8317, 0.7244, 0.7996, 0.4295, 0.4966]  # Example AUCs
colors = ['skyblue', 'lightgreen', 'orchid', 'goldenrod', 'sandybrown', 'forestgreen']

plt.figure(figsize=(12, 6))
bars = plt.bar(models, scores, color=colors)
plt.ylim(0, 1)
plt.ylabel("Score (AUC or Accuracy)")
plt.title("🔍 Model Performance Comparison")

# Annotate scores on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{score:.4f}", ha='center', fontsize=12, weight='bold')

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Random Forest Classifier**

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Access feature importances
rf_model_only = rf_model.stages[-1]  # the RandomForestClassifier in the pipeline
importances = rf_model_only.featureImportances.toArray()

# Match importances to features
feature_importance_df = pd.DataFrame({
    'Feature': valid_features,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='green')
plt.gca().invert_yaxis()
plt.title("📈 Feature Importance (Random Forest - Binary Classification)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()