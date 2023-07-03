from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkConf
import time

start_time = time.time()

# Initialize SparkSession
# spark = SparkSession.builder.appName("ClassificationExample").getOrCreate()
conf = SparkConf().setAppName("classification").setMaster("local[*]")
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# Read the training data CSV files
train_features_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\eye_csv\\pca_X_train - Copy.csv"
train_labels_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\eye_csv\\y_train - Copy.csv"
# train_features_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\braintumor\\pca_vgg19_X_train - Copy.csv"
# train_labels_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\braintumor\\y_train - Copy.csv"
train_features = spark.read.csv(train_features_file, header=True, inferSchema=True)
train_labels = spark.read.csv(train_labels_file, header=True, inferSchema=True)

# Assign explicit aliases to the columns to resolve ambiguity
train_features = train_features.toDF(*[f"feat_{i}" for i in range(len(train_features.columns))])
train_labels = train_labels.withColumnRenamed("_c0", "label")

# Join the features and labels on a common index column
train_data = train_features.join(train_labels, train_features.feat_0 == train_labels.index).drop("feat_0", "index")

# Read the test data CSV files
test_features_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\eye_csv\\pca_X_test - Copy.csv"
test_labels_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\eye_csv\\y_test - Copy.csv"
# test_features_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\braintumor\\pca_vgg19_X_test - Copy.csv"
# test_labels_file = "C:\\uni\\2 - second term\\Cloud Computing\\Project\\braintumor\\y_test - Copy.csv"
test_features = spark.read.csv(test_features_file, header=True, inferSchema=True)
test_labels = spark.read.csv(test_labels_file, header=True, inferSchema=True)

# Assign explicit aliases to the columns to resolve ambiguity
test_features = test_features.toDF(*[f"feat_{i}" for i in range(len(test_features.columns))])
test_labels = test_labels.withColumnRenamed("_c0", "label")

# Join the features and labels on a common index column
test_data = test_features.join(test_labels, test_features.feat_0 == test_labels.index).drop("index")

# Prepare the feature vector column
feature_cols = train_data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

data_preparation_time = time.time() - start_time

#%% Random Forest Classfier
print("Random Forest")

start_model = time.time()

# Create the random forest classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Create a pipeline to chain feature vector assembler and random forest stages
pipeline = Pipeline(stages=[assembler, rf])

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using multiclass classification evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Gradient-Boosted Tree Classifier -- binary classification
print("Gradient-Boosted Tree")

start_model = time.time()

# Create the Gradient-Boosted Tree Classifier
gbt = GBTClassifier(labelCol="label", featuresCol="features")

# Create a pipeline to chain feature vector assembler and random forest stages
pipeline = Pipeline(stages=[assembler, gbt])

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using multiclass classification evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Decision Tree Classifier
print("Decision Tree")

start_model = time.time()

# Create the Decision Tree Classifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# Create a pipeline to chain feature vector assembler and random forest stages
pipeline = Pipeline(stages=[assembler, dt])

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using multiclass classification evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Support Vector Machine Classifier -- binary only
print("Support Vector Machine")

start_model = time.time()

# Create the Support Vector Machine Classifier
svm = LinearSVC(labelCol="label", featuresCol="features")

# Create a pipeline to chain feature vector assembler and random forest stages
pipeline = Pipeline(stages=[assembler, svm])

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using multiclass classification evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Multilayer Perceptron Classifier
print("Multilayer Perceptron")

start_model = time.time()

# Create the Multilayer Perceptron Classifier
layers = [len(feature_cols), 10, 5, 2]  # Customize the hidden layers as per your requirement
mlp = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=layers)

# Create a pipeline to chain feature vector assembler and random forest stages
pipeline = Pipeline(stages=[assembler, mlp])

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using multiclass classification evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Logistic Regression Classifier
print("Logistic Regression")

start_model = time.time()

# Create the Multilayer Perceptron Classifier
lr = LogisticRegression(labelCol="label", featuresCol="features")

# Create a pipeline to chain feature vector assembler and random forest stages
pipeline = Pipeline(stages=[assembler, lr])

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using multiclass classification evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

#%% Naive Bayes Classifier  -- nonnegative only
print("Naive Bayes")

start_model = time.time()

# Create the Naive Bayes Classifier
nb = NaiveBayes(labelCol="label", featuresCol="features")

# Create a pipeline to chain feature vector assembler and random forest stages
pipeline = Pipeline(stages=[assembler, nb])

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using multiclass classification evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

elapsed_time = time.time() - start_model
print("Elapsed time: {:.2f} seconds".format(data_preparation_time + elapsed_time))

