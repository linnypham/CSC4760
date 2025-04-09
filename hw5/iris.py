from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

# Step 1: Initialize Spark Session
spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

# Step 2: Load Dataset
data_url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/iris.csv"
iris_df = spark.read.csv(data_url, header=True, inferSchema=True)

# Step 3: Preprocess Data
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
iris_df = assembler.transform(iris_df)
iris_df = iris_df.withColumnRenamed("species", "label")

# Split data into training and testing sets
train_data, test_data = iris_df.randomSplit([0.8, 0.2], seed=42)

# Step 4: Linear SVM Classification
svm = LinearSVC(maxIter=10, regParam=0.1)
svm_model = svm.fit(train_data)
svm_predictions = svm_model.transform(test_data)

# Step 5: Neural Network Classification
layers = [4, 5, 3]  # Input layer (4 features), hidden layer (5 nodes), output layer (3 classes)
mlp = MultilayerPerceptronClassifier(layers=layers, seed=42)
mlp_model = mlp.fit(train_data)
mlp_predictions = mlp_model.transform(test_data)

# Step 6: Evaluate Models
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
svm_accuracy = evaluator.evaluate(svm_predictions)
mlp_accuracy = evaluator.evaluate(mlp_predictions)

print(f"SVM Accuracy: {svm_accuracy}")
print(f"MLP Accuracy: {mlp_accuracy}")

# Save results to file
svm_predictions.select("features", "label", "prediction").write.csv("svm_results.csv")
mlp_predictions.select("features", "label", "prediction").write.csv("mlp_results.csv")

# Visualize Results (Example for SVM)
svm_pandas_df = svm_predictions.toPandas()
plt.scatter(svm_pandas_df["features"].apply(lambda x: x[0]), svm_pandas_df["features"].apply(lambda x: x[1]), c=svm_pandas_df["prediction"])
plt.title("SVM Classification Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
