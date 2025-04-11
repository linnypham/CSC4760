from pyspark.sql import SparkSession  
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, MultilayerPerceptronClassifier, OneVsRest
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize Spark
spark = SparkSession.builder \
    .appName("IrisClassification") \
    .getOrCreate()

# 1. Load and prepare data
df = spark.read.csv("iris.csv", header=True, inferSchema=True)
assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="assembled_features"
)

# Add standardization 
scaler = StandardScaler(
    inputCol="assembled_features", 
    outputCol="features",
    withStd=True, 
    withMean=True
)

label_indexer = StringIndexer(inputCol="species", outputCol="label")
train, test = df.randomSplit([0.8, 0.2], seed=42)

# 2. Linear SVM with OneVsRest 
svm = LinearSVC(maxIter=100, regParam=0.01)  
ovr = OneVsRest(classifier=svm)
svm_pipeline = Pipeline(stages=[assembler, scaler, label_indexer, ovr])
svm_model = svm_pipeline.fit(train)
svm_pred = svm_model.transform(test)

# 3. MLP Classifier 
mlp = MultilayerPerceptronClassifier(
    layers=[4, 10, 8, 3], 
    seed=42,
    blockSize=128,
    maxIter=200 
)
mlp_pipeline = Pipeline(stages=[assembler, scaler, label_indexer, mlp])
mlp_model = mlp_pipeline.fit(train)
mlp_pred = mlp_model.transform(test)

# 4. Evaluation
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"SVM Accuracy: {evaluator.evaluate(svm_pred):.2%}")
print(f"MLP Accuracy: {evaluator.evaluate(mlp_pred):.2%}")

# 5. Generate confusion matrices
def create_confusion_matrix(predictions, title):
    # Convert to pandas for easier processing
    pred_df = predictions.select("label", "prediction").toPandas()
    
    # Get class mapping from StringIndexer
    label_mapping = {idx: label for idx, label in enumerate(
        predictions.schema["label"].metadata["ml_attr"]["vals"]
    )}
    
    # Create confusion matrix
    cm = np.zeros((3, 3))
    for i in range(len(pred_df)):
        true_label = int(pred_df.iloc[i, 0])
        pred_label = int(pred_df.iloc[i, 1])
        cm[true_label, pred_label] += 1
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues if 'SVM' in title else plt.cm.Reds)
    plt.title(title)
    plt.colorbar()
    
    class_names = [label_mapping[i] for i in range(3)]
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), 'd'), 
                    horizontalalignment="center", color="white" if cm[i, j] > 5 else "black")
    
    plt.tight_layout()
    return plt

# Create and save both confusion matrices
create_confusion_matrix(svm_pred, "SVM Results").savefig("svm_confusion.png")
create_confusion_matrix(mlp_pred, "MLP Results").savefig("mlp_confusion.png")

spark.stop()
