from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("KMeansClustering").getOrCreate()

# 1. Load dataset
dataset = spark.read.format("libsvm").load("kmeans_input.txt")

# 2. Train K-Means model
kmeans = KMeans(featuresCol="features", k=2)
model = kmeans.fit(dataset)

# 3. Get cluster centers
centers = model.clusterCenters()
center1 = centers[0]
center2 = centers[1]
print(f'Center 1: {center1}')
print(f'Center 2: {center2}')

# 4. Predict clusters
predictions = model.transform(dataset)
pdf = predictions.toPandas()

# 5. Visualize results
plt.figure(figsize=(10, 6))

# Plot data points
cluster1 = pdf[pdf['prediction'] == 0]
cluster2 = pdf[pdf['prediction'] == 1]

plt.scatter(cluster1['features'].apply(lambda x: x[0]), 
            cluster1['features'].apply(lambda x: x[1]), 
            c='red', marker='x', label='Cluster 1')

plt.scatter(cluster2['features'].apply(lambda x: x[0]), 
            cluster2['features'].apply(lambda x: x[1]), 
            c='blue', marker='o', label='Cluster 2')

# Plot cluster centers
plt.scatter(center1[0], center1[1], s=150, 
           c='red', marker='^', edgecolor='black', label='Center 1')
plt.scatter(center2[0], center2[1], s=150,
           c='blue', marker='s', edgecolor='black', label='Center 2')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Results (k=2)')
plt.legend()
plt.grid(True)
plt.savefig("kmeans_graph.png")
plt.show()

# Stop Spark session
spark.stop()
