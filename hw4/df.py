from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace

# Initialize Spark Session
spark = SparkSession.builder.appName("AverageSalary").getOrCreate()

# Load JSON into DF
df = spark.read.option("multiline", "true").json("/home/linny/hw4/workers.json")

# Salary to Int
df = df.withColumn("Salary", regexp_replace(col("Salary"), ",", "").cast("int"))

# Group by Gender and Department, then compute the average salary
result = df.groupBy("Gender", "Department").avg("Salary")

# Rename column
result = result.withColumnRenamed("avg(Salary)", "Average_Salary")

# Show the result
result.show()
