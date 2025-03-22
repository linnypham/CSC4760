from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, avg

# Initialize Spark session
spark = SparkSession.builder.appName("AverageSalaryComputation").getOrCreate()

# Load JSON data into DataFrame
df = spark.read.json("workers.json")

# Convert Salary column to numeric (remove commas and cast to integer)
df = df.withColumn("Salary", regexp_replace(col("Salary"), ",", "").cast("int"))

# Compute average salary grouped by Gender and Department
avg_salary_df = df.groupBy("Gender", "Department").agg(avg("Salary").alias("Avg_Salary"))

# Show results
avg_salary_df.show()
