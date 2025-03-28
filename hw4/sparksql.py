from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col

# Initialize Spark Session
spark = SparkSession.builder.appName("AverageSalarySQL").getOrCreate()

# Load JSON into DF
df = spark.read.option("multiline", "true").json("workers.json")

# Salary to Int
df = df.withColumn("Salary", regexp_replace(col("Salary"), ",", "").cast("int"))

# Create a temporary SQL table from the DataFrame
df.createOrReplaceTempView("workers")

# Use SparkSQL to compute the average salary for each Gender-Department combination
query = """
SELECT Gender, Department, AVG(Salary) AS Average_Salary
FROM workers
GROUP BY Gender, Department
"""

# Execute the query and store the result in a DataFrame
result = spark.sql(query)

# Show the result
result.show()
