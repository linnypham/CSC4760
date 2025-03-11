from pyspark import SparkContext

# Initialize Spark context
sc = SparkContext("local", "Pi Approximation")

# Number of rectangles (increase N for better accuracy)
N = 1000000

# Width of each rectangle
delta_x = 1.0 / N

# Function to calculate f(x)
def f(x):
    return 4.0 / (1 + x * x)

# Create an RDD representing indices of rectangles
x_values = sc.parallelize(range(N))

# Calculate midpoints and apply the function f(x)
midpoints = x_values.map(lambda i: f((i + 0.5) * delta_x))

# Sum up the areas of rectangles
pi_approximation = midpoints.reduce(lambda a, b: a + b) * delta_x

print(f"Approximation of π: {pi_approximation}")

# Stop the Spark context
sc.stop()
