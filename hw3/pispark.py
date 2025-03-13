from pyspark import SparkContext

sc = SparkContext("local", "Pi Approximation")
N = 1000000
delta_x = 1.0 / N

def f(x):
    return 4.0 / (1 + x * x)

x_values = sc.parallelize(range(N))
midpoints = x_values.map(lambda i: f((i + 0.5) * delta_x))
pi_approximation = midpoints.reduce(lambda a, b: a + b) * delta_x
print(f"Approximation of π: {pi_approximation}")
sc.stop()