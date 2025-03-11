import ray

# Initialize Ray
ray.init()

# Number of rectangles (increase N for better accuracy)
N = 1000000

# Width of each rectangle
delta_x = 1.0 / N

# Function to calculate f(x)
def f(x):
    return 4.0 / (1 + x * x)

# Define a task for parallel computation
@ray.remote
def compute_area(i):
    x_i = (i + 0.5) * delta_x  # Midpoint of the interval
    return f(x_i)

# Create a list of tasks to compute areas in parallel
tasks = [compute_area.remote(i) for i in range(N)]

# Gather results from all tasks and sum them up
pi_approximation = sum(ray.get(tasks)) * delta_x

print(f"Approximation of π: {pi_approximation}")

# Shutdown Ray
ray.shutdown()
