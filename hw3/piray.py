import ray
import time

start = time.time()

ray.init()

N = 1000000

delta_x = 1.0 / N

def f(x):
    return 4.0 / (1 + x * x)

@ray.remote
def compute_area(i):
    x_i = (i + 0.5) * delta_x  
    return f(x_i)

tasks = [compute_area.remote(i) for i in range(N)]

pi_approximation = sum(ray.get(tasks)) * delta_x

end = time.time()
duration = end-time
print(f"Approximation of π: {pi_approximation}")

print(f'Time: {duration} seconds')

ray.shutdown()
