from pyspark import SparkContext

# Initialize SparkContext
sc = SparkContext(appName="AverageSalaryComputation")

# Load data 
file_path = "workers.txt"
data_rdd = sc.textFile(file_path)

# Parse the data into structured format
def parse_line(line):
    fields = line.split("\t")
    name = fields[0]
    department = fields[1]
    gender = fields[2]
    salary = int(fields[3].replace(",", ""))
    return (gender, department, salary)

parsed_rdd = data_rdd.map(parse_line)

# Filter and compute average salaries for each category
def compute_average(rdd, gender, department):
    filtered_rdd = rdd.filter(lambda x: x[0] == gender and x[1] == department)
    count = filtered_rdd.count()
    total_salary = filtered_rdd.map(lambda x: x[2]).sum()
    return total_salary / count if count > 0 else 0

# Compute averages
male_it_avg = compute_average(parsed_rdd, "Male", "IT")
female_it_avg = compute_average(parsed_rdd, "Female", "IT")
male_sales_avg = compute_average(parsed_rdd, "Male", "Sales")
female_sales_avg = compute_average(parsed_rdd, "Female", "Sales")

# Print 
print(f"Average salary for Male IT: {male_it_avg}")
print(f"Average salary for Female IT: {female_it_avg}")
print(f"Average salary for Male Sales: {male_sales_avg}")
print(f"Average salary for Female Sales: {female_sales_avg}")

# Stop 
sc.stop()
