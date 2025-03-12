import matplotlib.pyplot as plt

x = [1000,10000,100000,1000000]
y1 = [4.90, 5.00, 5.50, 8.20]
y2 = [0.63, 0.65, 0.69, 1.33]

fig, ax = plt.subplots()
ax.plot(x,y1,'x-',label='1 core')
ax.plot(x,y2,'o-',label='2 cores')
plt.title("Pi by Spark")
plt.xlabel('N')
plt.ylabel("Seconds")
plt.legend()
plt.show()