import matplotlib.pyplot as plt

x = [1000,10000,100000]
y2 = [8.68,56.19,808.60]
y1 = [16.32, 56.78,643.48]

fig, ax = plt.subplots()
ax.plot(x,y1,'x-',label='1 node')
ax.plot(x,y2,'o-',label='2 nodes')
plt.title("Pi by Ray")
plt.xlabel('N')
plt.ylabel("Seconds")
plt.legend()
plt.show()