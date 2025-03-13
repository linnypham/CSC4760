import matplotlib.pyplot as plt

x = [1000,10000,100000]
y1 = [48.70,56.78,643.48]
y2 = [9.57, 45.61,958.69]

fig, ax = plt.subplots()
ax.plot(x,y1,'x-',label='1 node')
ax.plot(x,y2,'o-',label='2 nodes')
plt.title("Pi by Ray")
plt.xlabel('N')
plt.ylabel("Seconds")
plt.legend()
plt.show()