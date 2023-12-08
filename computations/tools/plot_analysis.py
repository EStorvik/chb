import matplotlib.pyplot as plt

import numpy as np

path = "/home/erlend/src/fenics/output/chb/halfnhalf/threewaysplit/"

data = np.empty(2)

with open(path+"log/gamma5_time.txt", "r") as f:
    c = 0
    for line in f.readlines():
        c+=1
        #print(line)
        line = line.rstrip()
        line = line.replace("(", "")
        line = line.replace(")", "")
        sline = line.split(",")
        if c<199:
            data = np.vstack([data, np.array([float(sline[0]), float(sline[1])])])
        
plt.figure()
plt.plot(data[:,0], data[:, 1])
plt.show()

# print(array_from_file)