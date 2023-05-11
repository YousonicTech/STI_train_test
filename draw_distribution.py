import numpy as np
import matplotlib.pyplot as plt
import numpy as np
path="add_data_distribution.npy"
path2="data_distribution.npy"
datas = np.load(path,allow_pickle=True).item()
datas=sorted(datas.items(), key=lambda x: x[0])

datas2 = np.load(path2,allow_pickle=True).item()
datas2 = sorted(datas2.items(), key=lambda x: x[0])


total=0
for datai in datas:
    #统计value的总和
    print(datai[1])
    total=total+datai[1]
print(total)

total=0
for datai in datas2:
    #统计value的总和
    total=total+datai[1]
print(total)


'''data_round=sorted(data_round.items(), key=lambda x: x[0])

print(data_round)
data_round=np.array(data_round)
x_val=data_round[:,0]
y_val=data_round[:,1]

print(x_val)
print(y_val)

plt.plot(x_val,y_val)
#plt.scatter(x_val, y_val)

plt.title('train distribution')
plt.xlabel('sti')
plt.ylabel('num')

plt.savefig("round_distribution.png")'''

