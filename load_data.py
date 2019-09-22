import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

df = pd.read_csv('./data/trajectories-0400-0415.txt', header=None, sep='\s+', skiprows=[0])
df1 = df[df[0] == 1]
# df[3] = (df[3] - min(df[3])) // 100
# df = df.sort_values(by=[0])
# print(df)
# plt.figure(8)
# currentAxis = plt.gca()
# df1 = df[df[0] == 1]
plt.plot(df1[6], df1[7])
plt.show()
# df2=pd.read_csv('./data/trajectory.csv',nrows=100)
# print(df2.columns)
