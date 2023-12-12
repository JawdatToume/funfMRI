import sys
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np
import os

file = sys.argv[1]
file2 = sys.argv[2]

reader = open(file, 'r')
allData = reader.readlines()
reader2 = open(file2, 'r')
allData2 = reader2.readlines()
data = {}
xData = []
yData = []
finalData = []
k = 0

# for file in os.listdir("Results"):
#     print(file)
#     yData.append(k)
#     finalData.append([])
#     reader = open(os.path.join("Results",file), 'r')
#     allData = reader.readlines()[1:]
#     for i in range(len(allData)):
#         xData.append([])
#         #yData.append([])
#         allData[i] = allData[i].split(',')[2:]
#         #allData2[i] = allData2[i].split(',')[2:]
#         for j in range(len(allData[i])):
#             if allData[i][j] == '':# or allData2[i][j] == '':
#                 #allData[i][j] = 0
#                 continue
#             xData[i].append(float(allData[i][j]))
#             #yData[i].append(float(allData2[i][j]))
#         finalData[k].append(np.mean(xData[i]))
#     k+=1

for i in range(len(allData)):
    xData.append([])
    yData.append([])
    allData[i] = allData[i].split(',')[2:]
    allData2[i] = allData2[i].split(',')[2:]
    for j in range(len(allData[i])):
        if allData[i][j] == '' or allData[i][j] == 'nan' or allData2[i][j] == '':
            #allData[i][j] = 0
            continue
        try:
            xData[i].append(float(allData[i][j]))
            yData[i].append(float(allData2[i][j]))
        except:
            pass
    print(allData[i][1])
print(allData)

#Regression Plots
# plots = []
# for i in range(1,9):#[1,4,8]:
#     data["x%s" %(str(i))] = np.array(xData[i])
#     data["y%s" %(str(i))] = np.array(yData[i])
#     plots.append(sns.regplot(data, x="x%s" %(str(i)), y="y%s" %(str(i)), label="CNN%s" %(str(i))))
#     plots[0].legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
#     sns.move_legend(plots[0], "upper left", bbox_to_anchor=(1, 1))

# plt.title("Across subject prediction")
# plt.xlabel("Correlation when Trained on averaged data from Subjects 1-4")
# plt.ylabel("Correlation when Trained on averaged data from Subject 1")

#Histogram Plots
# plots = []
colours = ['pink','red','orange','yellow','green','blue','indigo','violet']
cols = colours[:len(xData)]
data["x1"] = np.array([*range(1,len(xData),1)])
xDataMean = [np.mean(xData[i]) for i in range(1,len(xData))]
xDataSTD = [np.std(xData[i]) for i in range(1,len(xData))]
data["y1"] = np.array(xDataMean)
#sdata["yfull"] = np.array(xData)
print(xData[2])
print(xDataMean)
print(xDataSTD)
data["std"] = np.array(xDataSTD)
#print(data["yfull"].shape)
plots = sns.barplot(data, x="x1", y="y1", errorbar=("ci",95), palette=cols)
errorLength = [0.95*(xDataSTD[i])/(np.sqrt(len(xData[i+1]))) for i in range(len(xDataMean))]
print(errorLength)
data["std"] = np.array(errorLength).T
bars = plots.errorbar(data=data, x="x1", y="y1", yerr="std", ls='', lw=3, color='black')#, align='center')for (hue, df_hue), dogde_dist in zip(x.groupby('Subgroup'), np.linspace(-0.4, 0.4, 2 * num_hues + 1)[1::2]):
xys = bars.lines[0].get_xydata()
bars.remove()
plots.errorbar(data=data, x=xys[:, 0] - 1, y='y1', yerr='std', ls='', lw=3, color='black')
#, label="Layer%s" %(str(1))))
#plots[i-1].legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
#sns.move_legend(plots[i-1], "upper left", bbox_to_anchor=(1, 1))

# plt.title("Comparison of prediction when training on 1 subject vs. more than 1 subject")
# plt.ylabel("Correlation when training on just subject 1")
# plt.xlabel("Correlation when training on subjects 1 and 2")
#plt.title(file)
plt.ylabel("Correlation")
plt.xlabel("Layer")
plt.ylim(-0.03,0.03)

plt.savefig(file[:-4]+".png")