import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import confusion_matrix, classification_report


x=pd.read_csv("betaValues5000all.csv", sep=";",header=None)


fake=pd.read_csv("fake81_500_PTPR,A_last_sig_new.csv", sep=";",header=None) # TODO: HIER AUS ÄNDERN



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print("principalComponents-------------")
print(principalComponents)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
print("principalDf------------")
print(principalDf)



Dx=pd.read_csv("Dx.csv",sep=";",header=None)
Dx.columns = ['Dx']
print("Dx---------------")
print(Dx)


  

finalDf = pd.concat([principalDf,Dx], axis = 1)
print("finalDf------------")
print(finalDf)
finalDf.to_csv("finalpca.csv",sep= ";",header=None,index=False)
  

"""
def centroid(PCA1, PCA2):
    #_x_list = [vertex [0] for vertex in vertexes]
    #_y_list = [vertex [1] for vertex in vertexes]
    _len = len(PC1)
    _x = sum(PCA1) / _len
    _y = sum(PCA2) / _len
    return(_x, _y)

df_grouped = finalDf.groupby('Dx')


# iterate over each group
for group_name, df_group in df_grouped:
    print ("group_name ===", group_name)
    print (df_group)

    # calcualte the centroid of each class label
    print (df_group["principal component 1"].values)
    print (df_group["principal component 2"].values)

    PCA1 = df_group["principal component 1"].values
    PCA2 = df_group["principal component 2"].values

    print ("---------------------------------\n")
    exit()

exit()
"""
########################################
################## KNN #################
########################################
# tumor names to class label
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Dx['Dx_num'] = le.fit_transform(Dx['Dx'])
print (Dx)


# KNN on PCA1 and PCA2
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)

X = principalDf.values
y = Dx['Dx_num'].values
neigh.fit(X, y)


test_first_line = principalDf.iloc[0, :].values
print (test_first_line)

print(neigh.predict([test_first_line]))
########################################
################## KNN #################
########################################



# all colors
unique_colour=Dx.iloc[:,0].unique()
print("unique_colour----------")
print(unique_colour)
print("len(unique_colour--------------")
print(len(unique_colour))
#colors = [np.random.rand(3,) for i in range(len(unique_colour))]
colors = ["red" if i == "b'PTPR, A'" else "gray" for i in unique_colour] # TODO: "b'GBM, RTK II'"
print("colors------")
print(colors)
print(len(colors))

data_color = []
for i, j in zip(unique_colour,colors):
    print ("%20s, %5s" % (i, j))
    data_color.append([i, j])
    
df_colors = pd.DataFrame(data_color, columns = ['unique_colour', 'colors'])
df_colors



targets=Dx.to_numpy()
print("targets------")
print(targets.shape)
print(targets)




fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)


print ("targets,colors == ", targets.shape,np.array(colors).shape)

INDEX = 1
for target in targets:
    print ("target == ", target)
    color = df_colors[df_colors['unique_colour'] == target[0]]["colors"]#[0]
    print ("----", type(color), color)
    color = color.values
    print (color)

    print ("----------------------------------")
    print(target)
    print (color)
    print (INDEX)
    print ("----------------------------------")
    
    #print (finalDf.iloc[:,-1])
    np.where(finalDf.iloc[:,-1] == target[0])
    indicesToKeep = finalDf.iloc[:,-1] == target[0]
    print (indicesToKeep.sum())
    
    #print (len(finalDf.loc[indicesToKeep, 'principal component 1']))
    
    #print("color----------------------------------------------------------- ", color)
    #print("finalDf.iloc[:,-1]------")
    #print(finalDf.iloc[:,-1])
    #print("indicesToKeep.shape------------")
    #print(indicesToKeep.shape)
    #print(finalDf.loc[indicesToKeep, 'principal component 1'].shape)
    
    if color == "red":
        print ("NUM OF POINTS ================= ", print (indicesToKeep.sum()))
        
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], color = "red", s = 50)
    else:
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], color = "blue", s = 50, marker="+")
    INDEX = INDEX + 1
    
    print ("********************")
#fake=fake.T    #wenn man mehr als ein Punkt haben will
fakeresult=pca.transform(fake)
print("fakeresult.shape-----------")
print(fakeresult.shape)
print(fakeresult)


KNN_pred_result = neigh.predict(fakeresult)
print("KNN_pred_result === ", KNN_pred_result)
y_pred = list(le.inverse_transform(KNN_pred_result))
print("y_pred === ", y_pred)

y_true = ["b'PTPR,A'"]  #name des labels
labels = list(set(y_true + y_pred))  #["ant", "bird", "cat"]
conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
class_report = classification_report(y_true,y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print ("conf_mat")
print (conf_mat)
print ("classification_report")
print (class_report)
print ("tn, fp, fn, tp")
print (tn, fp, fn, tp)

for i in range(fakeresult.shape[0]):
    print ("POSITION ====================== ", fakeresult[i,0],fakeresult[i,1])
    ax.scatter(fakeresult[i,0],fakeresult[i,1],c="green",s=300,marker="*")

ax.grid()
#ax.legend(y.iloc[:,0].unique())
ax.legend(Dx.iloc[:,0].unique(),loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("KNN_PCA_fakeclass: ")
