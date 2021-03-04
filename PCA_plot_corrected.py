import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import confusion_matrix, classification_report


x=pd.read_csv("betaValues5000all.csv", sep=";",header=None)


fake=pd.read_csv("fake81_500_PTPR,A_last_sig_new.csv", sep=";",header=None) # TODO: HIER AUS ÄNDERN
fake=pd.read_csv("fake81_500_PTPR,A_last_sig_new 21.40.13.csv", sep=";",header=None) # TODO: HIER AUS ÄNDERN



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
  


#"""
def centroid(PCA1, PCA2):
    #_x_list = [vertex [0] for vertex in vertexes]
    #_y_list = [vertex [1] for vertex in vertexes]
    _len = len(PCA1)
    _x = sum(PCA1) / _len
    _y = sum(PCA2) / _len
    return(_x, _y)

df_grouped = finalDf.groupby('Dx')


# iterate over each group
centroid_x, centroid_y, centroid_name = [], [], []
for group_name, df_group in df_grouped:

    print ("group_name ===", group_name)
    print (df_group)

    # calcualte the centroid of each class label
    print (df_group["principal component 1"].values)
    print (df_group["principal component 2"].values)

    PCA1 = df_group["principal component 1"].values
    PCA2 = df_group["principal component 2"].values

    centroid_group = centroid(PCA1, PCA2)
    print ("centroid_group ===", centroid_group)
    print ("---------------------------------\n")
    centroid_x.append(centroid_group[0])
    centroid_y.append(centroid_group[1])
    centroid_name.append(group_name)


df_centroids = pd.DataFrame(list(zip(centroid_name, centroid_x, centroid_y)),
    columns=["centroid_name", "centroid_x", "centroid_y"])

print (df_centroids)


"""
# test plot for the centroid
fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(PCA1, PCA2, color = "red", s = 50)
ax.scatter(centroid[0], centroid[1], color = "blue", s = 50, marker="+")

plt.show()
"""






#"""
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
"""


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
        
fakeresult=pca.transform(fake)
print("fakeresult.shape-----------")
print(fakeresult.shape)
print(fakeresult)

def euclidean_distance(points, single_point):
    dist = (points - single_point)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    return dist


def plot_centroids_and_points(
        all_centroids,
        df_centroids,
        y_true_name,
        y_pred,
        y_centroid=None,
    ):

    print ("y_pred ======= ", y_pred)
    print ("y_pred ======= ", y_pred.size)
    print ("y_pred ======= ", len(y_pred))

    fig = plt.figure(figsize = (20,20))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    unique_colour=df_centroids.iloc[:,0].unique()
    print("unique_colour----------")
    print(unique_colour)
    print("len(unique_colour--------------")
    print(len(unique_colour))
    #colors = [np.random.rand(3,) for i in range(len(unique_colour))]
    colors = ["red" if i == y_true_name else "gray" for i in unique_colour] # TODO: "b'GBM, RTK II'"
    print("colors------")
    print(colors)
    print(len(colors))

    data_color = []
    for i, j in zip(unique_colour,colors):
        print ("%20s, %5s" % (i, j))
        data_color.append([i, j])
        
    df_colors = pd.DataFrame(data_color, columns = ['unique_colour', 'colors'])
    df_colors


    INDEX = 1
    for index, row_centroid in df_centroids.iterrows():
        print ("row_centroid === ", row_centroid)
        print ("row_centroid === ", type(row_centroid))
        PCA1 = row_centroid["centroid_x"]
        PCA2 = row_centroid["centroid_y"]
        centroid_name = row_centroid["centroid_name"]
        print ("row_centroid == ", row_centroid)
        color = df_colors[df_colors['unique_colour'] == row_centroid["centroid_name"]]["colors"]#[0]
        print ("----", type(color), color)
        color = color.values
        print (color)

        print ("----------------------------------")
        #print(target)
        print (color)
        print (INDEX)
        print ("----------------------------------")
        
        #print (finalDf.iloc[:,-1])
        #np.where(finalDf.iloc[:,-1] == target[0])
        #indicesToKeep = finalDf.iloc[:,-1] == target[0]
        #print (indicesToKeep.sum())
        
        #print (len(finalDf.loc[indicesToKeep, 'principal component 1']))
        
        #print("color----------------------------------------------------------- ", color)
        #print("finalDf.iloc[:,-1]------")
        #print(finalDf.iloc[:,-1])
        #print("indicesToKeep.shape------------")
        #print(indicesToKeep.shape)
        #print(finalDf.loc[indicesToKeep, 'principal component 1'].shape)
        
        ax.scatter(PCA1, PCA2, color = color, s = 50)

        """
        if color == "red":
            print ("NUM OF POINTS ================= ", print (indicesToKeep.sum()))
            ax.scatter(PCA1, PCA2, color = "red", s = 50)
            #ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], color = "red", s = 50)
        else:
            ax.scatter(PCA1, PCA2, color = "gray", s = 50, marker="+")
            #ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], color = "blue", s = 50, marker="+")
        """
        INDEX = INDEX + 1
        
        print ("********************")

    # plot the prediction point
    if len(y_pred) == 1: # only one point
        ax.scatter(y_pred[0][0], y_pred[0][1], color = "blue", s = 50, marker="+")
    else: # more than one point
        y_pred_x = [i[0] for i in y_pred]
        y_pred_y = [i[1] for i in y_pred]
        ax.scatter(y_pred_x, y_pred_y, color = "blue", s = 50, marker="+")

    # plot centroid point
    print (y_centroid)
    y_centroid_x = y_centroid["centroid_x"]
    y_centroid_y = y_centroid["centroid_y"]
    y_centroid_name = y_centroid["centroid_name"]
    ax.scatter(y_centroid_x, y_centroid_y, color = "blue", s = 50, marker="*")

    plt.show()


targets=Dx.to_numpy()
print("targets------")
print(targets.shape)
print(targets)




all_centroids = df_centroids[['centroid_x', 'centroid_y']].values
print ("fakeresult === ")
print (fakeresult)

distances = euclidean_distance(all_centroids, fakeresult)
#df_centroids['a'].apply(lambda x: x + 1)
print ("distances")
print (distances)
print (np.argmin(distances), min(distances) )
print (df_centroids.iloc[np.argmin(distances)]["centroid_name"])

y_true_name = "b'PTPR, A'"
y_pred = fakeresult
y_pred_name = df_centroids.iloc[np.argmin(distances)]["centroid_name"]
y_centroid = df_centroids.iloc[np.argmin(distances)]#[["centroid_x", "centroid_y"]]

print ("********************************************")
print (y_centroid)
print ("********************************************")


plot_centroids_and_points(
    all_centroids,
    df_centroids,
    y_true_name,
    y_pred,
    y_centroid,
)

"""
KNN_pred_result = neigh.predict(fakeresult)
print("KNN_pred_result === ", KNN_pred_result)
y_pred = list(le.inverse_transform(KNN_pred_result))
print("y_pred === ", y_pred)
"""

y_true = ["b'PTPR,A'"]
y_pred = [y_pred_name]
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
plt.savefig("plot_red.png")
