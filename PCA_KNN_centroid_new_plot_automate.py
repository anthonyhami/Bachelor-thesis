import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import confusion_matrix, classification_report


# """
def centroid(PCA1, PCA2):
    # _x_list = [vertex [0] for vertex in vertexes]
    # _y_list = [vertex [1] for vertex in vertexes]
    _len = len(PCA1)
    _x = sum(PCA1) / _len
    _y = sum(PCA2) / _len
    return (_x, _y)


def return_df_centroids(df_grouped):

    # iterate over each group
    centroid_x, centroid_y, centroid_name = [], [], []
    for group_name, df_group in df_grouped:

        print("group_name ===", group_name)
        print(df_group)

        # calcualte the centroid of each class label
        print(df_group["principal component 1"].values)
        print(df_group["principal component 2"].values)

        PCA1 = df_group["principal component 1"].values
        PCA2 = df_group["principal component 2"].values

        centroid_group = centroid(PCA1, PCA2)
        print("centroid_group ===", centroid_group)
        print("---------------------------------\n")
        centroid_x.append(centroid_group[0])
        centroid_y.append(centroid_group[1])
        centroid_name.append(group_name)
    return centroid_x, centroid_y, centroid_name


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


# """
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


def plot_all_points(
    Dx, finalDf, y_true_name, fakeresult, class_index,
):

    # all colors
    unique_colour = Dx.iloc[:, 0].unique()
    print("unique_colour----------")
    print(unique_colour)
    print("len(unique_colour--------------")
    print(len(unique_colour))
    # colors = [np.random.rand(3,) for i in range(len(unique_colour))]
    colors = [
        "red" if i == y_true_name else "gray" for i in unique_colour
    ]  # TODO: "b'GBM, RTK II'"
    print("colors------")
    print(colors)
    print(len(colors))

    data_color = []
    for i, j in zip(unique_colour, colors):
        print("%20s, %5s" % (i, j))
        data_color.append([i, j])

    df_colors = pd.DataFrame(data_color, columns=["unique_colour", "colors"])
    df_colors

    targets = Dx.to_numpy()
    print("targets------")
    print(targets.shape)
    print(targets)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)

    print("targets,colors == ", targets.shape, np.array(colors).shape)

    count_red_points = 0
    for target in targets:
        print("target == ", target)
        color = df_colors[df_colors["unique_colour"] == target[0]]["colors"]  # [0]
        print("----", type(color), color)
        color = color.values
        print(color)

        print("----------------------------------")
        print(target)
        print(color)
        print("----------------------------------")

        # print (finalDf.iloc[:,-1])
        # np.where(finalDf.iloc[:,-1] == target[0])
        indicesToKeep = finalDf.iloc[:, -1] == target[0]
        print(indicesToKeep.sum())

        # print (len(finalDf.loc[indicesToKeep, 'principal component 1']))

        # print("color----------------------------------------------------------- ", color)
        # print("finalDf.iloc[:,-1]------")
        # print(finalDf.iloc[:,-1])
        # print("indicesToKeep.shape------------")
        # print(indicesToKeep.shape)
        # print(finalDf.loc[indicesToKeep, 'principal component 1'].shape)

        if color == "red":
            print("NUM OF POINTS ================= ", print(indicesToKeep.sum()))

            ax.scatter(
                finalDf.loc[indicesToKeep, "principal component 1"],
                finalDf.loc[indicesToKeep, "principal component 2"],
                color="red",
                s=50,
            )
            count_red_points = count_red_points +1
        else:
            ax.scatter(
                finalDf.loc[indicesToKeep, "principal component 1"],
                finalDf.loc[indicesToKeep, "principal component 2"],
                color="blue",
                s=50,
                marker="+",
            )

        print("********************")

    for i in range(fakeresult.shape[0]):
        print("POSITION ====================== ", fakeresult[i, 0], fakeresult[i, 1])
        ax.scatter(fakeresult[i, 0], fakeresult[i, 1], c="green", s=300, marker="*")

    ax.grid()
    # ax.legend(y.iloc[:,0].unique())
    ax.legend(Dx.iloc[:, 0].unique(), loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig("PCA_all/all_points_%s.png" % (str(class_index)))
    # plt.show()

    print ("count_red_points ==== ", count_red_points)


def euclidean_distance(points, single_point):
    dist = (points - single_point) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    return dist


def plot_centroids_and_points(
    all_centroids, df_centroids, y_true_name, y_pred, class_index, y_centroid=None,
):

    print("y_pred ======= ", y_pred)
    print("y_pred ======= ", y_pred.size)
    print("y_pred ======= ", len(y_pred))

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)

    unique_colour = df_centroids.iloc[:, 0].unique()
    print("unique_colour----------")
    print(unique_colour)
    print("len(unique_colour--------------")
    print(len(unique_colour))
    # colors = [np.random.rand(3,) for i in range(len(unique_colour))]
    colors = [
        "red" if i == y_true_name else "gray" for i in unique_colour
    ]  # TODO: "b'GBM, RTK II'"
    print("colors------")
    print(colors)
    print(len(colors))

    data_color = []
    for i, j in zip(unique_colour, colors):
        print("%20s, %5s" % (i, j))
        data_color.append([i, j])

    df_colors = pd.DataFrame(data_color, columns=["unique_colour", "colors"])
    df_colors

    for index, row_centroid in df_centroids.iterrows():
        print("row_centroid === ", row_centroid)
        print("row_centroid === ", type(row_centroid))
        PCA1 = row_centroid["centroid_x"]
        PCA2 = row_centroid["centroid_y"]
        centroid_name = row_centroid["centroid_name"]
        print("row_centroid == ", row_centroid)
        color = df_colors[df_colors["unique_colour"] == row_centroid["centroid_name"]][
            "colors"
        ]  # [0]
        print("----", type(color), color)
        color = color.values
        print(color)

        print("----------------------------------")
        # print(target)
        print(color)
        print("----------------------------------")

        # print (finalDf.iloc[:,-1])
        # np.where(finalDf.iloc[:,-1] == target[0])
        # indicesToKeep = finalDf.iloc[:,-1] == target[0]
        # print (indicesToKeep.sum())

        # print (len(finalDf.loc[indicesToKeep, 'principal component 1']))

        # print("color----------------------------------------------------------- ", color)
        # print("finalDf.iloc[:,-1]------")
        # print(finalDf.iloc[:,-1])
        # print("indicesToKeep.shape------------")
        # print(indicesToKeep.shape)
        # print(finalDf.loc[indicesToKeep, 'principal component 1'].shape)

        ax.scatter(PCA1, PCA2, color=color, s=50)

        """
        if color == "red":
            print ("NUM OF POINTS ================= ", print (indicesToKeep.sum()))
            ax.scatter(PCA1, PCA2, color = "red", s = 50)
            #ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], color = "red", s = 50)
        else:
            ax.scatter(PCA1, PCA2, color = "gray", s = 50, marker="+")
            #ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], color = "blue", s = 50, marker="+")
        """

        print("********************")

    print("y_pred == ")
    print(y_pred)
    print("y_pred ======= ", y_pred)
    print("y_pred ======= ", y_pred.size)
    print("y_pred ======= ", len(y_pred))

    # plot the prediction point
    if len(y_pred) == 1:  # only one point
        ax.scatter(y_pred[0][0], y_pred[0][1], color="blue", s=50, marker="+")
    else:  # more than one point
        y_pred_x = [i[0] for i in y_pred]
        y_pred_y = [i[1] for i in y_pred]
        ax.scatter(y_pred_x, y_pred_y, color="blue", s=50, marker="+")

    """
    # plot centroid point
    print (y_centroid)
    y_centroid_x = y_centroid["centroid_x"]
    y_centroid_y = y_centroid["centroid_y"]
    y_centroid_name = y_centroid["centroid_name"]
    ax.scatter(y_centroid_x, y_centroid_y, color = "blue", s = 50, marker="*")
    """

    plt.savefig("PCA_centroids/all_centroids_%s.png" % (str(class_index)))
    # plt.show()


def nearest_centroid(df_centroids, distances, y_true_name):
    df_distances = pd.DataFrame(columns=["centroid_name", "distance"])
    df_distances["centroid_name"] = df_centroids["centroid_name"].values
    df_distances["distance"] = distances

    print("-------------")
    print("-------------")
    print("-------------")
    print("-------------")
    print(df_distances)

    df_distances = df_distances.sort_values(by=["distance"])
    print(df_distances)

    nearest_position = np.where(df_distances["centroid_name"] == y_true_name)[0] + 1
    print("nearest_position ================ ", nearest_position)
    print (df_distances[df_distances["centroid_name"] == y_true_name]["distance"])
    print (df_distances[df_distances["centroid_name"] == y_true_name]["distance"].values)
    print (df_distances[df_distances["centroid_name"] == y_true_name]["distance"].values[0])

    return nearest_position[0], df_distances[df_distances["centroid_name"] == y_true_name]["distance"].values[0]


def main():

    df_all_end_results = pd.DataFrame(
        columns=["Class", "Precison", "Recal", "F1", "Accuracy", "NearestNeigbors"]
    )

    x = pd.read_csv("betaValues5000all.csv", sep=";", header=None)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    print("principalComponents-------------")
    print(principalComponents)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )
    print("principalDf------------")
    print(principalDf)

    Dx = pd.read_csv("Dx.csv", sep=";", header=None)
    Dx.columns = ["Dx"]
    print("Dx---------------")
    print(Dx)

    finalDf = pd.concat([principalDf, Dx], axis=1)
    print("finalDf------------")
    print(finalDf)
    finalDf.to_csv("finalpca.csv", sep=";", header=None, index=False)

    df_grouped = finalDf.groupby("Dx")
    centroid_x, centroid_y, centroid_name = return_df_centroids(df_grouped)
    df_centroids = pd.DataFrame(
        list(zip(centroid_name, centroid_x, centroid_y)),
        columns=["centroid_name", "centroid_x", "centroid_y"],
    )

    print(df_centroids)

    targets = Dx.to_numpy()
    print("targets------")
    print(targets.shape)
    print(targets)

    y_true_all_class_names = [
        "b'A IDH'",
        "b'A IDH, HG'",
        "b'ANA PA'",
        "b'ATRT, MYC'",
        "b'ATRT, SHH'",
        "b'ATRT, TYR'",
        "b'CHGL'",
        "b'CHORDM'",
        "b'CN'",
        "b'CNS NB, FOXR2'",
        "b'CONTR, ADENOPIT'",
        "b'CONTR, CEBM'",
        "b'CONTR, HEMI'",
        "b'CONTR, HYPTHAL'",
        "b'CONTR, INFLAM'",
        "b'CONTR, PINEAL'",
        "b'CONTR, PONS'",
        "b'CONTR, REACT'",
        "b'CONTR, WM'",
        "b'CPH, ADM'",
        "b'CPH, PAP'",
        "b'DLGNT'",
        "b'DMG, K27'",
        "b'EFT, CIC'",
        "b'ENB, A'",
        "b'ENB, B'",
        "b'EPN, MPE'",
        "b'EPN, PF A'",
        "b'EPN, PF B'",
        "b'EPN, RELA'",
        "b'EPN, SPINE'",
        "b'EPN, YAP'",
        "b'ETMR'",
        "b'EWS'",
        "b'GBM, G34'",
        "b'GBM, MES'",
        "b'GBM, MID'",
        "b'GBM, MYCN'",
        "b'GBM, RTK I'",
        "b'GBM, RTK II'",
        "b'GBM, RTK III'",
        "b'HGNET, BCOR'",
        "b'HGNET, MN1'",
        "b'HMB'",
        "b'IHG'",
        "b'LGG, DIG/DIA'",
        "b'LGG, DNT'",
        "b'LGG, GG'",
        "b'LGG, MYB'",
        "b'LGG, PA MID'",
        "b'LGG, PA PF'",
        "b'LGG, PA/GG ST'",
        "b'LGG, RGNT'",
        "b'LGG, SEGA'",
        "b'LIPN'",
        "b'LYMPHO'",
        "b'MB, G3'",
        "b'MB, G4'",
        "b'MB, SHH CHL AD'",
        "b'MB, SHH INF'",
        "b'MB, WNT'",
        "b'MELAN'",
        "b'MELCYT'",
        "b'MNG'",
        "b'O IDH'",
        "b'PGG, nC'",
        "b'PIN T,  PB A'",
        "b'PIN T,  PB B'",
        "b'PIN T, PPT'",
        "b'PITAD, ACTH'",
        "b'PITAD, FSH LH'",
        "b'PITAD, PRL'",
        "b'PITAD, STH DNS A'",
        "b'PITAD, STH DNS B'",
        "b'PITAD, STH SPA'",
        "b'PITAD, TSH'",
        "b'PITUI'",
        "b'PLASMA'",
        "b'PLEX, AD'",
        "b'PLEX, PED A'",
        "b'PLEX, PED B'",
        "b'PTPR, A'",
        "b'PTPR, B'",
        "b'PXA'",
        "b'RETB'",
        "b'SCHW'",
        "b'SCHW, MEL'",
        "b'SFT HMPC'",
        "b'SUBEPN, PF'",
        "b'SUBEPN, SPINE'",
        "b'SUBEPN, ST'",
    ]

    df_summary = pd.DataFrame(
        columns=[
            "class_index",
            "fake_index",
            "real_name",
            "pred_name",
            "number_of_nearest_classes",
            "distance",
        ]
    )

    for class_index in range(91):

        file_name = "fake_sigmoid_%s.csv" % (
            str(class_index)
        )  # "fake_orginal_%s.csv" % (str(class_index))
        # fake=pd.read_csv("fake81_500_PTPR,A_last_sig_new.csv", sep=";",header=None) # TODO: HIER AUS ÄNDERN
        fake = pd.read_csv(file_name, sep=";", header=None)  # TODO: HIER AUS ÄNDERN

        fakeresult = pca.transform(fake)
        print("fakeresult.shape-----------")
        print(fakeresult.shape)
        print(fakeresult)

        y_true_name = y_true_all_class_names[class_index]  # "b'PTPR, A'"
        #plot_all_points(Dx, finalDf, y_true_name, fakeresult, class_index)

        all_centroids = df_centroids[["centroid_x", "centroid_y"]].values

        y_pred_index_all, y_pred_name_all, nearest_position_all = [], [], []
        index_fake_point = 1
        for fake_point in fakeresult:
            # fakeresult_x = [i[0] for i in y_pred]
            # fakeresult_y = [i[1] for i in y_pred]
            # fake_point_x = fake_point[0]
            # fake_point_y = fake_point[1]

            distances = euclidean_distance(all_centroids, fake_point)
            # df_centroids['a'].apply(lambda x: x + 1)
            print("distances")
            print(distances)
            print(np.argmin(distances), min(distances))
            print(df_centroids.iloc[np.argmin(distances)]["centroid_name"])
            print("distances.shape ==== ", distances.shape)
            print("fakeresult.shape === ", fakeresult.shape)
            print("y_true_name ======= ", y_true_name)
            print("len(----)", len(df_centroids["centroid_name"].values))

            y_pred_index_all.append(np.argmin(distances))
            y_pred_name_all.append(
                df_centroids.iloc[np.argmin(distances)]["centroid_name"]
            )

            nearest_position, distance_nearest_position = nearest_centroid(
                df_centroids, distances, y_true_name
            )

            nearest_position_all.append(nearest_position)

            df_summary.loc[len(df_summary)] = [
                class_index,
                index_fake_point,
                y_true_name,
                df_centroids.iloc[np.argmin(distances)]["centroid_name"],
                nearest_position,
                distance_nearest_position,
            ]
            index_fake_point = index_fake_point + 1

        # calculate the nearest position for the centroid of the fakeresult
        print(fakeresult)

        fakeresult_x = [i[0] for i in fakeresult]
        fakeresult_y = [i[1] for i in fakeresult]

        fakeresult_centroid = centroid(fakeresult_x, fakeresult_y)
        print("fakeresult_centroid == ", fakeresult_centroid)
        distances = euclidean_distance(all_centroids, fakeresult_centroid)

        nearest_position, distance_nearest_position = nearest_centroid(
            df_centroids, distances, y_true_name
        )
        nearest_position_all.append(nearest_position)

        df_summary.loc[len(df_summary)] = [
            class_index,
            index_fake_point,
            y_true_name,
            df_centroids.iloc[np.argmin(distances)]["centroid_name"],
            nearest_position,
            distance_nearest_position,
        ]

        print("+++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++")

        print("nearest_position_all")
        print(nearest_position_all)
        print(len(nearest_position_all))

        print("df_summary")
        print(df_summary)



        y_true = df_centroids.loc[df_centroids["centroid_name"] == y_true_name]
        y_pred = fakeresult.copy()
        y_pred_name = df_centroids.iloc[np.argmin(distances)]["centroid_name"]
        # y_centroid = df_centroids.iloc[np.argmin(distances)]#[["centroid_x", "centroid_y"]]

        print("********************************************")
        # print (y_centroid)
        print("********************************************")

        """
        plot_centroids_and_points(
            all_centroids,
            df_centroids,
            y_true_name,
            y_pred,
            # y_centroid,
            class_index,
        )
        """

        """
        KNN_pred_result = neigh.predict(fakeresult)
        print("KNN_pred_result === ", KNN_pred_result)
        y_pred = list(le.inverse_transform(KNN_pred_result))
        print("y_pred === ", y_pred)
        """
        print("y_true == ", y_true)
        y_true = y_true[["centroid_x", "centroid_y"]].values
        print("y_true == ", y_true)
        print("y_true == ", y_true[0])

        y_true = [list(y_true[0]) for i in range(len(fakeresult))]  # ["b'PTPR,A'"]
        y_true_name = [y_true_name] * len(fakeresult)  # ["b'PTPR,A'"]
        y_pred = fakeresult.copy()  # [y_pred_name]

        print("\n")
        print("*" * 20)
        print("y_true ==================== ", y_true)
        print("y_true_name =============== ", y_true_name)
        print("y_pred ==================== ", y_pred)
        print("y_pred_name_all =========== ", y_pred_name_all)
        print("len(y_pred) =============== ", len(y_pred))
        print("len(y_true) =============== ", len(y_true))
        print("*" * 20)
        labels = list(set(y_true_name + y_pred_name_all))  # ["ant", "bird", "cat"]
        print("labels ==================== ", labels)
        conf_mat = confusion_matrix(
            y_true_name, y_pred_name_all, labels=labels
        )  # TODO: konnte sein dass diese Zeile falsch .. bitte kontrolieren
        print("conf_mat ==================== ", conf_mat)
        class_report = classification_report(
            y_true_name, y_pred_name_all, output_dict=True
        )
        print("class_report ==================== ", class_report)
        # tn, fp, fn, tp = confusion_matrix(y_true_name, y_true_name).ravel()
        print("conf_mat")
        print(conf_mat)
        print("classification_report")
        print(class_report)
        # print ("tn, fp, fn, tp")
        # print (tn, fp, fn, tp)

        precision = class_report[y_true_all_class_names[class_index]]["precision"]
        recall = class_report[y_true_all_class_names[class_index]]["recall"]
        f1_score = class_report[y_true_all_class_names[class_index]]["f1-score"]
        accuracy = class_report["accuracy"]
        # ["Class", "Precison", "Recal", "F1", "Accuracy", "NearestNeigbors"]
        df_all_end_results.loc[len(df_all_end_results)] = [
            y_true_all_class_names[class_index],
            precision,
            recall,
            f1_score,
            accuracy,
            " / ".join(set(y_pred_name_all)),
        ]

    print(df_all_end_results)

    df_all_end_results.to_csv("df_all_end_results.csv", sep=";", index=False)

    df_summary.to_csv("df_summary.csv", sep=";", index=False)


main()
