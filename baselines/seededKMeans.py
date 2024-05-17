import time

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from active_semi_clustering.semi_supervised.labeled_data import SeededKMeans
import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def SeededKMeans_ordinary(path):
    data = scipy.io.loadmat(path)
    X = np.array(data["fea"]).astype(float)
    X = MinMaxScaler().fit_transform(X)
    y = np.array(data["gnd"]).astype(float).squeeze()
    y = y - np.min(y)

    PL = generate_constraint_labels(y, int(y.shape[0]*0.1))
    print(PL, np.sum(PL))
    time1 = time.time()
    clusterer = SeededKMeans(n_clusters=len(np.unique(y)))
    clusterer.fit(X, PL)
    time2 = time.time()
    ARI = adjusted_rand_score(y,clusterer.labels_)
    NMI = normalized_mutual_info_score(y,clusterer.labels_)
    return ARI, NMI, time2-time1


def generate_constraint_labels(y, N):
    ratio = N / y.shape[0]
    PL = - np.ones_like(y)
    for valuei in np.unique(y):
        num_classi = np.sum(y==valuei)
        num_PLi = np.maximum(int(num_classi*ratio), 1)
        indicesi = np.where(y==valuei)
        indicesi_chosen = np.random.choice(indicesi[0], size=num_PLi, replace=False)
        PL[indicesi_chosen] = y[indicesi_chosen]
    return PL

if __name__=='__main__':
    # for dataset in ["Yale", "ORL", "COIL20", "COIL100", "USPS", "MNIST", "CovType"]:
    for dataset in ["COIL100", "USPS", "MNIST", "CovType"]:
        path = "/home/zengguangjie/SSSE/datasets/ordinary/{}.mat".format(dataset)
        ARIs = []
        NMIs = []
        times = []
        with open("seededKMeans/{}.txt".format(dataset), 'w') as f:
            for i in range(10):
                try:
                    ARI, NMI, time_i = SeededKMeans_ordinary(path)
                    ARIs.append(ARI)
                    NMIs.append(NMI)
                    times.append(time_i)
                except:
                    continue
            f.write("ARI:\t")
            for i in range(len(ARIs)):
                f.write("{}\t".format(ARIs[i]))
            f.write("average:\t{}\t".format(np.mean(ARIs)))
            f.write("std:\t{}\n".format(np.std(ARIs)))

            f.write("NMI:\t")
            for i in range(len(NMIs)):
                f.write("{}\t".format(NMIs[i]))
            f.write("average:\t{}\t".format(np.mean(NMIs)))
            f.write("std:\t{}\n".format(np.std(NMIs)))

            f.write("time:\t")
            for i in range(len(times)):
                f.write("{}\t".format(times[i]))
            f.write("average:\t{}\t".format(np.mean(times)))
            f.write("std:\t{}\n".format(np.std(times)))

            print(np.mean(ARIs), np.mean(NMIs))
