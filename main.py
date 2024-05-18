
import time
import numpy as np
import argparse
import scipy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pynndescent
from graph_construction import graph_from_knn, generate_constraints_pairwise_fast, knn_k_estimating, \
    generate_constraints_label_fast
from SSSE_hierarchical import TreeSSSE
from SSSE_partitioning import FlatSSSE
from hierarchical_single_graph import PartitionTree_SSE
from util import dendrogram_purity_expected, Graph
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True, choices=['SSSE_partitioning_pairwise', 'SSSE_partitioning_label',
                                                         'SSSE_partitioning_bio_pairwise', 'SSSE_partitioning_bio_label', 'SSSE_hierarchical'])
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--constraint_ratio', type=float, required=True)
parser.add_argument('--constraint_weight', default=1.0, type=float)
parser.add_argument('--sigmasq', default=50, type=float, help='square of RBF kernel band width, i.e., sigma^2')
parser.add_argument('--exp_repeats', default=10, type=int)
parser.add_argument('--knn_constant', default=20, type=float)
parser.add_argument('--metric', default='euclidean')
parser.add_argument('--n_seeds', default=10, type=int, help='number of seeds when performing sampling_neighbor()')
parser.add_argument('--sampling', default='neighbor', type=str, choices=['random', 'neighbor'])
parser.add_argument('--knn_k', default=7, type=int)
parser.add_argument('--n_threads', default=30, type=int)
parser.add_argument('--n_montecalo', default=1000, type=int)
parser.add_argument('--sampling_size', default=1000, type=int)
parser.add_argument('--k_con',default=2, type=int)
parser.add_argument('--mergestop_SE', default=0, type=int, help='take SE into consideration or not when merge stops')
parser.add_argument('--normalization', default="MinMaxScaler", choices=['StandardScaler', 'MinMaxScaler'])
# for hierarchical clustering
parser.add_argument('--hie_knn_k', default=7)

args = parser.parse_args()


def SSSE_pairwise_clustering(path):
    args.metric = 'euclidean'
    args.normalization = 'MinMaxScaler'
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    y = np.array(data['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    n_instance_actual = n_instance
    if n_instance > args.sampling_size * 2.5:
        n_instance_actual = args.sampling_size
    args.knn_k = knn_k_estimating(n_cluster=n_cluster, n_instance=n_instance_actual, knn_constant=args.knn_constant)
    ARIs = []
    NMIs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:, :args.knn_k]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:, :args.knn_k]
        weight_knn = np.exp(- dist_knn ** 2 / (2 * args.sigmasq))
        graph = graph_from_knn(ind_knn, weight_knn)
        if args.constraint_ratio > 0:
            graph_con = generate_constraints_pairwise_fast(y, args.constraint_ratio * n_instance,
                                                           args.constraint_ratio * n_instance, X, args, weight_knn)
        else:
            graph_con = Graph(n_instance)
        flatSSSE = FlatSSSE(graph, graph_con, n_cluster, args.sampling_size, args)
        y_pred = flatSSSE.run_parallel()
        time2 = time.time()
        ARI = adjusted_rand_score(y, y_pred)
        NMI = normalized_mutual_info_score(y, y_pred)
        ARIs.append(ARI)
        NMIs.append(NMI)
        times.append(time2 - time1)
        print(ARI, NMI)
    print("average: {}\t{}\t{}\n".format(args.dataset, np.mean(ARIs), np.mean(NMIs)))

def SSSE_label_clustering(path):
    args.metric = 'euclidean'
    args.normalization = 'MinMaxScaler'
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    y = np.array(data['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    n_instance_actual = n_instance
    if n_instance > args.sampling_size * 2.5:
        n_instance_actual = args.sampling_size
    args.knn_k = knn_k_estimating(n_cluster=n_cluster, n_instance=n_instance_actual, knn_constant=args.knn_constant)
    ARIs = []
    NMIs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:, :args.knn_k ]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:, :args.knn_k ]
        weight_knn = np.exp(- dist_knn ** 2 / (2 * args.sigmasq))
        graph = graph_from_knn(ind_knn, weight_knn)
        if args.constraint_ratio > 0:
            graph_con = generate_constraints_label_fast(y, args.constraint_ratio * n_instance, args.constraint_ratio * n_instance, X, args, weight_knn)
        else:
            graph_con = Graph(n_instance)
        flatSSSE = FlatSSSE(graph, graph_con, n_cluster, args.sampling_size, args)
        y_pred = flatSSSE.run_parallel()
        time2 = time.time()
        ARI = adjusted_rand_score(y, y_pred)
        NMI = normalized_mutual_info_score(y, y_pred)
        ARIs.append(ARI)
        NMIs.append(NMI)
        times.append(time2-time1)
        print(path, ARI, NMI)
    print("average: {}\t{}\t{}\n".format(args.dataset, np.mean(ARIs), np.mean(NMIs)))

def SE_partitioning_clustering(path):
    args.metric = 'euclidean'
    args.normalization = 'MinMaxScaler'
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    y = np.array(data['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    n_instance_actual = n_instance
    args.knn_k = knn_k_estimating(n_cluster=n_cluster, n_instance=n_instance_actual, knn_constant=args.knn_constant)
    ARIs = []
    NMIs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:, :args.knn_k]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:, :args.knn_k]
        weight_knn = np.exp(- dist_knn ** 2 / (2 * args.sigmasq))
        graph = graph_from_knn(ind_knn, weight_knn)
        graph_con = Graph(n_instance)
        from partitioning_single_graph import FlatSSE
        flatSSE = FlatSSE(graph, graph_con)
        y_pred = flatSSE.build_tree(moving=False)
        time2 = time.time()
        ARI = adjusted_rand_score(y, y_pred)
        NMI = normalized_mutual_info_score(y, y_pred)
        ARIs.append(ARI)
        NMIs.append(NMI)
        times.append(time2 - time1)
        print(path, ARI, NMI)
    print("average: {}\t{}\t{}\n".format(args.dataset, np.mean(ARIs), np.mean(NMIs)))

def SE_partitioning_clustering_scalable(path):
    args.metric = 'euclidean'
    args.normalization = 'MinMaxScaler'
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    y = np.array(data['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    n_instance_actual = n_instance
    if n_instance > args.sampling_size * 2.5:
        n_instance_actual = args.sampling_size
    args.knn_k = knn_k_estimating(n_cluster=n_cluster, n_instance=n_instance_actual, knn_constant=args.knn_constant)
    ARIs = []
    NMIs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:, :args.knn_k ]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:, :args.knn_k ]
        weight_knn = np.exp(- dist_knn ** 2 / (2 * args.sigmasq))
        graph = graph_from_knn(ind_knn, weight_knn)
        graph_con = Graph(n_instance)
        flatSSSE = FlatSSSE(graph, graph_con, n_cluster, args.sampling_size, args)
        y_pred = flatSSSE.run_parallel()
        print(np.unique(y_pred).shape[0])
        time2 = time.time()
        ARI = adjusted_rand_score(y, y_pred)
        NMI = normalized_mutual_info_score(y, y_pred)
        ARIs.append(ARI)
        NMIs.append(NMI)
        times.append(time2-time1)
        print(path, ARI, NMI)
    print("average: {}\t{}\t{}\n".format(args.dataset, np.mean(ARIs), np.mean(NMIs)))

def SSSE_hierar_clustering(path):
    args.metric = "cosine"
    args.normalization = 'StandardScaler'
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    y = np.array(data['gnd']).astype(float).flatten()
    n_instance = y.shape[0]
    args.knn_k = args.hie_knn_k
    DPs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:,1:args.knn_k]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:,1:args.knn_k]
        weight_knn = 1 - dist_knn
        graph = graph_from_knn(ind_knn, weight_knn)
        if args.constraint_ratio > 0:
            graph_con = generate_constraints_pairwise_fast(y, args.constraint_ratio * n_instance, args.constraint_ratio * n_instance, X, args, weight_knn)
        else:
            graph_con = Graph(n_instance)
        treeSSSE = TreeSSSE(graph, graph_con, None, args.sampling_size, args)
        t = treeSSSE.run_parallel()
        time2 = time.time()
        DP = dendrogram_purity_expected(t, y, args.n_montecalo)
        DPs.append(DP)
        times.append(time2-time1)
        print(path, DP)
    print("average: {}\t{}\n".format(args.dataset, np.mean(DPs)))

def SE_hierar_clustering(path):
    args.metric = "cosine"
    args.normalization = 'StandardScaler'
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    y = np.array(data['gnd']).astype(float).flatten()
    n_instance = y.shape[0]
    args.knn_k = args.hie_knn_k
    DPs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:,1:args.knn_k]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:,1:args.knn_k]
        weight_knn = 1 - dist_knn
        graph = graph_from_knn(ind_knn, weight_knn)
        graph_con = Graph(n_instance)
        treeSSE = PartitionTree_SSE(graph, graph_con)
        root_id, hierarchical_tree_node = treeSSE.build_tree()
        t = treeSSE.subtree2etetree(root_id, hierarchical_tree_node)
        time2 = time.time()
        DP = dendrogram_purity_expected(t, y, args.n_montecalo)
        DPs.append(DP)
        times.append(time2-time1)
        print(path, DP)
    print("average: {}\t{}\n".format(args.dataset, np.mean(DPs)))

def SE_hierar_clustering_scalable(path):
    args.metric = "cosine"
    args.normalization = 'StandardScaler'
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    y = np.array(data['gnd']).astype(float).flatten()
    n_instance = y.shape[0]
    args.knn_k = args.hie_knn_k
    DPs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:, 1:args.knn_k]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:, 1:args.knn_k]
        weight_knn = 1 - dist_knn
        graph = graph_from_knn(ind_knn, weight_knn)
        graph_con = Graph(n_instance)
        treeSSSE = TreeSSSE(graph, graph_con, None, args.sampling_size, args)
        t = treeSSSE.run_parallel()
        time2 = time.time()
        DP = dendrogram_purity_expected(t, y, args.n_montecalo)
        DPs.append(DP)
        times.append(time2 - time1)
        print(path, DP)
    print("average: {}\t{}\n".format(args.dataset, np.mean(DPs)))

def SSSE_pairwise_clustering_bio(path):
    args.metric = 'cosine'
    args.normalization = "MinMaxScaler"
    args.mergestop_SE = 1
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    y = np.array(data['gnd']).astype(float).squeeze()
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    n_instance_actual = n_instance
    if n_instance > args.sampling_size * 2.5:
        n_instance_actual = args.sampling_size
    args.knn_k = knn_k_estimating(n_cluster=n_cluster, n_instance=n_instance_actual, knn_constant=args.knn_constant)
    ARIs = []
    NMIs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:, :args.knn_k ]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:, :args.knn_k ]
        weight_knn = 1 - dist_knn
        graph = graph_from_knn(ind_knn, weight_knn)
        if args.constraint_ratio > 0:
            graph_con = generate_constraints_pairwise_fast(y, args.constraint_ratio * n_instance, args.constraint_ratio * n_instance, X, args, weight_knn)
        else:
            graph_con = Graph(n_instance)
        flatSSSE = FlatSSSE(graph, graph_con, n_cluster, args.sampling_size, args)
        y_pred = flatSSSE.run_parallel()
        time2 = time.time()
        ARI = adjusted_rand_score(y, y_pred)
        NMI = normalized_mutual_info_score(y, y_pred)
        ARIs.append(ARI)
        NMIs.append(NMI)
        times.append(time2-time1)
        print(path, ARI, NMI)
    print("average: {}\t{}\t{}\n".format(args.dataset, np.mean(ARIs), np.mean(NMIs)))

def SSSE_label_clustering_bio(path):
    args.metric = 'cosine'
    args.normalization = "MinMaxScaler"
    args.mergestop_SE = 1
    if args.dataset == 'Karagiannis':
        args.normalization = "StandardScaler"
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    y = np.array(data['gnd']).astype(float).squeeze()
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    n_instance_actual = n_instance
    if n_instance > args.sampling_size * 2.5:
        n_instance_actual = args.sampling_size
    args.knn_k = knn_k_estimating(n_cluster=n_cluster, n_instance=n_instance_actual, knn_constant=args.knn_constant)
    ARIs = []
    NMIs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:, :args.knn_k ]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:, :args.knn_k ]
        weight_knn = 1 - dist_knn
        graph = graph_from_knn(ind_knn, weight_knn)
        if args.constraint_ratio > 0:
            graph_con = generate_constraints_label_fast(y, args.constraint_ratio * n_instance, args.constraint_ratio * n_instance, X, args, weight_knn)
        else:
            graph_con = Graph(n_instance)
        flatSSSE = FlatSSSE(graph, graph_con, n_cluster, args.sampling_size, args)
        y_pred = flatSSSE.run_parallel()
        time2 = time.time()
        ARI = adjusted_rand_score(y, y_pred)
        NMI = normalized_mutual_info_score(y, y_pred)
        ARIs.append(ARI)
        NMIs.append(NMI)
        times.append(time2-time1)
        print(path, ARI, NMI)
    print("average: {}\t{}\t{}\n".format(args.dataset, np.mean(ARIs), np.mean(NMIs)))

def SE_partitioning_clustering_scalable_bio(path):
    args.metric = 'cosine'
    args.normalization = "MinMaxScaler"
    args.mergestop_SE = 1
    if args.dataset == 'Karagiannis':
        args.normalization = "StandardScaler"
    data = scipy.io.loadmat(path)
    X = np.array(data['fea']).astype(float)
    y = np.array(data['gnd']).astype(float).squeeze()
    if args.normalization == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif args.normalization == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    else:
        raise Exception("not implemented")
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    n_instance_actual = n_instance
    if n_instance > args.sampling_size * 2.5:
        n_instance_actual = args.sampling_size
    args.knn_k = knn_k_estimating(n_cluster=n_cluster, n_instance=n_instance_actual, knn_constant=args.knn_constant)
    ARIs = []
    NMIs = []
    times = []
    for _ in range(args.exp_repeats):
        time1 = time.time()
        index = pynndescent.NNDescent(X, metric=args.metric, n_neighbors=100, pruning_degree_multiplier=10.0)
        ind_knn = np.array(index.neighbor_graph[0]).astype(int)[:, :args.knn_k ]
        dist_knn = np.array(index.neighbor_graph[1]).astype(float)[:, :args.knn_k ]
        weight_knn = 1 - dist_knn
        graph = graph_from_knn(ind_knn, weight_knn)
        graph_con = Graph(n_instance)
        flatSSSE = FlatSSSE(graph, graph_con, n_cluster, args.sampling_size, args)
        y_pred = flatSSSE.run_parallel()
        print(np.unique(y_pred).shape[0])
        time2 = time.time()
        ARI = adjusted_rand_score(y, y_pred)
        NMI = normalized_mutual_info_score(y, y_pred)
        ARIs.append(ARI)
        NMIs.append(NMI)
        times.append(time2-time1)
        print(path, ARI, NMI)
    print("average: {}\t{}\t{}\n".format(args.dataset, np.mean(ARIs), np.mean(NMIs)))


if __name__=='__main__':
    if args.method == "SSSE_partitioning_pairwise":
        path = "./datasets/clustering/{}.mat".format(args.dataset)
        SSSE_pairwise_clustering(path)
    elif args.method == "SSSE_partitioning_label":
        path = "./datasets/clustering/{}.mat".format(args.dataset)
        SSSE_label_clustering(path)
    elif args.method == "SSSE_hierarchical":
        path = "./datasets/clustering/{}.mat".format(args.dataset)
        SSSE_hierar_clustering(path)
    elif args.method == "SSSE_partitioning_bio_pairwise":
        path = "./datasets/RNA-seq/{}.mat".format(args.dataset)
        SSSE_pairwise_clustering_bio(path)
    elif args.method == "SSSE_partitioning_bio_label":
        path = "./datasets/RNA-seq/{}.mat".format(args.dataset)
        SSSE_label_clustering_bio(path)

