import numpy as np
import pickle
import matplotlib.pyplot as plt

from d2_clustering import D2Clustering
from Projection_d2_clustering import ProjectionD2Clustering


def loaddata(filename, d):
    stride = np.empty([1, 0], dtype=np.int)
    probs = np.empty([1, 0])
    supps = np.empty([d, 0])

    idx = 0
    with open(filename, 'r') as f:
        while True:
            s = f.readline()
            if not s:
                break
            stride = np.concatenate((stride, np.array([[s]], dtype=np.int)), axis=1)
            a = np.array([float(x) for x in f.readline().strip('\n').strip(' ').split(' ') if x != ''])
            probs = np.concatenate((probs, a.reshape([1, -1])), axis=1)
            X = np.empty([d, int(s)])
            for i in range(d):
                x = np.array([float(x) for x in f.readline().strip('\n').strip(' ').split(' ') if x != ''])
                X[i, :] = x
            supps = np.concatenate((supps, X), axis=1)
            idx += 1
    #             print(idx, s)
    #             if idx > 500:
    #                 break
    print('Data loaded!')

    return stride, probs, supps, idx


def run_clustering():
    d = 300
    n = 16
    maxiter = 10
    filename = 'BBCsportdata_d300_n16_tfidf.d2s'
    stride, probs, supps, idx = loaddata(filename, d)

    nb_exp = 5
    AMIs = np.zeros([2, maxiter, nb_exp])

    label_true = []
    with open("BBCsportdata_d300_n16_tfidf_labels.d2s", 'r') as f:
        while True:
            la = f.readline()
            if not la:
                break
            label_true.append(int(la))

    if 1:
        for exp in range(nb_exp):
            print("Experiments num: ", exp)
            init_point = np.random.randint(len(label_true))
            # init_point = 200

            d2Clustering = D2Clustering(n_clusters=5, max_iter=maxiter)
            AMIs[0, :, exp] = d2Clustering.parallel_fit(stride, probs, supps, n, label_true=label_true,
                                                        init_point=init_point, eta=1, otreg=0.5, fixed_supp=False)

            k = 2
            projd2Clustering = ProjectionD2Clustering(n_clusters=5, max_iter=maxiter, tau=0.05)
            AMIs[1, :, exp] = projd2Clustering.parallel_fit(stride, probs, supps, n, k, label_true=label_true,
                                                            init_point=init_point, eta=1, otreg=0.5, fixed_supp=False)

        with open('./bbcsport_ami.pkl', 'wb') as f:
            pickle.dump([AMIs], f)

    else:
        with open('./bbcsport_ami.pkl', 'rb') as f:
            [AMIs] = pickle.load(f)

    print(AMIs)

    line = ['-', '-']
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'purple']
    plt.figure(figsize=(12, 8))

    captions = ['D2 clustering', 'PD2 clustering']

    for t in range(2):
        AMI_mean = np.mean(AMIs[t, :, :], axis=1)
        plt.plot(np.arange(maxiter ), AMI_mean, ls=line[t], c=colors[t], lw=4, ms=20, label=captions[t])

    plt.xlabel('Iteration', fontsize=25)
    plt.ylabel('AMI', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=18, ncol=2)
    plt.savefig('figs/exp5_bbcsport.png')

if __name__ == "__main__":
    run_clustering()
