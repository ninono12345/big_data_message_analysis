import sklearn.cluster
import sklearn.cluster._hdbscan
from sklearn.cluster import HDBSCAN



def reduce_umap(emb_to_use, n_components, n_epochs, n_neighbours=20, cpu=False, timer=False):
    begin = time.time()
    emb_to_use = cp.asarray(emb_to_use)
    if cpu:
        model = sklearn.manifold.UMAP(n_neighbors=n_neighbours, n_components=n_components, min_dist=0.001, n_epochs=n_epochs, verbose=False)
    else:
        model = cuml.UMAP(n_neighbors=n_neighbours, n_components=n_components, min_dist=0.001, n_epochs=n_epochs, verbose=False)
    reduced_umap_emb = model.fit_transform(emb_to_use)
    # print(type(reduced_umap_emb))
    reduced_umap_emb = cp.asnumpy(reduced_umap_emb)
    if timer:
        print("umap time",time.time() - begin)
    return reduced_umap_emb

# nenaudojam, bet galim naudoti kaip umap alternatyva, maziau resursu reikalauja, bet gali tik iki 2 dimensiju mazint
def reduce_tsne(emb_to_use, cpu=True, timer=False):
    begin = time.time()
    # model = sklearn.manifold.TSNE()
    model = cuml.TSNE()
    reduced_tsne_emb = model.fit_transform(emb_to_use)
    if timer:
        print("tsne time",time.time() - begin)
    return reduced_tsne_emb

# nenaudojam. Gerai, kai nori plotinti 2d duomenis, nes outlieriai sugadina visa vaizda.
# veikia su bet kiek dimensiju, kuriau, nes maniau, kad su klasterizavimu pades, bet is esmes nera skirtumo
def reject_outliers(emb_to_use, text_ids, m = 2):
    axis_cnt = emb_to_use.shape[1]

    axis_medians = [np.median(emb_to_use[:,i]) for i in range(axis_cnt)]
    axis_d = [np.abs(emb_to_use[:,i] - axis_medians[i]) for i in range(axis_cnt)]
    axis_mdev = [np.median(axis_d[i]) for i in range(axis_cnt)]
    axis_s = [axis_d[i]/axis_mdev[i] for i in range(axis_cnt)]

    # to_remove[]
    while True:
        axis_to_remove = [np.where(axis_s[i] > m)[0] for i in range(axis_cnt)]
        to_remove = np.concatenate(axis_to_remove)
        to_remove = [int(i) for i in to_remove]
        if len(set(to_remove)) > emb_to_use.shape[0]*0.1:
            m += 1
            print("cancel remove", len(to_remove))
            continue
        break
    print("to_remove",len(to_remove),len(set(to_remove)), emb_to_use.shape[0]-len(to_remove))
    print(to_remove)
    print(type(text_ids))

    cleaned_emb = np.delete(emb_to_use, to_remove, axis=0)
    print(cleaned_emb.shape)

    text_ids_set = set(text_ids)
    # text_inds3 = [i for i in range(len(text_ids)) if i not in to_remove]
    text_inds3 = list(set(list(range(len(text_ids)))) - set(to_remove))
    print("inds", len(text_ids_set), len(text_inds3))

    text_ids3 = np.delete(text_ids, to_remove).tolist()
    print("text_ids3",len(text_ids3))
    

    return cleaned_emb, text_ids3, text_inds3

# naudojam siaip analizei, tiesiog palikau jei idomu savo dataseta perziuret
# passinam 2d embeddingus ir labels is HDBSCAN
def plot_clusters(emb_to_use, labels=None):
    centroids = {}
    print("emb_to_use", type(emb_to_use))
    print("plot_clusters labels", len(set(labels)))
    # labels = np.array(labels)

    if labels is not None:
        for l in np.unique(labels):
            c = emb_to_use[labels == l]
            c = c.mean(0)
            centroids[l] = c
        
    plt.figure(figsize=(24, 20))
    print("done")
    
    
    if labels is not None:
        plt.scatter(emb_to_use[:, 0], emb_to_use[:, 1], c=labels, s=1, alpha=0.7, cmap=plt.cm.get_cmap('jet'))
        for l, c in centroids.items():
            plt.annotate(str(l), (c[0], c[1]), fontsize=10)
    else:
        plt.scatter(emb_to_use[:, 0], emb_to_use[:, 1], c='blue', s=1, alpha=0.7, cmap=plt.cm.get_cmap('jet'))

def hdbscan_cluster(emb_to_use, text_ids, mcs=None, rt=False, cpu=False):
    # begin = time.time()
    cluster_only_inp = {text_id: emb_to_use[i] for i, text_id in enumerate(text_ids)}
    embeddings = list(cluster_only_inp.values())
    # embedding_array = cp.asarray(embeddings)
    if mcs is None:
        mcs = int(len(cluster_only_inp)/500)
    ms = mcs if mcs < 150 else 150
    # hbeg = time.time()
    if cpu:
        clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        # clusterer.fit(cp.asnumpy(embedding_array))
        clusterer.fit(embeddings)
    else:
        clusterer = cuml.cluster.HDBSCAN(min_cluster_size=mcs, min_samples=mcs)
        # clusterer.fit(embedding_array)
        clusterer.fit(cp.asarray(embeddings))
    # print("hdb", time.time()-hbeg, "cpu", cpu)
    labels = clusterer.labels_.tolist()

    cluster_to_text_ids = {i: [] for i in set(labels)}
    cluster_text_inds = {i: [] for i in set(labels)}
    for text_idx, text_id in enumerate(cluster_only_inp.keys()):
        label = labels[text_idx]
        cluster_to_text_ids[label].append(text_id)
        cluster_text_inds[label].append(text_idx)
    
    cluster_to_text_ids = {str(k): v for k, v in sorted(cluster_to_text_ids.items(), key=lambda x: len(x[1]), reverse=True) if v}
    cluster_text_inds = {str(k): v for k, v in sorted(cluster_text_inds.items(), key=lambda x: len(x[1]), reverse=True) if v}
    
    # print("hdbfull", time.time()-begin)
    # clusterer.condensed_tree_
    # clusterer.single_linkage_tree_
    if rt:
        return cluster_to_text_ids, cluster_text_inds, labels, clusterer.condensed_tree_
    return cluster_to_text_ids, cluster_text_inds, labels

def kmeans_cluster_cpu(emb_to_use, text_ids, cc=100):
    # from sklearn.cluster import MiniBatchKMeans
    # minibatch_kmeans = MiniBatchKMeans(
    #     n_clusters=cc,
    #     n_init=100,
    # )

    minibatch_kmeans = cuml.KMeans(n_clusters=cc)

    minibatch_kmeans.fit(emb_to_use)
    labels = minibatch_kmeans.labels_.tolist()

    cluster_to_text_ids = {i: [] for i in set(labels)}
    cluster_text_inds = {i: [] for i in set(labels)}
    for text_idx, text_id in enumerate(text_ids):
        label = labels[text_idx]
        cluster_to_text_ids[label].append(text_id)
        cluster_text_inds[label].append(text_idx)
    
    cluster_to_text_ids = {str(k): v for k, v in sorted(cluster_to_text_ids.items(), key=lambda x: len(x[1]), reverse=True) if v}
    cluster_text_inds = {str(k): v for k, v in sorted(cluster_text_inds.items(), key=lambda x: len(x[1]), reverse=True) if v}

    return cluster_to_text_ids, cluster_text_inds, labels