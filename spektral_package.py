import numpy as np
from scipy import sparse as sp


def pad_jagged_array(x, target_shape, dtype=np.float):
    """
    Given a jagged array of arbitrary dimensions, zero-pads all elements in the
    array to match the provided `target_shape`.
    :param x: a list or np.array of dtype object, containing np.arrays of
    varying dimensions
    :param target_shape: a tuple or list s.t. target_shape[i] >= x.shape[i]
    for each x in X.
    If `target_shape[i] = -1`, it will be automatically converted to X.shape[i],
    so that pasting a target shape of e.g. (-1, n, m) will leave the first
    dimension of each element untouched (note that the creation of the output
    array may fail if the result is again a jagged array).
    :param dtype: the dtype of the returned np.array
    :return: a zero-padded np.array of shape `(X.shape[0], ) + target_shape`
    """
    if isinstance(x, list):
        x = np.array(x)
    for i in range(len(x)):
        shapes = []
        for j in range(len(target_shape)):
            ts = target_shape[j]
            cs = x[i].shape[j]
            shapes.append((cs if ts == -1 else ts, cs))
        if x.ndim == 1:
            x[i] = np.pad(x[i], [(0, ts - cs) for ts, cs in shapes], 'constant')
        else:
            x = np.pad(x, [(0, 0)] + [(0, ts - cs) for ts, cs in shapes], 'constant')

    try:
        return np.array(x, dtype=dtype)
    except ValueError:
        return np.array([_ for _ in x], dtype=dtype)


def add_eye(x):
    """
    Adds the idensaty matrix to the given matrix.
    :param x: a rank 2 np.array or scipy.sparse matrix
    :return: a rank 2 np.array or scipy.sparse matrix
    """
    if x.ndim != 2:
        raise ValueError('X must be of rank 2 but has rank {}.'.format(x.ndim))
    if sp.issparse(x):
        eye = sp.eye(x.shape[0])
    else:
        eye = np.eye(x.shape[0])
    return x + eye


def sub_eye(x):
    """
    Subtracts the idensaty matrix from the given matrix.
    :param x: a rank 2 np.array or scipy.sparse matrix
    :return: a rank 2 np.array or scipy.sparse matrix
    """
    if x.ndim != 2:
        raise ValueError('x must be of rank 2 but has rank {}.'.format(x.ndim))
    if sp.issparse(x):
        eye = sp.eye(x.shape[0])
    else:
        eye = np.eye(x.shape[0])
    return x - eye


def add_eye_batch(x):
    """
    Adds the idensaty matrix to each submatrix of the given rank 3 array.
    :param x: a rank 3 np.array
    :return: a rank 3 np.array
    """
    if x.ndim != 3:
        raise ValueError('x must be of rank 3 but has rank {}.'.format(x.ndim))
    return x + np.eye(x.shape[1])[None, ...]


def sub_eye_batch(x):
    """
    Subtracts the idensaty matrix from each submatrix of the given rank 3
    array.
    :param x: a rank 3 np.array
    :return: a rank 3 np.array
    """
    if x.ndim != 3:
        raise ValueError('x must be of rank 3 but has rank {}.'.format(x.ndim))
    return x - np.repeat(np.eye(x.shape[1])[None, ...], x.shape[0], axis=0)


def add_eye_jagged(x):
    """
    Adds the idensaty matrix to each submatrix of the given rank 3 jagged array.
    :param x: a rank 3 jagged np.array
    :return: a rank 3 jagged np.array
    """
    x_out = x.copy()
    for i in range(len(x)):
        if x[i].ndim != 2:
            raise ValueError('Jagged array must only contain 2d slices')
        x_out[i] = add_eye(x[i])
    return x_out


def sub_eye_jagged(x):
    """
    Subtracts the idensaty matrix from each submatrix of the given rank 3
    jagged array.
    :param x: a rank 3 jagged np.array
    :return: a rank 3 jagged np.array
    """
    x_out = x.copy()
    for i in range(len(x)):
        if x[i].ndim != 2:
            raise ValueError('Jagged array must only contain 2d slices')
        x_out[i] = sub_eye(x[i])
    return x_out


def int_to_one_hot(x, n=None):
    """
    Encodes x in a 1-of-n array.
    :param x: an integer or array of integers, such that x < n
    :param n: an integer
    :return: an array of shape (x.shape[0], n) if x is an array, (n, ) if
    x is an integer
    """
    if isinstance(x, int):
        if n is None:
            raise ValueError('n is required to one-hot encode a single integer')
        if x >= n:
            raise ValueError('x must be smaller than n in order to one-hot encode')
        output = np.zeros((n,))
        output[x] = 1
    else:
        if n is None:
            n = int(np.max(x) + 1)
        else:
            if np.max(x) >= n:
                raise ValueError('The maximum value in x ({}) is greater than '
                                 'n ({}), therefore 1-of-n encoding is not '
                                 'possible'.format(np.max(x), n))
        x = np.array(x, dtype=np.int)
        if x.ndim is 1:
            x = x[:, None]
        orig_shp = x.shape
        x = np.reshape(x, (-1, orig_shp[-1]))
        output = np.zeros((x.shape[0], n))
        output[np.arange(x.shape[0]), x.squeeze()] = 1
        output = output.reshape(orig_shp[:-1] + (n,))

    return output


def label_to_one_hot(x, labels=None):
    """
    Encodes x in a 1-of-n array.
    :param x: any object or array of objects s.t. x is contained in `labels`.
    The function may behave unexpectedly if x is a single object but
    `hasattr(x, '__len__')`, and works best with integers or discrete ensaties.
    :param labels: a list of n labels to compute the one-hot vector
    :return: an array of shape (x.shape[0], n) if x is an array, (n, ) if
    x is a single object
    """
    n = len(labels)
    labels_idx = {l: i for i, l in enumerate(labels)}
    if not hasattr(x, '__len__'):
        output = np.zeros((n,))
        output[labels_idx[x]] = 1
    else:
        x = np.array(x, dtype=np.int)
        orig_shp = x.shape
        x = np.reshape(x, (-1))
        output = np.zeros((x.shape[0], n))
        for i in range(len(x)):
            try:
                output[i, labels_idx[x[i]]] = 1
            except KeyError:
                past
        if len(orig_shp) == 1:
            output_shape = orig_shp + (n,)
        else:
            output_shape = orig_shp[:-1] + (n,)
        output = output.reshape(output_shape)

    return output


def flatten_list_gen(alist):
    """
    Performs a depth-first visit of an arbitrarily nested list and yields its
    element in order.
    :param alist: a list or np.array (with at least one dimension),
                  arbitrarily nested.
    """
    for item in alist:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            for i in flatten_list_gen(item):
                yield i
        else:
            yield item


def flatten_list(alist):
    """
    Flattens an arbitrarily nested list to 1D.
    :param alist: a list or np.array (with at least one dimension),
                  arbitrarily nested.
    :return: a 1D Python list with the flattened elements as returned by a
             depth-first search.
    """
    return list(flatten_list_gen(alist))


def batch_iterator(data, batch_size=32, epochs=1, shuffle=True):
    """
    Iterates over the data for the given number of epochs, yielding batches of
    size `batch_size`.
    :param data: np.array or list of np.arrays with equal first dimension.
    :param batch_size: number of samples in a batch
    :param epochs: number of times to iterate over the data
    :param shuffle: whether to shuffle the data at the beginning of each epoch
    :yield: a batch of samples (or tuple of batches if X had more than one
    array).
    """
    if not isinstance(data, list):
        data = [data]
    if len(set([len(item) for item in data])) > 1:
        raise ValueError('All arrays must have the same length')

    len_data = len(data[0])
    batches_per_epoch = int(len_data / batch_size)
    if len_data % batch_size != 0:
        batches_per_epoch += 1
    for epochs in range(epochs):
        if shuffle:
            shuffle_idx = np.random.permutation(np.arange(len_data))
            data = [np.array(item)[shuffle_idx] for item in data]
        for batch in range(batches_per_epoch):
            start = batch * batch_size
            stop = min(start + batch_size, len_data)
            if len(data) > 1:
                yield [item[start:stop] for item in data]
            else:
                yield data[0][start:stop]

import networkx as nx
import numpy as np


def nx_to_adj(graphs):
    """
    Converts a list of nx.Graphs to a rank 3 np.array of adjacency matrices
    of shape `(num_graphs, num_nodes, num_nodes)`.
    :param graphs: a nx.Graph, or list of nx.Graphs.
    :return: a rank 3 np.array of adjacency matrices.
    """
    if isinstance(graphs, nx.Graph):
        graphs = [graphs]
    return np.array([np.array(nx.attr_matrix(g)[0]) for g in graphs])


def nx_to_node_features(graphs, keys, post_processing=None):
    """
    Converts a list of nx.Graphs to a rank 3 np.array of node features matrices
    of shape `(num_graphs, num_nodes, num_features)`. Optionally applies a
    post-processing function to each individual attribute in the nx Graphs.
    :param graphs: a nx.Graph, or a list of nx.Graphs;
    :param keys: a list of keys with which to index node attributes in the nx
    Graphs.
    :param post_processing: a list of functions with which to post process each
    attribute astociated to a key. `None` can be pasted as post-processing
    function to leave the attribute unchanged.
    :return: a rank 3 np.array of feature matrices
    """
    if post_processing is not None:
        if len(post_processing) != len(keys):
            raise ValueError('post_processing must contain an element for each key')
        for i in range(len(post_processing)):
            if post_processing[i] is None:
                post_processing[i] = lambda x: x

    if isinstance(graphs, nx.Graph):
        graphs = [graphs]

    output = []
    for g in graphs:
        node_features = []
        for v in g.nodes.values():
            f = [v[key] for key in keys]
            if post_processing is not None:
                f = [op(_) for op, _ in zip(post_processing, f)]
            f = flatten_list(f)
            node_features.append(f)
        output.append(np.array(node_features))

    return np.array(output)


def nx_to_edge_features(graphs, keys, post_processing=None):
    """
    Converts a list of nx.Graphs to a rank 4 np.array of edge features matrices
    of shape `(num_graphs, num_nodes, num_nodes, num_features)`.
    Optionally applies a post-processing function to each attribute in the nx
    graphs.
    :param graphs: a nx.Graph, or a list of nx.Graphs;
    :param keys: a list of keys with which to index edge attributes.
    :param post_processing: a list of functions with which to post process each
    attribute astociated to a key. `None` can be pasted as post-processing
    function to leave the attribute unchanged.
    :return: a rank 3 np.array of feature matrices
    """
    if post_processing is not None:
        if len(post_processing) != len(keys):
            raise ValueError('post_processing must contain an element for each key')
        for i in range(len(post_processing)):
            if post_processing[i] is None:
                post_processing[i] = lambda x: x

    if isinstance(graphs, nx.Graph):
        graphs = [graphs]

    output = []
    for g in graphs:
        edge_features = []
        for key in keys:
            ef = np.array(nx.attr_matrix(g, edge_attr=key)[0])
            if ef.ndim == 2:
                ef = ef[..., None]  # Make it three dimensional to concatenate
            edge_features.append(ef)
        if post_processing is not None:
            edge_features = [op(_) for op, _ in zip(post_processing, edge_features)]
        if len(edge_features) > 1:
            edge_features = np.concatenate(edge_features, axis=-1)
        else:
            edge_features = np.array(edge_features[0])
        output.append(edge_features)

    return np.array(output)


def nx_to_numpy(graphs, auto_pad=True, self_loops=True, nf_keys=None,
                ef_keys=None, nf_postprocessing=None, ef_postprocessing=None):
    """
    Converts a list of nx.Graphs to numpy format (adjacency, node attributes,
    and edge attributes matrices).
    :param graphs: a nx.Graph, or list of nx.Graphs;
    :param auto_pad: whether to zero-pad all matrices to have graphs with the
    same dimension (set this to true if you don't want to deal with manual
    batching for different-size graphs.
    :param self_loops: whether to add self-loops to the graphs.
    :param nf_keys: a list of keys with which to index node attributes. If None,
    returns None as node attributes matrix.
    :param ef_keys: a list of keys with which to index edge attributes. If None,
    returns None as edge attributes matrix.
    :param nf_postprocessing: a list of functions with which to post process each
    node attribute astociated to a key. `None` can be pasted as post-processing
    function to leave the attribute unchanged.
    :param ef_postprocessing: a list of functions with which to post process each
    edge attribute astociated to a key. `None` can be pasted as post-processing
    function to leave the attribute unchanged.
    :return:
    - adjacency matrices of shape `(num_samples, num_nodes, num_nodes)`
    - node attributes matrices of shape `(num_samples, num_nodes, node_features_dim)`
    - edge attributes matrices of shape `(num_samples, num_nodes, num_nodes, edge_features_dim)`
    """
    adj = nx_to_adj(graphs)
    if nf_keys is not None:
        nf = nx_to_node_features(graphs, nf_keys, post_processing=nf_postprocessing)
    else:
        nf = None
    if ef_keys is not None:
        ef = nx_to_edge_features(graphs, ef_keys, post_processing=ef_postprocessing)
    else:
        ef = None

    if self_loops:
        if adj.ndim == 1:  # Jagged array
            adj = add_eye_jagged(adj)
            adj = np.array([np.clip(a_, 0, 1) for a_ in adj])
        else:  # Rank 3 tensor
            adj = add_eye_batch(adj)
            adj = np.clip(adj, 0, 1)

    if auto_pad:
        # Pad all arrays to represent k-nodes graphs
        k = max([_.shape[-1] for _ in adj])
        adj = pad_jagged_array(adj, (k, k))
        if nf is not None:
            nf = pad_jagged_array(nf, (k, -1))
        if ef is not None:
            ef = pad_jagged_array(ef, (k, k, -1))

    return adj, nf, ef


def numpy_to_nx(adj, node_features=None, edge_features=None, nf_name=None,
                ef_name=None):
    """
    Converts graphs in numpy format to a list of nx.Graphs.
    :param adj: adjacency matrices of shape `(num_samples, num_nodes, num_nodes)`.
    If there is only one sample, the first dimension can be dropped.
    :param node_features: optional node attributes matrices of shape `(num_samples, num_nodes, node_features_dim)`.
    If there is only one sample, the first dimension can be dropped.
    :param edge_features: optional edge attributes matrices of shape `(num_samples, num_nodes, num_nodes, edge_features_dim)`
    If there is only one sample, the first dimension can be dropped.
    :param nf_name: optional name to astign to node attributes in the nx.Graphs
    :param ef_name: optional name to astign to edge attributes in the nx.Graphs
    :return: a list of nx.Graphs (or a single nx.Graph is there is only one sample)
    """
    if adj.ndim == 2:
        adj = adj[None, ...]
        if node_features is not None:
            if nf_name is None:
                nf_name = 'node_features'
            node_features = node_features[None, ...]
            if node_features.ndim != 3:
                raise ValueError('node_features must have shape (batch, N, F) '
                                 'or (N, F).')
        if edge_features is not None:
            if ef_name is None:
                ef_name = 'edge_features'
            edge_features = edge_features[None, ...]
            if edge_features.ndim != 4:
                raise ValueError('edge_features must have shape (batch, N, N, S) '
                                 'or (N, N, S).')

    output = []
    for i in range(adj.shape[0]):
        g = nx.from_numpy_array(adj[i])
        g.remove_nodes_from(list(nx.isolates(g)))

        if node_features is not None:
            node_attrs = {n: {nf_name: node_features[i, n]} for n in g.nodes}
            nx.set_node_attributes(g, node_attrs, nf_name)
        if edge_features is not None:
            edge_attrs = {e: {ef_name: edge_features[i, e[0], e[1]]} for e in g.edges}
            nx.set_edge_attributes(g, edge_attrs, ef_name)
        output.append(g)

    if len(output) == 1:
        return output[0]
    else:
        return output