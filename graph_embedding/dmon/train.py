# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
r"""Graph Clustering with Graph Neural Networks.

===============================
This is the implementation of our paper,
[Graph Clustering with Graph Neural Networks]
(https://arxiv.org/abs/2006.16904).

The included code creates a DMoN (Deep Modularity Network) as introduced in the
paper.

Example execution to reproduce the results from the paper.
------
# From google-research/
python3 -m graph_embedding.dmon.train \
--graph_path=graph_embedding/dmon/data/cora.npz --dropout_rate=0.5
"""

import networkx as nx
import numpy as np
import scipy.sparse
import sklearn.metrics
import tensorflow.compat.v2 as tf
from absl import app
from absl import flags

from graph_embedding.dmon import dmon
from graph_embedding.dmon import gcn
from graph_embedding.dmon import metrics
from graph_embedding.dmon import utils
from graph_embedding.dmon.utilities.converters import SalsaConverter
from graph_embedding.dmon.utilities.metrics import grode

tf.compat.v1.enable_v2_behavior()

from graph_embedding.dmon.utilities.visualization import show_results_on_graph

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'graph_path',
    None,
    'Input graph path.')
flags.DEFINE_list(
    'architecture',
    [64],
    'Network architecture in the format `a,b,c,d`.')
flags.DEFINE_float(
    'collapse_regularization',
    1,
    'Collapse regularization.',
    lower_bound=0)
flags.DEFINE_float(
    'dropout_rate',
    0,
    'Dropout rate for GNN representations.',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_integer(
    'n_clusters',
    16,
    'Number of clusters.',
    lower_bound=0)
flags.DEFINE_integer(
    'n_epochs',
    1000,
    'Number of epochs.',
    lower_bound=0)
flags.DEFINE_float(
    'learning_rate',
    0.001,
    'Learning rate.',
    lower_bound=0)


def load_npz(
        filename
):
    """Loads an attributed graph with sparse features from a specified Numpy file.

    Args:
      filename: A valid file name of a numpy file containing the input data.

    Returns:
      A tuple (graph, features, labels, label_indices) with the sparse adjacency
      matrix of a graph, sparse feature matrix, dense label array, and dense label
      index array (indices of nodes that have the labels in the label array).
    """
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        adjacency = scipy.sparse.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape'])

        features = scipy.sparse.csr_matrix(
            (loader['feature_data'], loader['feature_indices'],
             loader['feature_indptr']),
            shape=loader['feature_shape'])

        label_indices = loader['label_indices']
        labels = loader['labels']
    assert adjacency.shape[0] == features.shape[
        0], 'Adjacency and feature size must be equal!'
    assert labels.shape[0] == label_indices.shape[
        0], 'Labels and label_indices size must be equal!'
    return adjacency, features, labels, label_indices


def convert_scipy_sparse_to_sparse_tensor(
        matrix):
    """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

    Args:
      matrix: A scipy sparse matrix.

    Returns:
      A ternsorflow sparse matrix (rank-2 tensor).
    """
    matrix = matrix.tocoo()
    return tf.sparse.SparseTensor(
        np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
        matrix.shape)


def build_dmon(input_features,
               input_graph,
               input_adjacency):
    """Builds a Deep Modularity Network (DMoN) model from the Keras inputs.

    Args:
      input_features: A dense [n, d] Keras input for the node features.
      input_graph: A sparse [n, n] Keras input for the normalized graph.
      input_adjacency: A sparse [n, n] Keras input for the graph adjacency.

    Returns:
      Built Keras DMoN model.
    """
    output = input_features
    for n_channels in FLAGS.architecture:
        output = gcn.GCN(int(n_channels))([output, input_graph])
    pool, pool_assignment = dmon.DMoN(
        FLAGS.n_clusters,
        collapse_regularization=FLAGS.collapse_regularization,
        dropout_rate=FLAGS.dropout_rate)([output, input_adjacency])
    return tf.keras.Model(
        inputs=[input_features, input_graph, input_adjacency],
        outputs=[pool, pool_assignment])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # Load and process the data (convert node features to dense, normalize the
    # graph, convert it to Tensorflow sparse tensor.
    # adjacency, features, labels, label_indices = load_npz(FLAGS.graph_path)
    sc = SalsaConverter(root_folder='graph_embedding/dmon/data/salsa_cpp')
    all_graphs = sc.convert()
    frame_no = 0
    training_graph = all_graphs[frame_no]

    labels = np.array([(m) for m in nx.get_node_attributes(training_graph, 'membership').values()])
    label_indices = np.arange(labels.shape[0])

    # adjacency, features, labels, label_indices = load_npz(FLAGS.graph_path)
    adjacency = nx.adj_matrix(training_graph)
    features = np.identity(n=adjacency.shape[0])

    # features = features.todense()
    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]
    graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(adjacency.copy()))

    # Create model input placeholders of appropriate size
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

    model = build_dmon(input_features, input_graph, input_adjacency)

    # Computes the gradients wrt. the sum of losses, returns a list of them.
    def grad(model, inputs):
        with tf.GradientTape() as tape:
            _ = model(inputs, training=True)
            loss_value = sum(model.losses)
        return model.losses, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
    model.compile(optimizer, None)

    for epoch in range(FLAGS.n_epochs):
        loss_values, grads = grad(model, [features, graph_normalized, graph])
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        clusters = obtain_clusters(features, graph, graph_normalized, model)
        _, _, _, _, _, card_f1_score = grode(labels, clusters)
        _, _, _, _, _, full_f1_score = grode(labels, clusters, crit='full')
        print(f'epoch {epoch}, losses: ' +
              ' '.join([f'{loss_value.numpy():.4f}' for loss_value in
                        loss_values]) + f' Card F1: {card_f1_score:.3f} - Full F1: {full_f1_score:.3f}')

    clusters = obtain_clusters(features, graph, graph_normalized, model)

    # Prints some metrics used in the paper.
    print('Conductance:', metrics.conductance(adjacency, clusters))
    print('Modularity:', metrics.modularity(adjacency, clusters))
    print(
        'NMI:',
        sklearn.metrics.normalized_mutual_info_score(
            labels, clusters[label_indices], average_method='arithmetic'))
    precision = metrics.pairwise_precision(labels, clusters[label_indices])
    recall = metrics.pairwise_recall(labels, clusters[label_indices])
    print('F1:', 2 * precision * recall / (precision + recall))

    show_results_on_graph(training_graph, clusters, frame_no=frame_no)


def obtain_clusters(features, graph, graph_normalized, model):
    # Obtain the cluster assignments.
    _, assignments = model([features, graph_normalized, graph], training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
    return clusters


if __name__ == '__main__':
    app.run(main)
