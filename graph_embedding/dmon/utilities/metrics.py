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

"""TODO(tsitsulin): add headers, tests, and improve style."""
from typing import List

import numpy as np
from sklearn.metrics.cluster import contingency_matrix


def grode(gt_as_list, det_as_list, crit='card'):
    def duplicates(lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    def get_groups_from_list(group_list: List):
        groups_as_set = set(group_list)
        people_as_groups = []
        for group_no in groups_as_set:
            people_as_groups.append(duplicates(group_list, group_no))

        return people_as_groups

    gt = get_groups_from_list(gt_as_list)
    det = get_groups_from_list(det_as_list)

    TP = 0
    # For each GT group
    for gt_group in gt:
        # Select the GT group and its cardinality (= number of elements)
        gt_group = set(gt_group)
        gt_card = len(gt_group)

        if gt_card <= 1:
            continue

        # for each detected group
        for det_group in det:
            # Select the detected group and its cardinality (= number of elements)
            det_group = set(det_group)
            det_card = len(det_group)

            if det_card <= 1:
                continue

            inters = gt_group.intersection(det_group)
            if crit == 'card':
                if (gt_card == 2 and det_card == 2):
                    if len(gt_group ^ det_group) == 0:
                        TP += 1
                        break
                elif len(inters) / max(gt_card, det_card) >= 2 / 3:
                    TP += 1
                    break
            else:
                if len(gt_group ^ det_group) == 0:
                    TP += 1

    FP = len(det) - TP
    FN = len(gt) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if TP == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return TP, FP, FN, precision, recall, f1_score


def precision(y_true, y_pred):
    true_positives, false_positives, _, _ = _compute_counts(y_true, y_pred)
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
    true_positives, _, false_negatives, _ = _compute_counts(y_true, y_pred)
    return true_positives / (true_positives + false_negatives)


def accuracy_score(y_true, y_pred):
    true_positives, false_positives, false_negatives, true_negatives = _compute_counts(
        y_true, y_pred)
    return (true_positives + true_negatives) / (
            true_positives + false_positives + false_negatives + true_negatives)


def _compute_counts(y_true, y_pred):  # TODO(tsitsulin): add docstring pylint: disable=missing-function-docstring
    contingency = contingency_matrix(y_true, y_pred)
    same_class_true = np.max(contingency, 1)
    same_class_pred = np.max(contingency, 0)
    diff_class_true = contingency.sum(axis=1) - same_class_true
    diff_class_pred = contingency.sum(axis=0) - same_class_pred
    total = contingency.sum()

    true_positives = (same_class_true * (same_class_true - 1)).sum()
    false_positives = (diff_class_true * same_class_true * 2).sum()
    false_negatives = (diff_class_pred * same_class_pred * 2).sum()
    true_negatives = total * (
            total - 1) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, false_negatives, true_negatives


def modularity(adjacency, clusters):
    degrees = adjacency.sum(axis=0).A1
    m = degrees.sum()
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix) ** 2) / m
    return result / m


def conductance(adjacency, clusters):  # TODO(tsitsulin): add docstring pylint: disable=missing-function-docstring
    inter = 0
    intra = 0
    cluster_idx = np.zeros(adjacency.shape[0], dtype=np.bool)
    for cluster_id in np.unique(clusters):
        cluster_idx[:] = 0
        cluster_idx[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_idx, :]
        inter += np.sum(adj_submatrix[:, cluster_idx])
        intra += np.sum(adj_submatrix[:, ~cluster_idx])
    return intra / (inter + intra)
