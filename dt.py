import numpy as np
from collections import Counter

# Represents a node in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature         # Index of the feature to split on (for internal nodes)
        self.threshold = threshold     # Threshold value for the split
        self.left = left               # Left child node
        self.right = right             # Right child node
        self.value = value             # Class label for leaf nodes (most common label in the subset)

    def is_leaf_node(self):
        # A node is a leaf if it has a prediction label (i.e., no further splits)
        return self.value is not None


# The actual decision tree class
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split  # Minimum samples needed to split a node
        self.max_depth = max_depth                  # Maximum tree depth
        self.n_features = n_features                # Number of features to consider when splitting (for Random Forests)
        self.root = None                            # Root node of the tree

    def fit(self, X, y):
        # If n_features isn't set, use all features; otherwise use a subset
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        # Build the tree recursively
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions (recursion base case)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # If stopping condition met, return a leaf node with the most common label
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Randomly choose subset of features to consider for this split (used in Random Forests)
        feat_idx = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best feature and threshold to split on
        best_feature, best_thresh = self._best_split(X, y, feat_idx)

        # Split dataset into left and right branches based on best split
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        # Recursively grow the left and right subtrees
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        # Return current node
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        # Loop through all candidate features
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]           # Select one feature
            thresholds = np.unique(X_column)    # Try all unique values as split points

            for thr in thresholds:
                gain = self._info_gain(y, X_column, thr)  # Compute information gain

                if gain > best_gain:
                    # Update best split
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _info_gain(self, y, X_column, thr):
        # Information Gain = entropy(parent) - [weighted avg] entropy(children)

        # 1. Calculate entropy of current node (parent)
        parent_entropy = self._entropy(y)

        # 2. Split the column on threshold
        left_idxs, right_idxs = self._split(X_column, thr)

        # If a split results in empty branch, return 0 gain
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # 3. Compute weighted average entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # 4. Compute and return information gain
        return parent_entropy - child_entropy

    def _split(self, X_column, split_threshold):
        # Splits data into left and right branches based on threshold
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        # Calculates entropy using formula: -sum(p * log(p))

        hist = np.bincount(y)  # Count occurrences of each class
        ps = hist / len(y)     # Probabilities of each class

        # Create a list to hold the values of p * log(p) for each p in ps, but only if p > 0
        entropy_terms = []
        for p in ps:
            if p > 0:
                entropy_terms.append(p * np.log(p))

        # Sum the values in the list
        sum_entropy = sum(entropy_terms)

        # Negate the sum and return it
        return -sum_entropy


    def _most_common_label(self, y):
        # Returns the most frequent label in y
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        # Create an empty list to store results
        results = []

        # Loop over each sample x in the dataset X
        for x in X:
            # Traverse the decision tree for each sample and store the result
            result = self._traverse_tree(x, self.root)
            results.append(result)

        # Convert the list of results to a NumPy array and return it
        return np.array(results)

    def _traverse_tree(self, x, node):
        # Recursively traverses tree down to a leaf node
        if node.is_leaf_node():
            return node.value

        # Decide to go left or right
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
