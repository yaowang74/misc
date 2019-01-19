# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:00:44 2018

@author: H211803
"""

import ast
import time
import numpy as np
import pandas as pd
import pydotplus.graphviz
# =============================================================================
# import matplotlib
# matplotlib.use('Agg')
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import CommonModules.errors as err
import sklearn.feature_selection as fs
from io import BytesIO
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from scipy.special import stdtr
from CommonModules.utils import FeatureInit
from CommonModules.loggerinitializer import InitializeLogger


np.set_printoptions(suppress=True)

logger = InitializeLogger("SupervisedSegmentation")


class SupervisedSegmentation:
    """

    """

    def __init__(self, initial_df, cfg_content):
        """

        """
        logger.info("Target column of input dataframe is: {}."
                    .format(cfg_content["Input"]["target"]))

        self.data_y = initial_df[[cfg_content["Input"]["target"]]]
        self.target_col = cfg_content["Input"]["target"]

        self.tree_type = [key for key in cfg_content["Algorithm"] if
                          cfg_content["Algorithm"].getboolean(key)][0].lower()
        logger.info("Selected algorithm(s) is(are): {}"
                    .format(self.tree_type))

        self.algo_param_dict = cfg_content["AlgoProperties"]
        logger.info("Parameters for selected algorithm(s) are: {}"
                    .format([(key, values) for key, values in
                             self.algo_param_dict.items()]))

        self.selected_var_list =\
            ast.literal_eval(self.algo_param_dict['var_in_use'])
        aug_var_list = self.selected_var_list + [self.target_col]
        logger.info("Selected columns along with target column are: {}".format(
                aug_var_list))
        initial_df = initial_df[aug_var_list]
        initial_df = initial_df[aug_var_list]
        self.out_path = cfg_content['Output']
        logger.info("Output file location: {}"
                    .format(self.out_path['path'] + self.out_path['filename']))
        logger.info("Extracting numeric columns...")
        self.num_data = initial_df.select_dtypes(include=np.number)
        if not self.num_data.empty:
            logger.info("First 5 rows of numerical columns:")
            logger.info(self.num_data.head())
        else:
            logger.info("There are no numerical columns.")

        logger.info("Extracting categorical columns...")
        self.cat_data = initial_df.select_dtypes(include='category')
        if not self.cat_data.empty:
            logger.info("First 5 rows of categorical columns:")
            logger.info(self.cat_data.head())
        else:
            logger.info("There are no categorical columns.")
        logger.info("One hot encoding for categorical columns...")
        self.data_X = self._get_cat_dummies()
        logger.info("Created dummy variables using one hot encoder.")

    def _get_cat_dummies(self):
        """

        """
        if self.target_col in self.cat_data.columns.values.tolist():
            cat_feature_df = self.cat_data.drop(columns=[self.target_col])
            num_feature_df = self.num_data
        else:
            cat_feature_df = self.cat_data
            num_feature_df = self.num_data.drop(columns=[self.target_col])

        cat_col_list = cat_feature_df.columns.values.tolist()
        logger.info("Encoding categorical columns with binary " +
                    "representation...")
        cat_feature_df = pd.get_dummies(data=cat_feature_df,
                                        prefix=cat_col_list,
                                        drop_first=True)
        logger.info("Enocded categorical columns using binary " +
                    "representation:")
        logger.info(cat_feature_df.head(10))
        data_X = pd.concat([num_feature_df, cat_feature_df], axis=1)

        logger.info("Checking variance of each variable...")
        variance_filter = fs.VarianceThreshold()
        variance_filter.fit(data_X)
        variance_dict = dict(zip(data_X.columns.values.tolist(),
                                 variance_filter.variances_))
        logger.info("Variance for each feature:")
        logger.info(variance_dict)
        logger.info("Zero variance variables are:")
        const_col = [key for key in variance_dict if variance_dict[key] == 0.0]
        logger.info(const_col)

        logger.info("Removing all zero variance (constant) variables...")
        data_X = data_X.drop(columns=const_col)

        logger.info("All features that have positive variance:")
        logger.info(data_X.columns)

        logger.info("Feature matrix is: ")
        logger.info(data_X.head())
        logger.info("Target matrix is: ")
        logger.info(self.data_y.head())
        return data_X

    def rank_feature_by_importance(self):
        """Train a tree and rank features according to their relative
        importance

        """
        if self.tree_type.lower() == 'decisiontreeregressor':
            trained_tree = tree.DecisionTreeRegressor(
                    criterion='mse',
                    max_depth=10,
                    min_samples_split=float(self.algo_param_dict[
                            'min_samples_split']),
                    min_samples_leaf=float(
                            self.algo_param_dict['min_samples_leaf']),
                    random_state=int(self.algo_param_dict['random_state']),
                    min_impurity_decrease=float(self.algo_param_dict[
                            'min_impurity_decrease']))
        elif self.tree_type.lower() == 'decisiontreeclassifier':
            trained_tree = tree.DecisionTreeClassifier()
        else:
            raise ValueError("Algorithm name must be 'decisiontreeregressor'" +
                             ", or 'decisiontreeclassifier'")

        logger.info("Start to get feature relative importance...")
        trained_tree.fit(X=self.data_X, y=self.data_y)
        feat_impt = pd.DataFrame(
                {'feature_name': self.data_X.columns.values.tolist(),
                 'importance': trained_tree.feature_importances_})
        feat_impt.sort_values(by=['importance'], ascending=False, inplace=True)
        logger.info("Feature importances:")
        logger.info(feat_impt)
        return feat_impt

    def grow_by_height(self):
        """

        """
        logger.info("Growing tree by height...")
        if self.tree_type.lower() == 'decisiontreeregressor':
            for height in range(1, int(self.algo_param_dict['max_depth']) + 1):
                logger.info("A tree of hieght {}".format(height))
                trained_tree = tree.DecisionTreeRegressor(
                        criterion='mse',
                        max_depth=height,
                        min_samples_split=float(self.algo_param_dict[
                                'min_samples_split']),
                        min_samples_leaf=float(
                                self.algo_param_dict['min_samples_leaf']),
                        random_state=int(self.algo_param_dict['random_state']),
                        min_impurity_decrease=float(self.algo_param_dict[
                                'min_impurity_decrease']))
                trained_tree.fit(X=self.data_X, y=self.data_y)

                graph_str = self._save_tree_img(trained_tree, 'none')
                self._tree_visualizer(graph_str)
        elif self.tree_type.lower() == 'decisiontreeclassifier':
            trained_tree = tree.DecisionTreeClassifier()
        else:
            raise ValueError("Algorithm name must be 'decisiontreeregressor'" +
                             ", or 'decisiontreeclassifier'")
        return trained_tree

    def grow_by_var(self, ranked_feat_impt):
        """

        """
        logger.info("Growing tree by variables...")
        n_largest = min(int(self.algo_param_dict['max_depth']),
                        ranked_feat_impt.shape[0])
        feature_list = ranked_feat_impt.nlargest(
                n=n_largest,
                columns=['importance'])['feature_name'].tolist()
        logger.info("Top {} features are {}".format(n_largest, feature_list))
        graph_str_list = []
        if self.tree_type.lower() == 'decisiontreeregressor':
            for height in range(1, n_largest + 1):
                trained_tree = tree.DecisionTreeRegressor(
                        criterion='mse',
                        max_depth=height,
                        min_samples_split=float(self.algo_param_dict[
                                'min_samples_split']),
                        min_samples_leaf=float(
                                self.algo_param_dict['min_samples_leaf']),
                        random_state=int(self.algo_param_dict['random_state']),
                        min_impurity_decrease=float(self.algo_param_dict[
                                'min_impurity_decrease']))
                trained_tree.fit(X=self.data_X[feature_list[0:height]],
                                 y=self.data_y)

                graph_str = self._save_tree_img(trained_tree,
                                                feature_list[0:height])
                graph_str_list.append(graph_str)
                self._tree_visualizer(graph_str)
        elif self.tree_type.lower() == 'decisiontreeclassifier':
            trained_tree = tree.DecisionTreeClassifier()
        else:
            raise ValueError("Algorithm name must be 'decisiontreeregressor'" +
                             ", or 'decisiontreeclassifier'")
        return trained_tree

    def _save_tree_img(self, fitted_tree, used_feat_list='none'):
        """

        """
        if used_feat_list == 'none':
            used_var_list = self.data_X.columns.values.tolist()
        elif isinstance(used_feat_list, list):
            used_var_list = used_feat_list
        else:
            raise ValueError("Invalid argument: used_feat_list")
        dot_data = tree.export_graphviz(fitted_tree,
                                        out_file=None,
                                        feature_names=used_var_list,
                                        label='all',
                                        filled=True,
                                        impurity=False,
                                        proportion=True,
                                        rounded=True,
                                        precision=0)
        graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
        png_string = graph.create_png()
        bsio = BytesIO()
        bsio.write(png_string)
        bsio.seek(0)
        return bsio

    @staticmethod
    def _tree_visualizer(graph_bytes_str):
        """

        """
        img = mpimg.imread(graph_bytes_str)
        plt.imshow(img, aspect='equal')
        plt.axis('off')
        plt.show(block=True)

    def _extract_sibling_sereis(self, fitted_tree, sibling_tuple):
        """

        """
        leaf_idx_array = fitted_tree.apply(self.data_X)
        leaf_idx_df = pd.DataFrame({'leaf': leaf_idx_array.tolist()})
        target = self.data_y.reindex()

        left_loc_idx = leaf_idx_df.index[leaf_idx_df.leaf == sibling_tuple[0]]
        right_loc_idx = leaf_idx_df.index[leaf_idx_df.leaf == sibling_tuple[1]]
        left_leaf = target.loc[left_loc_idx]
        right_leaf = target.loc[right_loc_idx]
# =============================================================================
#         logger.info("Sample from left leaf: {}".format(left_leaf.head()))
#         logger.info("Sample from right leaf: {}".format(right_leaf.head()))
# =============================================================================
        return left_leaf, right_leaf

    @staticmethod
    def _is_leaf_node(fitted_tree, node_id):
        """

        """
        if node_id < 0:
            raise ValueError("Tree node id should be nonnegative.")
        else:
            if (fitted_tree.tree_.children_left[node_id] == -1 and
                    fitted_tree.tree_.children_right[node_id] == -1):
                return True
            else:
                return False

    def _has_two_leaves(self, fitted_tree, node_id, leaf_list):
        """For a given node return if there are two leaves under this node

        """
        if node_id < 0:
            raise ValueError("Tree node id should be nonnegative.")
        elif not leaf_list:
            raise ValueError("Tree leaf list should not be empty.")
        else:
            if self._is_leaf_node(fitted_tree, node_id):
                return False
            else:
                left_child = fitted_tree.tree_.children_left[node_id]
                right_child = fitted_tree.tree_.children_right[node_id]
                if (self._is_leaf_node(fitted_tree, left_child) and
                        self._is_leaf_node(fitted_tree, right_child) and
                        left_child + 1 == right_child):
                    return True
                else:
                    return False

    @staticmethod
    def _get_tstat_pval(sample_mean_1, sample_mean_2,
                        sample_std_1, sample_std_2,
                        sample_size_1, sample_size_2,
                        diff):
        """Calculate t-statistic and get p-value

        """
        delta_dof_1 = 1
        delta_dof_2 = 1

        sample_var_1 = (sample_std_1**2) / sample_size_1
        sample_var_2 = (sample_std_2**2) / sample_size_2

        sum_var = sample_var_1 + sample_var_2
        se = np.sqrt(sum_var)

        denom_part_1 = sample_var_1**2 / (sample_size_1 - delta_dof_1)
        denom_part_2 = sample_var_2**2 / (sample_size_2 - delta_dof_2)

        denom = denom_part_1 + denom_part_2

        degree_of_freedom = (sum_var**2) / denom

        t_statistic = ((sample_mean_1 - sample_mean_2) - np.abs(diff)) / se
        p_value = stdtr(degree_of_freedom, t_statistic)

        return t_statistic, p_value

    def t_test_sibling_leaf(self, fitted_tree):
        """

        """
        logger.info("left children: {}".format(
                fitted_tree.tree_.children_left))
        logger.info("right children: {}".format(
                fitted_tree.tree_.children_right))
        logger.info("number of nodes: {}".format(
                fitted_tree.tree_.node_count))

        leaf_idx_list = []
        sibling_tuple_list = []
        n_node = fitted_tree.tree_.node_count

        for node_id in range(n_node):
            if self._is_leaf_node(fitted_tree, node_id):
                leaf_idx_list.append(node_id)
        logger.info("Node ids for all leaves: {}".format(leaf_idx_list))

        for node_id in range(n_node):
            if self._has_two_leaves(fitted_tree, node_id, leaf_idx_list):
                sibling_tuple_list.append(
                        (fitted_tree.tree_.children_left[node_id],
                         fitted_tree.tree_.children_right[node_id]))
        logger.info("Sibling leaves: {}".format(sibling_tuple_list))

        for sibling_tuple in sibling_tuple_list:
            logger.info("Extracting sibling pair {}".format(sibling_tuple))
            left_leaf_df, right_leaf_df = self._extract_sibling_sereis(
                    fitted_tree, sibling_tuple)
            left_mean = np.mean(left_leaf_df, axis=0).values[0]
            left_std = np.std(left_leaf_df, axis=0, ddof=1).values[0]
            left_size = left_leaf_df.shape[0]
            right_mean = np.mean(right_leaf_df, axis=0).values[0]
            right_std = np.std(right_leaf_df, axis=0, ddof=1).values[0]
            right_size = right_leaf_df.shape[0]
            logger.info("Left leaf size: {}. Avg.: {}. Std. Dev.: {}".format(
                            left_size, left_mean, left_std))
            logger.info("Right leaf size: {}. Avg.: {}. Std. Dev.: {}".format(
                            right_size, right_mean, right_std))

            p_val = np.nan
            diff = float(self.algo_param_dict['groups_mean_diff'])
            if left_mean < right_mean:
                logger.info(
                        "Null hypotheses (H0): mu_R - mu_L >= {}".format(
                                self.algo_param_dict['groups_mean_diff']))
                logger.info(
                        "Alternative hypotheses (H1): mu_R - mu_L < {}".format(
                                self.algo_param_dict['groups_mean_diff']))
                t_stat, p_val = self._get_tstat_pval(right_mean, left_mean,
                                                     right_std, left_std,
                                                     right_size, left_size,
                                                     diff)
            else:
                logger.info(
                        "Null hypotheses (H0): mu_L - mu_R >= {}".format(
                                self.algo_param_dict['groups_mean_diff']))
                logger.info(
                        "Alternative hypotheses (H1): mu_L - mu_R < {}".format(
                                self.algo_param_dict['groups_mean_diff']))
                t_stat, p_val = self._get_tstat_pval(left_mean, right_mean,
                                                     left_std, right_std,
                                                     left_size, right_size,
                                                     diff)
            logger.info("t_statistic: {0:8.3f}, p-value: {1:8.5f}".format(
                    float(t_stat), float(p_val)))
            if p_val > 0.05:
                logger.info(
                        "At significance level of 0.05, " +
                        "we cannot reject null hypothesis (H0)")
            else:
                logger.info(
                        "At significance level of 0.05, " +
                        "we reject null hypothesis (H0)")

    @staticmethod
    def output_decision_rules(fitted_tree):
        """http://scikit-learn.org/stable/auto_examples/
           tree/plot_unveil_tree_structure.html
           Works only for pre-pruned trees
        """
        n_nodes = fitted_tree.tree_.node_count
        children_left = fitted_tree.tree_.children_left
        children_right = fitted_tree.tree_.children_right
        feature = fitted_tree.tree_.feature
        threshold = fitted_tree.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        logger.info("The binary tree structure has {:d} nodes and has ".format(
                n_nodes) + "the following tree structure:")
        for i in range(n_nodes):
            if is_leaves[i]:
                logger.info("{}node={} leaf node.".format(
                        node_depth[i] * "\t", i))
            else:
                logger.info("{}node={} test node: go to node {} ".format(
                        node_depth[i] * "\t", i, children_left[i]) +
                            " if X[:, {}] <= {} else to node {}.".format(
                                    feature[i],
                                    threshold[i],
                                    children_right[i]))

    def output_decision_rules_fun(self, inner_tree):
        """inner_tree is fitted_tree.tree_

        """
        feat_list = self.data_X.columns.values.tolist()
        feat_used = [feat_list[i] if i != TREE_UNDEFINED
                     else "undefined!" for i in inner_tree.feature]
#        print("def tree({}):".format(", ".join(feat_list)))

        def recurse(node, depth):
            indent = "    " * depth
            if inner_tree.feature[node] != TREE_UNDEFINED:
                name = feat_used[node]
                threshold = inner_tree.threshold[node]
                print("{0}if {1} <= {2}:".format(indent, name, threshold))
                recurse(inner_tree.children_left[node], depth + 1)
                print("{}else: # if {} > {}".format(indent, name, threshold))
                recurse(inner_tree.children_right[node], depth + 1)
            else:
                print("{}return {}".format(indent, inner_tree.value[node]))

        recurse(0, 1)

    def _prune_index(self, inner_tree, index, threshold):
        """inner_tree is fitted_tree.tree_

        """
        if np.abs(inner_tree.value[inner_tree.children_left[index]][0][0] -
                  inner_tree.value[inner_tree.children_right[index]][0][0]) <\
           threshold:

            inner_tree.feature[index] = TREE_UNDEFINED
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF

        if inner_tree.children_left[index] != TREE_LEAF:
            self._prune_index(inner_tree, inner_tree.children_left[index],
                              threshold)
            self._prune_index(inner_tree, inner_tree.children_right[index],
                              threshold)
        return inner_tree

    def get_param(self, param_name):
        """

        """
        return self.algo_param_dict[param_name]

    def set_param(self, param_name, param_val):
        """

        """
        if param_name not in ['max_depth', 'min_samples_split',
                              'min_samples_leaf', 'min_impurity_decrease',
                              'groups_mean_diff', 'prune_threshold']:
            raise ValueError('Input parameter name is not valid.')
        if not isinstance(param_val, str):
            param_val = str(param_val)
        self.algo_param_dict[param_name] = param_val
        return self

    def run(self):
        """

        """
        self.rank_feature_by_importance()
#        feat_imprt_df = self.rank_feature_by_importance()
        trained_tree = self.grow_by_height()
#        self.grow_by_var(feat_imprt_df)
        print("Decision tree max height(depth) is: {}".format(
                self.algo_param_dict['max_depth']))
        print("Decision tree min percentage to split is: {}".format(
                self.algo_param_dict['min_samples_split']))
        print("Decision tree min percentage a leaf must contain is: {}".format(
                self.algo_param_dict['min_samples_leaf']))
        print("Leaf mean diff threshold for hypotheses test is: {}".format(
                self.algo_param_dict['groups_mean_diff']))
        print("Leaf consolidation threshold is: {}".format(
                self.algo_param_dict['prune_threshold']))
        print("Decision tree min impurity_decrease is: {}".format(
                self.algo_param_dict['min_samples_leaf']))
        if self.tree_type.lower() == "decisiontreeregressor":
            self.t_test_sibling_leaf(trained_tree)
        self.output_decision_rules(trained_tree)
        if self.algo_param_dict['prune_threshold'] == 'inf':
            logger.info("Not perfoming pruning.")
        elif int(self.algo_param_dict['prune_threshold']) > 0:
            pruned_tree = self._prune_index(trained_tree.tree_, 0,
                                            int(self.algo_param_dict[
                                                    'prune_threshold']))
            self.output_decision_rules_fun(pruned_tree)
            pruned_tree = self._save_tree_img(trained_tree)
            self._tree_visualizer(pruned_tree)


def main():
    """Main function

    """
    is_finished = False
    start_time = time.time()
    logger.info("Starting superivsed segmentation...")
    input_df, cfg_content = FeatureInit()

    segmenter = SupervisedSegmentation(input_df, cfg_content)
    while not is_finished:
        segmenter.run()
        user_input = 'no'
        user_input = input("Exit this application (yes/no)?: ")
        while user_input.lower() not in ['yes', 'no']:
            user_input = input("Exit this application (yes/no)?: ")
        if user_input.lower() == 'yes':
            is_finished = True
        else:
            is_finished = False
            seg_param_list = ['max_depth',
                              'min_samples_split',
                              'min_samples_leaf',
                              'min_impurity_decrease',
                              'groups_mean_diff',
                              'prune_threshold']
            for param in seg_param_list:
                print("\nCurrent value for '" + param + "' is: {}".format(
                        segmenter.get_param(param)))
                user_input = input(
                        "Please input new value (positive integer) for " +
                        "'" + param + "' (press 'Enter' to keep this value): ")
                if user_input == '':
                    pass
                else:
                    if param in ['max_depth', 'groups_mean_diff',
                                 'prune_threshold']:
                        try:
                            user_input = int(user_input)
                        except Exception as e:
                            logger.error("User input value is not integer.")
                            raise err.InputError("Input value is not integer.")
                    else:
                        try:
                            user_input = float(user_input)
                        except Exception as e:
                            logger.error("User input value is not float.")
                            raise err.InputError("Input value is not float.")
                        if user_input > 1:
                            raise ValueError("Input value for this parameter" +
                                             " should not be greater than 1.")
                    if user_input < 0:
                        logger.info("Taking absolute value for negative " +
                                    "input...")
                        user_input = np.abs(user_input)
                    print("User's input is: {}".format(user_input))
                    segmenter.set_param(param, user_input)

    run_time = time.time() - start_time
    logger.info("Complete run time: " + str(round(float(run_time / 60), 3)) +
                " minutes")
    logger.info("Job completed successfully")


if __name__ == "__main__":
    """

    """
    main()
