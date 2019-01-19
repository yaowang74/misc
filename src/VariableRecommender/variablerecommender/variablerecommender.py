# -*- coding: utf-8 -*-
"""
Created on Fri May  4 23:47:00 2018

@author: H211803
"""
import time
# import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.feature_selection as fs
from CommonModules.utils import FeatureInit
from CommonModules.loggerinitializer import InitializeLogger
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

sns.set(style='darkgrid')
np.set_printoptions(suppress=True)

logger = InitializeLogger("VariableRecommender")


class VariableRecommender:
    """Class of variable recommender, to be finished

    """

    def __init__(self, initial_df, cfg_content):
        """

        Args:


        """
        logger.info("Target column of input dataframe is: {}."
                    .format(cfg_content["Input"]["target"]))

        self.data_y = initial_df[[cfg_content["Input"]["target"]]]

        self.target_col = cfg_content["Input"]["target"]

        self.alg_name_lst = [key for key in cfg_content["Algorithm"] if
                             cfg_content["Algorithm"].getboolean(key)]
        logger.info("Selected algorithm(s) is(are): {}"
                    .format(self.alg_name_lst))

        self.alg_prop_lst = [cfg_content["AlgoProperties"][key] for key in
                             self.alg_name_lst]
        logger.info("Parameters for selected algorithm(s) are: {}"
                    .format(self.alg_prop_lst))

        self.out_path = cfg_content['Output']
        logger.info("Output file location: {}"
                    .format(self.out_path['path'] + self.out_path['filename']))

        logger.info("Extracting numeric columns...")
        self.num_data = initial_df.select_dtypes(include=np.number)
        if not self.num_data.empty:
            logger.info(self.num_data.head())
        else:
            logger.info("There are no numeric columns.")

        logger.info("Extracting categorical columns...")
        self.cat_data = initial_df.select_dtypes(include='category')
        if not self.cat_data.empty:
            logger.info("First 5 rows of categorical columns:")
            logger.info(self.cat_data.head())
        else:
            logger.info("There are no categorical columns.")

    def _save_to_file(self, data_to_save, keyword):
        """

        """
        with open(self.out_path['path'] + self.out_path['filename'],
                  "a+") as file:
            file.write(time.ctime() + ' ' + keyword + " \r\n")
            if isinstance(data_to_save, pd.DataFrame):
                for row in data_to_save.values:
                    file.write(str(row))
                file.write("\r\n")
                file.close()
            if isinstance(data_to_save, list):
                for element in data_to_save:
                    file.write(str(element))
                file.write("\r\n")
                file.close()
        return True

    def _correlation(self):
        """

        """
        corr_mat = pd.DataFrame()
        if 'correlation' in [key.lower() for key in self.alg_name_lst]:
            logger.info("Calculating correlation matrix...")
            corr_type = self.alg_prop_lst[self.alg_name_lst.index(
                    'correlation')]
            corr_mat = self.num_data.corr(corr_type)
            logger.info("Saving correlation matrix into file...")
            self._save_to_file(corr_mat, "Correlation matrix")

            logger.info("Correlation matrix has been saved")
            logger.info("Plotting correlation matrix...")
            f, ax = plt.subplots(figsize=(14, 8))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr_mat, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
            ax.set_title("Chart for " + corr_type + " correlation matrix")
            plt.show(block=True)
        else:
            logger.info('Correlation method is not selected.')
        return corr_mat

    def _chi_square_test(self):
        """

        """
        var_chi2_score = []
        var_chi2_pval = []
        if ('chi2' in [key.lower() for key in self.alg_name_lst]) or\
           ('chi_square' in [key.lower() for key in self.alg_name_lst]):
            if self.target_col in self.cat_data.columns.values.tolist():
                cat_df_no_target = self.cat_data.drop(
                        columns=[self.target_col])

                logger.info("Performing independence test between " +
                            "categorical features and target variable...")
                chi2_score, p_val = fs.chi2(cat_df_no_target,
                                            self.cat_data[self.target_col])
                var_chi2_score = list(zip(cat_df_no_target.columns
                                          .values.tolist(), chi2_score))
                logger.info("Chi2 score: {}.".format(var_chi2_score))

                var_chi2_pval = list(zip(cat_df_no_target.columns.values
                                         .tolist(), p_val))
                logger.info("P values are: {}.".format(var_chi2_pval))

                logger.info("Saving Chi2 results in text file...")
                self._save_to_file(var_chi2_score, "Chi2 Scores")
                logger.info("Chi2 scores were saved.")

                logger.info("Saving Chi2 test p-values in text file...")
                self._save_to_file(var_chi2_pval, "Chi2 test p-values")
                logger.info("Chi2 test p-values were saved.")

                chi2_df = pd.DataFrame.from_records(data=var_chi2_score,
                                                    columns=['variable_name',
                                                             'chi2_score'])
                chi2_df.sort_values(by='chi2_score', ascending=False,
                                    inplace=True)
                logger.info("Plotting Chi2 scores...")
                f, ax = plt.subplots(figsize=(14, 8))
                sns.set_color_codes("pastel")
                sns.barplot(x="chi2_score", y="variable_name", data=chi2_df,
                            label="Chi2 scores (" + "target: " +
                            self.target_col + ")", color="b")
                ax.set_title("Bar chart for Chi2 scores (target: " +
                             self.target_col + ")")
                sns.despine(left=True, bottom=True)
                plt.show(block=True)
            else:
                logger.info('Target variable should be categorical.')
        else:
            logger.info("Chi_square method is not selected.")
        return var_chi2_score, var_chi2_pval

    def _get_cat_dummies(self):
        """Equivalent to _one_hot_encode, but drop_first is True

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
        self.data_X = pd.concat([num_feature_df, cat_feature_df], axis=1)

        logger.info("Checking variance of each variable...")
        variance_filter = fs.VarianceThreshold()
        variance_filter.fit(self.data_X)
        variance_dict = dict(zip(self.data_X.columns.values.tolist(),
                                 variance_filter.variances_))
        logger.info("Variance for each feature:")
        logger.info(variance_dict)
        logger.info("Zero variance variables are:")
        const_col = [key for key in variance_dict if variance_dict[key] == 0.0]
        logger.info(const_col)

        logger.info("Removing all zero variance (constant) variables...")
        self.data_X = self.data_X.drop(columns=const_col)

        logger.info("All features that have positive variance:")
        logger.info(self.data_X.columns)

        logger.info("Feature matrix is: ")
        logger.info(self.data_X.head())
        logger.info("Target matrix is: ")
        logger.info(self.data_y.head())
        return self

    def _calculate_vif(self):
        """

        """
        vif_df = pd.DataFrame()
        if ('vif' in [key.lower() for key in self.alg_name_lst]):
            logger.info("Calculating VIF...")
            vif_df['feature_name'] = ['Intercept'] +\
                self.data_X.columns.values.tolist()

            feature_matrix = add_constant(self.data_X)

            logger.info("Start to caculate VIF for all features...")
            vif_list = [1.0 /
                        (1.0 -
                         OLS(feature_matrix[col].values,
                             feature_matrix.loc[:, feature_matrix.columns
                                                != col].values).fit().rsquared)
                        for col in feature_matrix]
            vif_df['vif'] = vif_list
            vif_df = vif_df[vif_df.feature_name != 'Intercept']

            vif_df.sort_values(by=['vif'], ascending=False, inplace=True)
            logger.info("VIF for each feature:")
            logger.info(vif_df[:])
            logger.info("Saving VIF into file")
            self._save_to_file(vif_df, "VIF score results")
            logger.info("VIF results saved.")

            f, ax = plt.subplots(figsize=(14, 8))
            sns.set_color_codes("pastel")
            sns.barplot(x="vif", y="feature_name", data=vif_df,
                        label="Ranked VIF", color="b")
            ax.set_title("Bar chart for VIF")
            ax.set(xlim=(0, 20))
            sns.despine(left=True, bottom=True)
            plt.axvline(5, color='r')
            plt.show(block=True)
        else:
            logger.info("VIF method is not selected.")
        return vif_df

    @staticmethod
    def _check_col_type(target_col, type_str):
        """Check if the target variable is binary, raise error if not.

        :param type_str: One of:
        'continuous': an array-like of floats that are not all integers,
                      and is 1d or a column vector.
        'continuous-multioutput': a 2d array of floats that are not all
                                  integers, and both dimensions are of
                                  size > 1.
        'binary': contains <= 2 discrete values and is 1d or a column vector.
        'multiclass': contains more than two discrete values, is not a
                      sequence of sequences, and is 1d or a column vector.
        'multiclass-multioutput': a 2d array that contains more than two
                                  discrete values, is not a sequence of
                                  sequences, and both dimensions are of
                                  size > 1.
        'multilabel-indicator': a label indicator matrix, an array of two
                                dimensions with at least two columns, and at
                                most 2 unique values.
        'unknown': array-like but none of the above, such as a 3d array,
                   sequence of sequences, or an array of non-sequence objects.
        """
        tar_type = type_of_target(target_col)
        if tar_type.lower() not in [type_str.lower()]:
            return False
        return True

    def _cal_iv_single_num_col(self, num_col, tar_col, bin_num_max, var_name):
        """Binning numerical column using Spearman correlation

        """
        if VariableRecommender._check_col_type(tar_col, "binary"):
            tar_col = tar_col.astype(np.int8)
        logger.info("Start binning numerical columns...")
        max_depth = int(np.log2(bin_num_max)) + 1
        min_samples_leaf = round(0.05 * len(num_col))
        criterion = "entropy"
        scoring = 'roc_auc'
        cv = 3
        logger.info("Decision tree settings:")
        logger.info("Max depth: {}".format(max_depth))
        logger.info("Min samples leaf: {}".format(min_samples_leaf))
        logger.info("Measurement of split: {}".format(criterion))
        logger.info("Cross validation settings:")
        logger.info("Scorer for cross validation: {}".format(scoring))
        logger.info("Cross validation splitting strategy: {}-fold".format(cv))
        num_col_quart = num_col.quantile([0, 0.25, 0.5, 0.75, 1])
        logger.info("Missing values are replaced with median.")
        num_col_new = num_col.fillna(value=np.median(num_col))
        num_col_new = num_col_new.values.reshape(num_col.shape[0], 1)

        init_height = 1
        cv_score = []

        if max_depth == 1:
            opt_height = 1
        else:
            for height in range(init_height, max_depth):
                clf = DecisionTreeClassifier(criterion=criterion,
                                             max_depth=height,
                                             min_samples_leaf=min_samples_leaf)
                score = cross_val_score(estimator=clf, X=num_col_new,
                                        y=tar_col, scoring=scoring, cv=cv)
                logger.info("Average " + scoring + " score for height {} is {}"
                            .format(height, score.mean()))
                cv_score.append(score.mean())
            opt_height = np.argmax(cv_score) + init_height
        logger.info("Optimal tree height is {}".format(opt_height))
        best_clf = DecisionTreeClassifier(criterion=criterion,
                                          max_depth=opt_height,
                                          min_samples_leaf=min_samples_leaf)
        best_clf.fit(X=num_col_new, y=tar_col)
        opt_bins = best_clf.tree_\
            .threshold[best_clf.tree_.feature >= 0].tolist()
        logger.info('Interval edges from decision tree : {}'.format(opt_bins))
        opt_bins.append(num_col_quart.loc[0.00])
        opt_bins.append(num_col_quart.loc[1.00])
        logger.info('Interval edges by augmenting " +\
                    "min and max values: {}'.format(opt_bins))
        opt_bins = [np.floor(edge) if edge < 0 else np.ceil(edge)
                    for edge in opt_bins]
        logger.info('Interval edges were rounded: {}'.format(opt_bins))
        opt_bins = np.sort(opt_bins)
        opt_bins = opt_bins.tolist()
        logger.info('Sorted interval edges: {}'.format(opt_bins))

        logger.info("Quartiles are:")
        logger.info(num_col_quart)

        num_tar = pd.DataFrame({"num_var": num_col, "tar_var": tar_col,
                                "interval": pd.cut(num_col, opt_bins,
                                                   duplicates='drop',
                                                   include_lowest=True)})
        num_tar = num_tar.groupby('interval')
        logger.info(num_tar.head())
        iv_df = pd.DataFrame({'var_name': var_name,
                              'min_value': num_tar.min(axis=0).num_var,
                              'max_value': num_tar.max(axis=0).num_var,
                              'count': num_tar.count().tar_var,
                              'event': num_tar.sum(axis=0).tar_var,
                              'non_event': (num_tar.count().tar_var -
                                            num_tar.sum(axis=0).tar_var)})
        iv_df.reset_index(drop=True, inplace=True)
        logger.info(iv_df.head())
        logger.info("Calculating IV for this numerical variable...")
        iv_df = iv_df.assign(event_rate=iv_df.event/iv_df.sum(axis=0).event,
                             non_event_rate=(iv_df.non_event /
                                             iv_df.sum(axis=0).non_event))
        iv_df = iv_df.assign(woe=np.log(iv_df.event_rate/iv_df.non_event_rate),
                             iv_interval=(iv_df.event_rate -
                                          iv_df.non_event_rate) *
                             np.log(iv_df.event_rate/iv_df.non_event_rate))
        logger.info("Replacing 'inf' with 0 for info value calculation...")
        iv_df.replace([np.inf, -np.inf], 0, inplace=True)
        iv_df['iv'] = iv_df.iv_interval.sum()
        self._save_to_file(iv_df, "IV: {}".format(var_name))
        logger.info(iv_df[:])
        return iv_df

    def _cal_iv_single_cat_col(self, cat_col, tar_col, var_name):
        """

        """
        if VariableRecommender._check_col_type(tar_col, "binary"):
            tar_col = tar_col.astype(np.int8)
        logger.info("Calculating categorical variable IV...")

        cat_tar = pd.DataFrame({'cat_var': cat_col, 'tar_var': tar_col})
        cat_tar = cat_tar.groupby('cat_var', as_index=True)
        iv_df = pd.DataFrame({'var_name': var_name,
                              'min_value': cat_tar.groups,
                              'max_value': cat_tar.groups,
                              'count': cat_tar.count().tar_var,
                              'event': cat_tar.sum().tar_var,
                              'non_event': (cat_tar.count().tar_var -
                                            cat_tar.sum().tar_var)})
        iv_df = iv_df.assign(event_rate=iv_df.event/iv_df.sum().event,
                             non_event_rate=(iv_df.non_event /
                                             iv_df.sum().non_event))
        iv_df = iv_df.assign(woe=(np.log(iv_df.event_rate /
                                         iv_df.non_event_rate)),
                             iv_interval=((iv_df.event_rate -
                                           iv_df.non_event_rate) /
                                          np.log(iv_df.event_rate /
                                                 iv_df.non_event_rate)))
        iv_df = iv_df.assign(iv=iv_df.iv_interval.sum())
        self._save_to_file(iv_df, "IV: {}".format(var_name))
        print(iv_df)
        logger.info(iv_df[:])
        return iv_df

    def _cal_info_value(self):
        """

        """
        if ('info_value' in [key.lower() for key in self.alg_name_lst]) or\
           ('information_value' in [key.lower() for key in self.alg_name_lst])\
           and VariableRecommender._check_col_type(self.data_y, "binary"):
            logger.info("Calculating information value...")
            iv_bin_max = int(
                self.alg_prop_lst[self.alg_name_lst.index("info_value")])
            if iv_bin_max < 2:
                raise ValueError("Max bin numbers has to be greater than 1.")
            num_iv = pd.DataFrame(columns=['var_name', 'iv'])
            cat_iv = pd.DataFrame(columns=['var_name', 'iv'])
            if not self.num_data.empty:
                temp_df = self.num_data.reset_index(drop=True)
                num_var_list = temp_df.columns.values.tolist()
                num_var_len = len(num_var_list)
                for i in range(0, num_var_len):
                    num_col_name = num_var_list[i]
                    single_num_iv =\
                        self._cal_iv_single_num_col(
                                temp_df.iloc[:, i],
                                self.data_y.iloc[:, 0],
                                iv_bin_max, num_col_name)
                    single_num_iv =\
                        single_num_iv[['var_name', 'iv']].drop_duplicates()
                    num_iv = num_iv.append(single_num_iv)
            else:
                logger.info("There are no numerical columns in data set.")

            if not self.cat_data.empty:
                temp_df = self.cat_data.drop(
                        columns=[self.target_col]).reset_index(drop=True)
                cat_var_list = temp_df.columns.values.tolist()
                cat_var_len = len(cat_var_list)
                for i in range(0, cat_var_len):
                    single_cat_iv =\
                        self._cal_iv_single_cat_col(
                                temp_df.iloc[:, i],
                                self.data_y.iloc[:, 0], cat_var_list[i])
                    single_cat_iv =\
                        single_cat_iv[['var_name', 'iv']].drop_duplicates()
                    cat_iv = cat_iv.append(single_cat_iv)
            else:
                logger.info("There are no categorical columns in data set.")
            final_iv_df = num_iv.append(cat_iv)
            final_iv_df.sort_values(by=['iv'], inplace=True, ascending=False)
            final_iv_df = final_iv_df.reset_index(drop=True)
            logger.info(final_iv_df[:])
            logger.info("Saving information values results to file...")
            self._save_to_file(final_iv_df, "Information Values")

            f, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(x="iv", y="var_name", data=final_iv_df,
                        label="Ranked Information Value", palette='BuGn_d')
            ax.set_title("Bar chart for information value")
            ax.set(xlim=(0, 1.5))
            sns.despine(left=True, bottom=True)
            plt.show(block=True)
            return final_iv_df
        else:
            logger.info("Information value method is not selected.")

    def _revursive_elimination(self):
        """Recursive feature selection using gradient boosting

        """
        if ('recursive' in [key.lower() for key in self.alg_name_lst]):
            logger.info("Start to performe recursive elimination method...")
            task_type = self.alg_prop_lst[self.alg_name_lst.index('recursive')]
            logger.info("The task is a " + task_type + " problem")
            learning_rate = 0.1
            n_estimators = 50
            max_depth = 3
            criterion = 'friedman_mse'
            min_samples_leaf = 0.05
            alpha = 0.9
            random_state = 74
            logger.info("learning_rate is {}.".format(learning_rate))
            logger.info("n_estimators is {}.".format(n_estimators))
            logger.info("max_depth is {}.".format(max_depth))
            logger.info("criterion is {}.".format(criterion))
            logger.info("min_samples_leaf is {}.".format(min_samples_leaf))
            logger.info("alpha is {}.".format(alpha))
            logger.info("random_state is {}.".format(random_state))
            if task_type == 'classification':

#                loss = 'deviance'
#                logger.info('loss function is {}.'.format(loss))
#                estimator = GradientBoostingClassifier(
#                        loss=loss,
#                        learning_rate=learning_rate,
#                        n_estimators=n_estimators,
#                        max_depth=max_depth,
#                        criterion=criterion,
#                        min_samples_leaf=min_samples_leaf,
#                        random_state=random_state)
#                logger.info("Cross validation parameters are fixed for now.")
#                logger.info("Each step remove {} feature.".format(1))
#                logger.info("Number of stratified folds is {}.".format(3))
#                logger.info("Scoring criterion is {}.".format('f1'))

                estimator = LogisticRegression()
                rfecv = fs.RFECV(
                        estimator=estimator,
                        step=1,
                        cv=StratifiedKFold(3),
                        scoring='f1')

                logger.info("Total number of columns: {}".format(
                        self.data_X.shape[1]))

                rfecv.fit(self.data_X, self.data_y.values.ravel())
                logger.info("Number of feature selected: {}."
                            .format(rfecv.n_features_))
            elif task_type == 'regression':
                loss = 'huber'
                logger.info('loss function is {}.'.format(loss))
                estimator = GradientBoostingRegressor(
                        loss=loss,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        criterion=criterion,
                        min_samples_leaf=min_samples_leaf,
                        alpha=alpha,
                        random_state=random_state)
                logger.info("Cross validation parameters are fixed for now.")
                logger.info("Each step remove {} feature.".format(1))
                logger.info("Number of folds is {}.".format(3))
                logger.info("Scoring criterion is {}.".format('r2'))
                rfecv = fs.RFECV(
                        estimator=estimator,
                        step=1,
                        cv=KFold(3),
                        scoring='r2')
                rfecv.fit(self.data_X, self.data_y.values.ravel())
                logger.info("Number of feature selected: {}."
                            .format(rfecv.n_features_))
            else:
                raise ValueError("Algorithm property for recusive feature " +
                                 "selection has to be either " +
                                 "'classification' or 'regression'")
#            f, ax = plt.subplots(figsize=(14, 8))
#            sns.barplot(list(range(1, len(rfecv.grid_scores_) + 1)),
#                        rfecv.grid_scores_)
#            ax.set_title("Bar chart for information value")
#            plt.show(block=True)
        else:
            logger.info("Feature selection using recursive method is not " +
                        "selected.")
        return rfecv

    def run(self):
        """

        """
        logger.info("General information on numerical sub dataframe:")
        if (not self.num_data.empty) and (self.alg_prop_lst):
            logger.info(str(self.num_data.describe()))
            self._correlation()
        else:
            logger.info("There is no numerical columns...")

        logger.info("General information on categorical sub dataframe:")
        if not self.cat_data.empty:
            logger.info(str(self.cat_data.describe()))
            self._chi_square_test()
            self._get_cat_dummies()
        else:
            logger.info("There is no categorical columns")
        self._calculate_vif()
        self._cal_info_value()
        rfecv = self._revursive_elimination()
        logger.info(rfecv.estimator_)


def main():
    """Main function

    """
    start_time = time.time()
    logger.info("Starting variable recommender...")
    input_df, cfg_content = FeatureInit()

#    del input
    recommender = VariableRecommender(input_df, cfg_content)
    recommender.run()

    run_time = time.time() - start_time
    logger.info("Complete run time: " + str(round(float(run_time / 60), 3)) +
                " minutes")
    logger.info("Job completed successfully")


if __name__ == "__main__":
    """

    """
    main()
