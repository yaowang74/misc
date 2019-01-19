# -*- coding: utf-8 -*-
"""
Main entrance

@author: H211803
"""
import os.path
import pandas as pd
import h2o
import pickle
import utils as ut
import lightgbm as lgb
from keras import models
from keras import layers
from utils import keras_to_categorical
from dataloader import DataLoader
from featureengine import FeatureEngine
from loggerinitializer import InitializeLogger
from config import MODEL_SAVE_PATH, TASK_TYPE, NO_DECOMMIT
from config import DOWNSAMPLE_FACTOR, RANDOM_SEED, PREDICT_DECOMMIT_WEEK_Y_N
from config import NTHREADS, MAX_MEM_SIZE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch


logger = InitializeLogger("SupplierDecommitPrediction")


class Prediction:
    """Class for Aerospace suppliers de-commit predictive model.

    """


    def __init__(self, master_df):
        """

        :param master_df: a master dataframe that contains all necessary
        information for modeling
        """
        logger.info("Starting modeling process ... \n")

        task = TASK_TYPE.strip().upper()
        valid_task_type = ['TRAIN', 'TRAIN_AND_SCORE', 'SCORE']

        if task not in valid_task_type:
            raise ValueError(
                "Task type much be one value in: {}".format(valid_task_type))

        self.master = master_df.copy()

        logger.info("Specifying response column name ... \n")

        self.label = 'IS_DECOMMIT'
        if PREDICT_DECOMMIT_WEEK_Y_N:
            self.label = 'DECOMMIT_NOC_WEEK_NUMBER'
        self.task_type = TASK_TYPE
        self.save_model_path = MODEL_SAVE_PATH

        col_list = self.master.columns.values.tolist()
        extra_col = [
            'PONumber', 'POItemNo', 'Schedulelinenumber', 'Plant',
            'VendorName', 'vendor2', 'IS_DECOMMIT', 'DECOMMIT_NOC_WEEK_NUMBER']

        logger.info("Specifying feature names list ... \n")

        self.feature = [x for x in col_list if x not in extra_col]


    def _train_test_split(self):
        """CUSTOMIZED method for splitting training, validation and test
        data set. NOTE: DUE TO SAMPLE SAMPLE SIZE AND FEWER ACTUAL DECOMMIT,
        WE IMPLEMENT THIS NON-STANDARD FUNCTION.

        :return: X_train, y_train, X_test, y_test, X_valid, y_valid
        """
        logger.info("Performing customized train test and validation split "
                    "...\n")

        cond = self.master['DECOMMIT_NOC_WEEK_NUMBER'] == NO_DECOMMIT
        good_po = self.master.loc[cond].copy()

        dup_col_list = good_po.columns.values.tolist()

        good_po.drop_duplicates(subset=dup_col_list, inplace=True)

        good_po_ds = good_po.sample(
            frac=DOWNSAMPLE_FACTOR, random_state=RANDOM_SEED)
        bad_po = self.master.loc[~cond].copy()
        downsampled = good_po_ds.append(bad_po)

        good_po_ds_valid = good_po.sample(
            frac=DOWNSAMPLE_FACTOR, random_state=2 * RANDOM_SEED)

        downsampled_valid = good_po_ds_valid.append(bad_po)

        self.X_train = downsampled.loc[:, self.feature]
        self.y_train = downsampled.loc[:, self.label]

        self.X_valid = downsampled_valid.loc[:, self.feature]
        self.y_valid = downsampled_valid.loc[:, self.label]

        # for now we use the entire set as test data set
        self.X_test = self.master.loc[:, self.feature]
        self.y_test = self.master.loc[:, self.label]

        return self


    def _train_tensorflow_keras(self):
        """neural network model using tensorflow and keras

        :return:
        """
        logger.info("Training a model using Keras with tensorflow bakcended "
                    "... \n")

        y_train_encoded, y_train_num_class, encoder_train = \
            keras_to_categorical(self.y_train)
        y_test_encoded, y_test_num_class, encoder_test = \
            keras_to_categorical(self.y_test)

        X_train_encoded = pd.get_dummies(self.X_train)
        X_test_encoded = pd.get_dummies(self.X_test)
        # due to downsampling, some vendors or purchasing groups may not be
        # included in the training samples, so here we filter those columns
        # only appear in training data
        train_features = X_train_encoded.columns.values.tolist()
        X_test_encoded = X_test_encoded[train_features]

        nn_model = models.Sequential()
        nn_model.add(
            layers.Dense(
                1024,
                activation='relu',
                input_shape=(X_train_encoded.shape[1],)
            )
        )
        nn_model.add(
            layers.Dense(
                1024,
                activation='relu'
            )
        )
        nn_model.add(
            layers.Dense(
                1024,
                activation='relu'
            )
        )
        nn_model.add(
            layers.Dense(
                y_train_num_class,
                activation='softmax'
            )
        )

        nn_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        nn_model.fit(
            X_train_encoded,
            y_train_encoded,
            epochs=100
        )

        test_loss, test_acc = nn_model.evaluate(
            X_test_encoded,
            y_test_encoded
        )
        print('nn_model test accuracy: {}'.format(test_acc))

        self.keras_nn_model = nn_model

        result_prob = nn_model.predict(X_test_encoded)
        # prediction_ = np.argmax(to_categorical(predictions), axis = 1)
        # prediction_ = encoder_test.inverse_transform(prediction_)

        y_pred = (result_prob > 0.5)
        cm = confusion_matrix(y_test_encoded.argmax(axis=1),
                              y_pred.argmax(axis=1))

        po_info = ['PONumber', 'POItemNo', 'Schedulelinenumber', 'Vendor',
                   'PurchasingGroup', 'PartNumber', 'Plant', 'VendorName',
                   'vendor2', 'DECOMMIT_NOC_WEEK_NUMBER']
        # NOTE, test data set and master share same PO
        output = self.master[po_info].reset_index(drop=True)

        prob_score = pd.DataFrame(result_prob * 100,
                                  columns=encoder_test.classes_)
        new_order_col = [
            'NO DECOMMIT', 'WEEK_13', 'WEEK_12', 'WEEK_11', 'WEEK_10',
            'WEEK_9', 'WEEK_8', 'WEEK_7', 'WEEK_6', 'WEEK_5', 'WEEK_4',
            'WEEK_3', 'WEEK_2', 'WEEK_1']
        prob_score = prob_score.reindex(columns=new_order_col)
        prob_score.reset_index(inplace=True, drop=True)

        output = pd.concat([output, prob_score], axis=1, ignore_index=False)

        logger.info(
            "Saving model output to {} ... \n".format(MODEL_SAVE_PATH))
        output.to_csv(os.path.join(MODEL_SAVE_PATH,
                                   'Model_Output_Keras_NN.csv'),
                      index=False)

        return self


    def _train_h2o_gbm(self):
        """use h2o to train gradient boosting machine model.

        :return:
        """
        logger.info("Training model using H2O ... \n")

        h2o.init(nthreads=NTHREADS, max_mem_size=MAX_MEM_SIZE)
        # h2o.cluster().show_status()

        self.X_train.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)

        self.X_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)

        self.X_valid.reset_index(drop=True, inplace=True)
        self.y_valid.reset_index(drop=True, inplace=True)

        train = ut.concat_df_horizontally(
            [self.X_train, self.y_train])

        test = ut.concat_df_horizontally(
            [self.X_test, self.y_test])

        valid = ut.concat_df_horizontally(
            [self.X_valid, self.y_valid]
        )

        train = h2o.H2OFrame(train)
        test = h2o.H2OFrame(test)
        valid = h2o.H2OFrame(valid)

        cat_col = [
            'Vendor', 'PurchasingGroup', 'PartNumber',
            self.label]

        train = ut.h2o_to_factor(train, cat_col)
        test = ut.h2o_to_factor(test, cat_col)
        valid = ut.h2o_to_factor(valid, cat_col)

        gbm_model = H2OGradientBoostingEstimator(
            model_id='gbm_fit',
            ntrees=500,
            score_tree_interval=5,
            stopping_rounds=3,
            stopping_metric="misclassification",
            stopping_tolerance=0.0005,
            seed=RANDOM_SEED)

        hyper_params = {'max_depth': list(range(1, 30, 2))}

        gbm_model_grid = H2OGridSearch(
            model=gbm_model,
            hyper_params=hyper_params,
            grid_id='depth_grid',
            search_criteria={'strategy': "Cartesian"})

        gbm_model_grid.train(
            x=self.X_train.columns.values.tolist(),
            y=self.label,
            training_frame=train,
            validation_frame=valid)

        best_model = gbm_model_grid[0]
        best_model_perf = best_model.model_performance(test)

        print("Trained GBM model performance by h2o on test data: {}\n".format(
            best_model_perf))

        # gbm_model.score_history()

        predict = best_model.predict(test)
        print(predict.head())

        # best_model.score_history()

        result_prob = best_model.predict(test)
        result_prob = result_prob.as_data_frame()
        result_prob = result_prob.iloc[:, 1:]
        result_score = result_prob * 100

        new_order_col = [
            'NO DECOMMIT', 'WEEK_13', 'WEEK_12', 'WEEK_11', 'WEEK_10',
            'WEEK_9', 'WEEK_8', 'WEEK_7', 'WEEK_6', 'WEEK_5', 'WEEK_4',
            'WEEK_3', 'WEEK_2', 'WEEK_1']
        result_score = result_score.reindex(columns=new_order_col)
        result_score.reset_index(inplace=True, drop=True)

        po_info = ['PONumber', 'POItemNo', 'Schedulelinenumber', 'Vendor',
                   'PurchasingGroup', 'PartNumber', 'Plant', 'VendorName',
                   'vendor2', 'DECOMMIT_NOC_WEEK_NUMBER']

        output = self.master[po_info].reset_index(drop=True)

        output = pd.concat([output, result_score], axis=1, ignore_index=False)

        logger.info(
            "Saving model output to {} ... \n".format(MODEL_SAVE_PATH))
        output.to_csv(os.path.join(MODEL_SAVE_PATH,
                                   'Model_Output_h2o_GBE.csv'),
                      index=False)

        self.h2o_gbm_model = best_model

        return self


    def _train_sklearn_gb(self):
        """use h2o to train gradient boosting machine model.

        :return:
        """
        logger.info("Training model using sklearn ... \n")

        X_train_encoded = pd.get_dummies(self.X_train)
        X_test_encoded = pd.get_dummies(self.X_test)

        train_features = X_train_encoded.columns.values.tolist()
        X_test_encoded = X_test_encoded[train_features]

        gb_model = GradientBoostingClassifier()

        gb_model.fit(X_train_encoded, self.y_train)
        print(
            "Predictive model accuracy on training data set: {}.\n".format(
                gb_model.score(X_train_encoded, self.y_train)))

        result_prob = gb_model.predict_proba(X_test_encoded)

        po_info = ['PONumber', 'POItemNo', 'Schedulelinenumber', 'Vendor',
                   'PurchasingGroup', 'PartNumber', 'Plant', 'VendorName',
                   'vendor2', 'DECOMMIT_NOC_WEEK_NUMBER']
        # NOTE, test data set and master share same PO
        output = self.master[po_info].reset_index(drop=True)

        prob_score = pd.DataFrame(result_prob * 100, columns=gb_model.classes_)
        new_order_col = [
            'NO DECOMMIT', 'WEEK_13', 'WEEK_12', 'WEEK_11', 'WEEK_10',
            'WEEK_9', 'WEEK_8', 'WEEK_7', 'WEEK_6', 'WEEK_5', 'WEEK_4',
            'WEEK_3', 'WEEK_2', 'WEEK_1']
        prob_score = prob_score.reindex(columns=new_order_col)
        prob_score.reset_index(inplace=True, drop=True)

        output = pd.concat([output, prob_score], axis=1, ignore_index=False)

        logger.info(
            "Saving model output to {} ... \n".format(MODEL_SAVE_PATH))
        output.to_csv(os.path.join(MODEL_SAVE_PATH,
                                   'Model_Output_sklearn_GBC.csv'),
                      index=False)

        self.sklearn_gb_model = gb_model

        return self


    def _train_lightgbm(self):
        """use LightGBM to train gradient boosting machine model.

        :return:
        """
        logger.info("Training model using LightGBM ... \n")

        X_train_encoded = pd.get_dummies(self.X_train)
        X_test_encoded = pd.get_dummies(self.X_test)

        train_features = X_train_encoded.columns.values.tolist()
        X_test_encoded = X_test_encoded[train_features]

        _, num_class_train, encoder_train = ut.keras_to_categorical(
            self.y_train)
        _, num_class_test, encoder_test = ut.keras_to_categorical(self.y_test)

        y_train = encoder_train.transform(self.y_train)
        y_test = encoder_test.transform(self.y_test)

        lgb_train = lgb.Dataset(X_train_encoded, y_train)
        lgb_eval = lgb.Dataset(X_test_encoded, y_test,
                               reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': num_class_train,
            'metric': 'multi_error',
            'num_leaves': 300,
            'min_data_in_leaf': 100,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.4,
            'lambda_l2': 0.5,
            'min_gain_to_split': 0.2,
            'verbose': 5,
            'is_unbalance': True,
            "device": "gpu"
        }

        gbm_model = lgb.train(
                        params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=500)

        result_prob = gbm_model.predict(
            X_test_encoded, num_iteration=gbm_model.best_iteration)

        importance = gbm_model.feature_importance()
        names = gbm_model.feature_name()

        po_info = ['PONumber', 'POItemNo', 'Schedulelinenumber', 'Vendor',
                   'PurchasingGroup', 'PartNumber', 'Plant', 'VendorName',
                   'vendor2', 'DECOMMIT_NOC_WEEK_NUMBER']
        # NOTE, test data set and master share same PO
        output = self.master[po_info].reset_index(drop=True)

        prob_score = pd.DataFrame(result_prob * 100,
                                  columns=encoder_test.classes_)
        new_order_col = [
            'NO DECOMMIT', 'WEEK_13', 'WEEK_12', 'WEEK_11', 'WEEK_10',
            'WEEK_9', 'WEEK_8', 'WEEK_7', 'WEEK_6', 'WEEK_5', 'WEEK_4',
            'WEEK_3', 'WEEK_2', 'WEEK_1']
        prob_score = prob_score.reindex(columns=new_order_col)
        prob_score.reset_index(inplace=True, drop=True)

        output = pd.concat([output, prob_score], axis=1, ignore_index=False)

        logger.info(
            "Saving model output to {} ... \n".format(MODEL_SAVE_PATH))
        output.to_csv(os.path.join(MODEL_SAVE_PATH,
                                   'Model_Output_LightGBM_GBC.csv'),
                      index=False)

        self.lightgbm_gb_model = gbm_model

        return self


    @staticmethod
    def _save_model(model, filename, model_string):
        """save model under directory specified in config file

        :model: trained model object
        :filename: full directory and filename that used to save model
        :model_string: string to specify which model to be saved
        :return:
        """
        logger.info("Saving {} in {} ...\n".format(model_string, filename))
        with open(filename, 'wb') as file:
            pickle.dump(model, file)


    @staticmethod
    def _load_model(filename):
        """load previously saved model

        :param filename: the file that stores a pre-trained model
        :return: pre-trained model
        """
        logger.info("Loading model from {} ...\n".format(filename))
        with open(filename, 'rb') as file:
            pickle_model = pickle.load(file)

        return pickle_model


    def run(self):
        """entry point

        :return:
        """
        if self.task_type.strip().upper() == 'TRAIN_AND_SCORE':
            self._train_test_split()
            self._train_tensorflow_keras()
            self._train_sklearn_gb()
            self._train_lightgbm()
            self._train_h2o_gbm()


def main():
    """

    :return:
    """
    print('\n')
    logger.info("Starting Suppliers De-commit Prediction application ...\n")

    dl = DataLoader()
    dl.load()

    fe = FeatureEngine(dl)
    fe.run()

    sdp = Prediction(fe.final)
    sdp.run()

    logger.info("Suppliers De-commit Prediction application is finished.\n")


if __name__ == "__main__":
    main()
