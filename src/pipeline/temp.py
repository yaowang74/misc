# temporarily used for scripting

import numpy as np
import os.path
from config import RANDOM_SEED
from prediction import Prediction
import utils as ut
import pandas as pd

from dataloader import DataLoader
from featureengine import FeatureEngine

# pandas display settings
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 90)


dl = DataLoader()
dl.load()


# dl.data_dict['receipts_sheet1'].head()
# dl.data_dict['decommits_decommit'].head()
# dl.data_dict['decommits_pd commit'].head()
# dl.data_dict['givrrdataload'].head()
# dl.data_dict['po change history_sheet1'].head()


fe = FeatureEngine(dl)

fe._process_base()
fe._process_receipts()
fe._process_decommit()
fe._process_material_master()
fe._process_po_change()
# fe._put_all_parts_together()


selected_col = [
    'PONum', 'POItem', 'Schedulelinenumber', 'OriginalCommitDate',
    'Date', 'IS_DECOMMIT', 'DECOMMIT_NOC_WEEK_NUMBER']
left_on = [
    'PONumber', 'POItemNo', 'Schedulelinenumber', 'OldValue',
    'DATE_CHANGE_HAPPEN_AT']
right_on = [
    'PONum', 'POItem', 'Schedulelinenumber',
    'OriginalCommitDate', 'Date']
decommit = fe.decommit_proc.loc[:, selected_col]

drop_col = [
    'PONum', 'POItem', 'OriginalCommitDate',
    'DATE_CHANGE_HAPPEN_AT', 'Date', 'OldValue']

kwargs = {
    'left': fe.po_change_flattened,
    'right': decommit,
    'left_on': left_on,
    'right_on': right_on,
    'how': 'left'
}

fe.temp = ut.merge_and_drop(
    drop_col=drop_col, **kwargs)

fe.temp.drop_duplicates(inplace=True)
fe.temp.reset_index(drop=True, inplace=True)

left_on = ['PONumber', 'POItemNo', 'Schedulelinenumber',
           'Vendor', 'PurchasingGroup', 'PartNumber', 'Plant',
           'VendorName', 'vendor2']
right_on = left_on

kwargs = {
    'left': fe.base,
    'right': fe.temp,
    'left_on': left_on,
    'right_on': right_on,
    'how': 'left'
}

fe.final = ut.merge_and_drop(drop_col=[], **kwargs)

fe.final['DECOMMIT_NOC_WEEK_NUMBER'].fillna(
    'NO_DECOMMIT', inplace=True)

fe.final['IS_DECOMMIT'].fillna(0, inplace=True)

# fe.temp = \
#     fe.po_change_flattened.merge(
#         right=decommit, how='inner',
#         left_on=left_on, right_on=right_on)


# fe._save_to_file()

# fe.run()

sdp = SupplierDecommitPrediction(fe.final)

sdp._train_test_split()


sdp._train_sklearn_gb()
sdp._train_tensorflow_keras()
sdp._train_h2o_gbm()
# ut.id_look_up('3502370124', receipts,
#               'PO_NUMBER')



# n1 = set(decommit.PONum)
# n2 = set(pd_commit.PONum)
# n3 = set(temp.PO_NUMBER)
# n4 = set(temp.PONumber)
# len(n1)
# len(n1 & n2)
# len(n1 & n3)
# len(n1 & n4)
##################################################
from keras import models
from keras import layers
from keras.utils import to_categorical

y_train_encoded, num_classes, encoder_train = ut.keras_to_categorical(
    sdp.y_train)
y_test_encoded, num_classes, encoder_test = ut.keras_to_categorical(sdp.y_test)

X_train_encoded = pd.get_dummies(sdp.X_train)
X_test_encoded = pd.get_dummies(sdp.X_test)

train_features = X_train_encoded.columns.values.tolist()
X_test_encoded = X_test_encoded[train_features]

nn_model = models.Sequential()
nn_model.add(
    layers.Dense(
        1024,
        activation='relu',
        input_shape=(
            X_train_encoded.shape[1],
        )
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
    num_classes,
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


predictions = nn_model.predict(X_test_encoded)
# prediction_ = np.argmax(to_categorical(predictions), axis = 1)
# prediction_ = encoder_test.inverse_transform(prediction_)



##################################################
fe.final.head(20)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

X_train_encoded = pd.get_dummies(sdp.X_train)
X_test_encoded = pd.get_dummies(sdp.X_test)

train_features = X_train_encoded.columns.values.tolist()
X_test_encoded = X_test_encoded[train_features]

clf = GradientBoostingClassifier()

clf.fit(X_train_encoded, sdp.y_train)
print(
    "Predictive model accuracy on training data set: {}.\n".format(
        clf.score(X_train_encoded, sdp.y_train)))

result_prob = clf.predict_proba(X_test_encoded)

po_info = ['PONumber', 'POItemNo', 'Schedulelinenumber', 'Vendor',
           'PurchasingGroup', 'PartNumber', 'Plant', 'VendorName',
           'vendor2', 'DECOMMIT_NOC_WEEK_NUMBER']
# NOTE, test data set and master share same PO
output = sdp.master[po_info].reset_index(drop=True)

prob_score = pd.DataFrame(result_prob * 100, columns=clf.classes_)
new_order_col = [
    'NO DECOMMIT', 'WEEK_13', 'WEEK_12', 'WEEK_11', 'WEEK_10',
    'WEEK_9', 'WEEK_8', 'WEEK_7', 'WEEK_6', 'WEEK_5', 'WEEK_4',
    'WEEK_3', 'WEEK_2', 'WEEK_1']
prob_score = prob_score.reindex(columns=new_order_col)
prob_score.reset_index(inplace=True, drop=True)

output = pd.concat([output, prob_score], axis=1, ignore_index=False)
######################################################

import h2o
from config import NTHREADS, MAX_MEM_SIZE
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

h2o.init(
    nthreads=NTHREADS,
    max_mem_size=MAX_MEM_SIZE
)

h2o.cluster().show_status()

sdp.X_train.reset_index(drop=True, inplace=True)
sdp.y_train.reset_index(drop=True, inplace=True)

sdp.X_test.reset_index(drop=True, inplace=True)
sdp.y_test.reset_index(drop=True, inplace=True)

sdp.X_valid.reset_index(drop=True, inplace=True)
sdp.y_valid.reset_index(drop=True, inplace=True)


train = ut.concat_df_horizontally(
    [sdp.X_train, sdp.y_train])

test = ut.concat_df_horizontally(
    [sdp.X_test, sdp.y_test])

valid = ut.concat_df_horizontally(
    [sdp.X_valid, sdp.y_valid]
)

train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)
valid = h2o.H2OFrame(valid)


cat_col = [
    'Vendor', 'PurchasingGroup', 'PartNumber',
    sdp.label]

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

hyper_params = {'max_depth': list(range(1,30,2))}

gbm_model_grid = H2OGridSearch(
    model=gbm_model,
    hyper_params=hyper_params,
    grid_id='depth_grid',
    search_criteria={'strategy': "Cartesian"})

# The use of a validation_frame is recommended with using early stopping
gbm_model_grid.train(
    x=sdp.X_train.columns.values.tolist(),
    y=sdp.label,
    training_frame=train,
    validation_frame=valid)

best_model = gbm_model_grid[0]

gbm_model_perf = best_model.model_performance(test)

print("Trained GBM by h2o: {}\n".format(best_model))

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

output = sdp.master[po_info].reset_index(drop=True)

output = pd.concat([output, result_score], axis=1, ignore_index=False)

#############################################################
import lightgbm as lgb

X_train_encoded = pd.get_dummies(sdp.X_train)
X_test_encoded = pd.get_dummies(sdp.X_test)

train_features = X_train_encoded.columns.values.tolist()
X_test_encoded = X_test_encoded[train_features]

_, num_class_train, encoder_train = ut.keras_to_categorical(
    sdp.y_train)
_, num_class_test, encoder_test = ut.keras_to_categorical(sdp.y_test)

y_train = encoder_train.transform(sdp.y_train)
y_test = encoder_test.transform(sdp.y_test)

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
output = sdp.master[po_info].reset_index(drop=True)

prob_score = pd.DataFrame(result_prob * 100,
                          columns=encoder_test.classes_)
new_order_col = [
    'NO DECOMMIT', 'WEEK_13', 'WEEK_12', 'WEEK_11', 'WEEK_10',
    'WEEK_9', 'WEEK_8', 'WEEK_7', 'WEEK_6', 'WEEK_5', 'WEEK_4',
    'WEEK_3', 'WEEK_2', 'WEEK_1']
prob_score = prob_score.reindex(columns=new_order_col)
prob_score.reset_index(inplace=True, drop=True)

output = pd.concat([output, prob_score], axis=1, ignore_index=False)
######################################################################3










































































