import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, Imputer
from sklearn.model_selection import StratifiedKFold
from Compute_gower_distance import select_train_samples
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, auc, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


RANDOM_STATE = 15485867

folder = '/Users/xuzhenxing/Documents/mimic_AKI_data/real_time_prediction/features/all/dropped/xy'
# folder = './xy'


def preprocessing(folder, time_interval, isnormalized=True):
    """Data preprocessing, Preprocessing  missing data with mean imputation; Normalize continous feature with MinMaxScaler;
    Normalize categorical feature with OneHotEncoder.

    Args:
        folder: dir path of source data;
        time_interval: interval of time, can be 24,48,72,96,120,144.
    Returns:
        x: features
        y: lables

    """

    all_xy = pd.read_csv(os.path.join(folder, 'all_{}hours_test_individualization_1thousand.csv'.format(time_interval)), index_col=0)
    # print (all_xy.shape)
    # print (all_xy.columns)

    medi = ['diuretics', 'nsaid', 'radio', 'angiotensin']
    pat = ['gender', 'age', 'ethnicity']
    # Total 9 comorbidity
    comm = ['congestive_heart_failure', 'peripheral_vascular', 'hypertension',
            'diabetes', 'liver_disease', 'mi', 'cad', 'cirrhosis', 'jaundice']

    # Total 8 chartevents
    chart = ['DiasBP_min', 'DiasBP_max', 'DiasBP_first', 'DiasBP_last', 'DiasBP_slope', 'DiasBP_avg',
             'Glucose_min', 'Glucose_max', 'Glucose_first', 'Glucose_last', 'Glucose_slope', 'Glucose_avg',
             'HeartRate_min', 'HeartRate_max', 'HeartRate_first', 'HeartRate_last', 'HeartRate_slope', 'HeartRate_avg',
             'MeanBP_min', 'MeanBP_max', 'MeanBP_first', 'MeanBP_last', 'MeanBP_slope', 'MeanBP_avg',
             'RespRate_min', 'RespRate_max', 'RespRate_first', 'RespRate_last', 'RespRate_slope', 'RespRate_avg',
             'SpO2_min', 'SpO2_max', 'SpO2_first', 'SpO2_last', 'SpO2_slope', 'SpO2_avg',
             'SysBP_min', 'SysBP_max', 'SysBP_first', 'SysBP_last', 'SysBP_slope', 'SysBP_avg',
             'Temp_min', 'Temp_max', 'Temp_first', 'Temp_last', 'Temp_slope', 'Temp_avg']

    # Total 12 labvents
    lab = ['BICARBONATE_first', 'BICARBONATE_last', 'BICARBONATE_min', 'BICARBONATE_max', 'BICARBONATE_avg',
           'BICARBONATE_slope', 'BICARBONATE_count',
           'BUN_first', 'BUN_last', 'BUN_min', 'BUN_max', 'BUN_avg', 'BUN_slope', 'BUN_count',
           'CHLORIDE_first', 'CHLORIDE_last', 'CHLORIDE_min', 'CHLORIDE_max', 'CHLORIDE_avg', 'CHLORIDE_slope',
           'CHLORIDE_count',
           'CREATININE_first', 'CREATININE_last', 'CREATININE_min', 'CREATININE_max', 'CREATININE_avg',
           'CREATININE_slope', 'CREATININE_count',
           'HEMOGLOBIN_first', 'HEMOGLOBIN_last', 'HEMOGLOBIN_min', 'HEMOGLOBIN_max', 'HEMOGLOBIN_avg',
           'HEMOGLOBIN_slope', 'HEMOGLOBIN_count',
           'INR_first', 'INR_last', 'INR_min', 'INR_max', 'INR_avg', 'INR_count',
           'PLATELET_first', 'PLATELET_last', 'PLATELET_min', 'PLATELET_max', 'PLATELET_avg', 'PLATELET_slope',
           'PLATELET_count',
           'POTASSIUM_first', 'POTASSIUM_last', 'POTASSIUM_min', 'POTASSIUM_max', 'POTASSIUM_avg', 'POTASSIUM_slope',
           'POTASSIUM_count',
           'PT_first', 'PT_last', 'PT_min', 'PT_max', 'PT_avg', 'PT_count',
           'PTT_first', 'PTT_last', 'PTT_min', 'PTT_max', 'PTT_avg', 'PTT_count',
           'WBC_first', 'WBC_last', 'WBC_min', 'WBC_max', 'WBC_avg', 'WBC_slope', 'WBC_count',
           'CALCIUM_first', 'CALCIUM_last', 'CALCIUM_min', 'CALCIUM_max', 'CALCIUM_avg', 'CALCIUM_count'
           ]

    if time_interval != 24:  # The 24h data lack of the feature 'CALCIUM_slope'
        lab.append('CALCIUM_slope')
    subset = medi + pat + comm + ['avg_urine'] + ['egfr_min'] + ['label'] # note that ['avg_urine'] + ['egfr_min'] is important, ignoring if they are empty.

    all_xy = all_xy.dropna(subset=subset)

    # print ('after dropping nan in the catergorical variables, the shape is {}'.format(all_xy.shape))

    all_conti_x = all_xy[chart + lab + ['avg_urine'] + ['egfr_min'] + ['age']]
    # print (all_conti_x.shape)
    # print (all_conti_x)
    all_categ_x = all_xy[['gender'] + ['ethnicity'] + medi + comm]
    # print (all_categ_x.shape)
    # print (all_categ_x)

    # Using mean imputer after drop the nan data in medication, patient demographic data, avg_ureine, egfr_min and label
    imp = Imputer(strategy='mean', axis=0)
    all_conti_x_fitted = imp.fit_transform(all_conti_x)

    def normalize(all_conti_x_fitted, all_categ_x):
        # using the MinMaxScaler to normalization the all_x
        min_max_scaler = MinMaxScaler()
        all_conti_x_fitted = min_max_scaler.fit_transform(all_conti_x_fitted)
        # print (all_conti_x_fitted.shape, all_conti_x_fitted)
        # all_conti_x = DataFrame(all_conti_x_fitted, columns=all_conti_x.columns)
        # print (all_conti_x.shape)

        onehot_enc = OneHotEncoder(sparse=False)  # dense format
        all_categ_x_fitted = onehot_enc.fit_transform(all_categ_x)
        # print (all_categ_x_fitted.shape, all_categ_x_fitted)
        return all_conti_x_fitted, all_categ_x_fitted

    if isnormalized:
        all_conti_x_fitted, all_categ_x_fitted = normalize(all_conti_x_fitted, all_categ_x)

    x = np.hstack((all_conti_x_fitted, all_categ_x_fitted))
    # y = all_xy['label']
    # x = np.array(x)
    # y = np.array(y)
    # print (x.shape, y.shape)
    # return x, y
    y = all_xy['label']
    z_icustay_id = y.index
    x = np.array(x)
    y = np.array(y)
    z_icustay_id = np.array(z_icustay_id)

    print (x.shape, y.shape)
    return x, y, z_icustay_id, all_xy


def perf_model(pipe, param_grid, name, X_train, X_test,
               y_train, y_test, scoring, verbose=0):
    gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scoring, cv=5, n_jobs=-1, verbose=verbose)
    gs.fit(X_train, y_train)

    y_train_pred = gs.predict(X_train)
    y_test_pred = gs.predict(X_test)

    acc_train = accuracy_score(y_true=y_train, y_pred=y_train_pred)
    acc_test = accuracy_score(y_true=y_test, y_pred=y_test_pred)

    fpr, tpr, _ = roc_curve(y_train, gs.predict_proba(X_train)[:, 1])
    auc_train = auc(fpr, tpr)

    fpr, tpr, _ = roc_curve(y_test, gs.predict_proba(X_test)[:, 1])
    auc_test = auc(fpr, tpr)

    confmat_train = confusion_matrix(y_true=y_train, y_pred=y_train_pred)
    confmat_test = confusion_matrix(y_true=y_test, y_pred=y_test_pred)

    print (' best parameter: ', gs.best_params_)
    print (' training acc:%.2f auc:%.2f ' % (acc_train, auc_train))
    print (' testing acc:%.2f auc:%.2f ' % (acc_test, auc_test))

    print (' train confusion matrix:\n', confmat_train)
    print (' testing confusion matrix:\n', confmat_test)
    print (' classification report:\n', classification_report(y_test, y_test_pred))

    train_report = np.array(precision_recall_fscore_support(y_train, y_train_pred))
    train_class1_report = train_report[:, 1]
    train_metrics = list(train_class1_report[:-1])
    train_metrics.extend([acc_train, auc_train])
    print ('training metrics: precision, recall, f1-score, acc, auc')
    print (train_metrics)

    test_report = np.array(precision_recall_fscore_support(y_test, y_test_pred))
    test_class1_report = test_report[:, 1]
    test_metrics = list(test_class1_report[:-1])
    test_metrics.extend([acc_test, auc_test])
    print ('test metrics: precision, recall, f1-score, acc, auc')
    print (test_metrics)

    return train_metrics, test_metrics
    """
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (recall)")

    plt.plot(fpr, tpr, label="acc:%f auc:%f" % (acc_test, auc_test))
    plt.legend(loc="best")
    plt.show()
    plt.close()

    precision, recall, _ = precision_recall_curve(y_train, gs.predict_proba(X_train)[:,1])
    average_precision = average_precision_score(y_test, gs.predict_proba(X_test)[:,1])
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.step(precision, recall, where='post', label='AP={0:0.2f}'.format(average_precision))
    plt.legend(loc="best")
    plt.show()
    plt.close()
    """


def try_dbdt(X_train, X_test, y_train, y_test, scoring):
    gbm = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120, min_samples_leaf=60,
                                     max_features=9, subsample=0.7, random_state=10)

    param_grid = {'max_depth': list(range(3, 14, 2)), 'min_samples_split': list(range(100, 801, 200))}
    train_metrics, test_metrics = perf_model(gbm, param_grid, 'GBDT', X_train, X_test, y_train, y_test, scoring, 0)
    return train_metrics, test_metrics


def try_models_cross(X_train, X_test, y_train, y_test, scoring):#  select data cross 5 Fold
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, stratify=Y, random_state=RANDOM_STATE)
    # """
    # print ('\n\nLinear Logistic Regression with L1 Penalty')
    # lgr_l1_train_metrics, lgr_l1_test_metrics = try_lgr_l1(X_train, X_test, y_train, y_test, scoring)
    #
    # print ('\n\nLinear Logistic Regression with L2 Penalty')
    # lgr_l2_train_metrics, lgr_l2_test_metrics = try_lgr_l2(X_train, X_test, y_train, y_test, scoring)
    #
    # print ('\n\nStochastic Gradient Descent')
    # Elastic_train_metrics, Elastic_test_metrics = try_sgd(X_train, X_test, y_train, y_test, scoring)
    #
    # print ('\n\nRandom Forest')
    # rf_train_metrics, rf_test_metrics = try_rf(X_train, X_test, y_train, y_test, scoring)
    # #
    print ('\n\nGradient Boosting Decision tree')
    xgboost_train_metrics, xgboost_test_metrics = try_dbdt(X_train, X_test, y_train, y_test, scoring)



if __name__ == '__main__':
    path = './logs/individualization_24_1th.txt'
    f = open(path, 'a+')
    orig_stdout = sys.stdout
    sys.stdout = f
    for time_interval in [24]:  # ,48,72,96,120,144]:
        x, y, z_icustay_id, all_xy = preprocessing(folder, time_interval)  # all_xy is for compute gower distance

        skf = StratifiedKFold(n_splits=5)
        print '%%%%%'
        num_fold = 0
        for train_index, test_index in skf.split(x, y):
            print '***************'
            # print 'This is the '+ str(i)+' times result of '+str(n_fold)+' fold'
            X_train_0, X_test_0 = x[train_index], x[test_index]
            y_train_0, y_test_0 = y[train_index], y[test_index]

            print '#####################'

            num_fold = num_fold + 1
            print 'this is the results of the %d fold in 5 folds:' %num_fold

            print 'the number of testing samples in this fold:', test_index.size

            train_z_icustay_id = z_icustay_id[train_index] # the icustay_id of samples in training set from 5 fold
            test_z_icustay_id = z_icustay_id[test_index] # the icustay_id of samples in testing set from 5 fold

            xg_one_fold_pred = [] # obtain the pred label of testing samples for one fold using xgboost
            xg_one_fold_proba = [] # obtain the proba  of testing samples for one fold using xgboost

            lr_one_fold_pred = [] # obtain the pred label of testing samples for one fold using lr
            lr_one_fold_proba = [] # obtain the proba  of testing samples for one fold using lr

            indicator_time = 0 # the indicator
            for i, j in zip(test_z_icustay_id, test_index):
                # i_index = np.where(test_z_icustay_id == i)
                # tem_test_z_icustay_id = np.delete(test_z_icustay_id, i_index)
                testing_sample_id = i

                all_xy_0 = all_xy.loc[train_z_icustay_id] # select training samples from  5 fold
                all_xy_training = all_xy_0.append(all_xy.loc[i]) # note that , containing the i

                m = 400  # m is the number of similar cases or similar controls

                X_test_00 = x[j]
                y_test = y[j]

                X_test = X_test_00.reshape(1, -1)

                # print 'start selecting......'

                Id_train_set = select_train_samples(testing_sample_id, all_xy_training, m, time_interval)  #  individulization

                ix = np.isin(z_icustay_id, Id_train_set)
                Id_train_set_index = list(np.where(ix))

                # Id_train_set_index = np.argwhere(z_icustay_id == Id_train_set)

                X_train = x[Id_train_set_index]
                y_train = y[Id_train_set_index]

                # print 'start training......'

                # scoring = 'roc_auc'

# xgboost

                xgboost_mod = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5,
                              min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                              objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
                xgboost_mod.fit(X_train, y_train)
                xg_y_pred = xgboost_mod.predict(X_test)
                xg_y_pred_proba = xgboost_mod.predict_proba(X_test)[:,1]

                xg_one_fold_pred.append(xg_y_pred)
                xg_one_fold_proba.append(xg_y_pred_proba)

# lr 

                logreg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10, fit_intercept=True,
                                            intercept_scaling=1, class_weight='balanced', random_state=None)
                logreg.fit(X_train, y_train)
                lr_y_pred = logreg.predict(X_test)
                lr_y_pred_proba = logreg.predict_proba(X_test)[:,1]

                lr_one_fold_pred.append(lr_y_pred)
                lr_one_fold_proba.append(lr_y_pred_proba)

                indicator_time = indicator_time + 1
                # print 'the next testing sample and total samples:', indicator_time, test_index.size

            xg_y_individual_pred = np.array(xg_one_fold_pred)
            xg_y_individual_proba = np.array(xg_one_fold_proba)

            lr_y_individual_pred = np.array(lr_one_fold_pred)
            lr_y_individual_proba = np.array(lr_one_fold_proba)

            one_fold_y_test = y[test_index]

            print 'this is the result of individual predictor using xgboost:'
            print 'the acc of one fold:', accuracy_score(one_fold_y_test, xg_y_individual_pred)
            print 'the classification_report :', classification_report(one_fold_y_test, xg_y_individual_pred)
            print 'the auc of one fold:', roc_auc_score(one_fold_y_test, xg_y_individual_proba)

            print 'this is the result of individual predictor using lr:'
            print 'the acc of one fold:', accuracy_score(one_fold_y_test, lr_y_individual_pred)
            print 'the classification_report :', classification_report(one_fold_y_test, lr_y_individual_pred)
            print 'the auc of one fold:', roc_auc_score(one_fold_y_test, lr_y_individual_pred)

# using non-individual predictor for classification

            xgboost_random = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5,
                                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
            xgboost_random.fit(X_train_0, y_train_0)
            y_pred_random = xgboost_random.predict(X_test_0)
            y_proba_random = xgboost_random.predict_proba(X_test_0)[:,1]

            y_test_random = y[test_index]

            print 'this is the result of non-individual predictor using xgboost:'
            print 'the acc is:',accuracy_score(y_test_random, y_pred_random)
            print 'the classification_report:', classification_report(y_test_random, y_pred_random)
            print 'the auc is:', roc_auc_score(y_test_random, y_proba_random)

            logreg_random = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10, fit_intercept=True,
                                        intercept_scaling=1, class_weight='balanced', random_state=None)
            logreg_random.fit(X_train_0, y_train_0)
            lr_y_pred_random = logreg_random.predict(X_test_0)
            lr_y_pred_proba_random = logreg_random.predict_proba(X_test_0)[:, 1]

            print 'this is the result of non-individual predictor using lr:'
            print 'the acc is:',accuracy_score(y_test_random, lr_y_pred_random)
            print 'the classification_report:', classification_report(y_test_random, lr_y_pred_random)
            print 'the auc is:', roc_auc_score(y_test_random, lr_y_pred_proba_random)

            # break
    sys.stdout = orig_stdout
    f.close()




