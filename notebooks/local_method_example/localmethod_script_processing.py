for time_interval in [24]:  # ,48,72,96,120,144]:
    x, y, z_icustay_id, all_xy = preprocessing(folder, time_interval)  # all_xy is for compute gower distance


    # x= [[1,3,4,5],[2,3,4,6],[1,3,5,8],[1,4,7,8]] ; x is numpy array, each item represents the value of feature
    # y = [1,0,1,1] ; y is label
    # z_icustay_id = [1234,345,678,991] ; is the id for each ICU stay
    # all_xy contains feature, label, and icustay_id, but, all_xy is csv format

    # if you want to use local learning, I will sent a ppt for you, I think it may be good for you.


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
        for testing_sample_id, testing_sample_icu in zip(test_z_icustay_id, test_index):

            all_xy_0 = all_xy.loc[train_z_icustay_id] # select training samples from  5 fold
            all_xy_training = all_xy_0.append(all_xy.loc[i]) # note that , containing the i

            m = 400  # m is the number of similar cases or similar controls

            X_test = x[testing_sample_icu].reshape(1, -1)
            y_test = y[testing_sample_icu]

            # print 'start selecting......'
            
            Id_train_set = select_train_samples(testing_sample_id, all_xy_training, m, time_interval)  #  individulization
            """
            #testing_sample_id: so all testing 
            #all_xy_training:all training rows except with the single test sample appended
            #m: #
            #time_interval:w/e
            ##output: icustay_id of 200 closest training and 200 closest grower testing samples
            """
            #testing_sample_id: so all testing 
            #all_xy_training:all training rows except with the single test sample appended
            #m: #
            #time_interval:w/e
            ##output: icustay_id of 200 closest training and 200 closest grower testing samples
    
            ix = np.isin(z_icustay_id, Id_train_set)
            X_train=x[ix]# parameters for m*2 training set
            y_train=y[ix]# labels for m*2 training set
            
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