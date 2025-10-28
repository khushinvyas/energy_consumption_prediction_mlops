#!/bin/bash

# Clear previous experiments for a clean run if desired
# dvc exp purge -a

echo "Running Random Forest experiments..."
dvc exp run -n exp_rf_1 --set-param train.model.params.n_estimators=50 --set-param train.model.params.max_depth=5
dvc exp run -n exp_rf_2 --set-param train.model.params.n_estimators=100 --set-param train.model.params.max_depth=10
dvc exp run -n exp_rf_3 --set-param train.model.params.n_estimators=150 --set-param train.model.params.max_depth=15
dvc exp run -n exp_rf_4 --set-param train.model.params.n_estimators=200 --set-param train.model.params.max_depth=10
dvc exp run -n exp_rf_5 --set-param train.model.params.n_estimators=100 --set-param train.model.params.max_depth=20
dvc exp run -n exp_rf_6 --set-param train.model.params.n_estimators=100 --set-param train.model.params.min_samples_split=5
dvc exp run -n exp_rf_7 --set-param train.model.params.n_estimators=200 --set-param train.model.params.min_samples_split=5
dvc exp run -n exp_rf_8 --set-param train.model.params.n_estimators=50 --set-param train.model.params.max_depth=5 --set-param train.model.params.min_samples_leaf=2
dvc exp run -n exp_rf_9 --set-param train.model.params.n_estimators=100 --set-param train.model.params.max_depth=10 --set-param train.model.params.min_samples_leaf=4
dvc exp run -n exp_rf_10 --set-param train.model.params.n_estimators=150 --set-param train.model.params.max_depth=null --set-param train.model.params.min_samples_leaf=2


echo "Running XGBoost experiments..."
dvc exp run -n exp_xgb_1 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=50 --set-param train.model.params.max_depth=3 --set-param train.model.params.learning_rate=0.1
dvc exp run -n exp_xgb_2 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=100 --set-param train.model.params.max_depth=5 --set-param train.model.params.learning_rate=0.05
dvc exp run -n exp_xgb_3 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=150 --set-param train.model.params.max_depth=7 --set-param train.model.params.learning_rate=0.1
dvc exp run -n exp_xgb_4 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=200 --set-param train.model.params.max_depth=5 --set-param train.model.params.learning_rate=0.2
dvc exp run -n exp_xgb_5 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=100 --set-param train.model.params.max_depth=3 --set-param train.model.params.subsample=0.8
dvc exp run -n exp_xgb_6 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=150 --set-param train.model.params.max_depth=5 --set-param train.model.params.colsample_bytree=0.7
dvc exp run -n exp_xgb_7 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=100 --set-param train.model.params.max_depth=7 --set-param train.model.params.reg_alpha=0.1
dvc exp run -n exp_xgb_8 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=120 --set-param train.model.params.max_depth=4 --set-param train.model.params.learning_rate=0.08 --set-param train.model.params.gamma=0.1
dvc exp run -n exp_xgb_9 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=80 --set-param train.model.params.max_depth=6 --set-param train.model.params.learning_rate=0.15 --set-param train.model.params.min_child_weight=1
dvc exp run -n exp_xgb_10 --set-param train.model.name="XGBoostRegressor" --set-param train.model.params.n_estimators=180 --set-param train.model.params.max_depth=5 --set-param train.model.params.learning_rate=0.07 --set-param train.model.params.subsample=0.9

echo "All experiments completed."