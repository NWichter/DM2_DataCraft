import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import random
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import RFECV,SelectKBest, SequentialFeatureSelector, f_classif, mutual_info_classif

from sklearn.impute import SimpleImputer, KNNImputer


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

from feature_engine.encoding import CountFrequencyEncoder

from xgboost import XGBClassifier

import matplotlib.pyplot as plt


class ModelSelection:
    def __init__(self,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_test: pd.DataFrame = None,
                 y_test: pd.DataFrame = None,
                 numerical_cols: list = None,
                 one_hot_cols: list = None,
                 freq_encod_cols: list = None,
                 ordinal_cols: list = None,
                 estimators: list = None,
                 cost_matrix: np.array = None
                ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.numerical_cols = numerical_cols
        self.one_hot_cols = one_hot_cols 
        self.freq_encod_cols = freq_encod_cols
        self.ordinal_cols = ordinal_cols
        self.estimators = estimators
        self.cost_matrix = cost_matrix
        
        self.grid_object = None
        self.grid_best_model = None
        self.grid_best_parameters = None
        self.grid_cv_results = None
        self.grid_best_score = None
        
        self.encoded_y_train = None
        
        self.col_transformer = None
        self.frequency_encoder = None
        
        self.f1_cv_results = dict()
        self.accuracy_cv_results = dict()
        self.auc_cv_results = dict()
        
        self.mean_result_f1 = dict()
        self.mean_result_accuracy = dict()
        self.mean_result_auc = dict()
        self.mean_result_cost_matrix = dict()
        
        self.best_estimator_name_f1 = None
        self.best_estimator_name_accuracy = None
        self.best_estimator_name_auc = None
        self.best_estimator_name_cost_matrix = None
        self.best_estimator_obj_f1 = None
        self.best_estimator_obj_accuracy = None
        self.best_estimator_obj_auc = None
        self.best_estimator_obj_cost_matrix = None
        
        self.target_label_mapping = None
        self.target_label_encoder = None
        
        self.test_preds = None
        self.test_f1_score = None
        self.test_auc_score = None
        self.test_accuracy_score = None
        
        self.error_cost_matrix_results = dict()
        self.only_train_model = None
        self.only_x_train_preds = None

        self.transformers = list()
        
    def encode_y_train(self):
        lab_encoder = LabelEncoder()
        self.encoded_y_train = lab_encoder.fit_transform(self.y_train.values.ravel())
        self.target_label_encoder = lab_encoder
        self.target_label_mapping = {
            label: category for label, category in enumerate(lab_encoder.classes_)
        }

    def create_numerical_col_pipe(self):
        return ("scaler", StandardScaler(), self.numerical_cols)
        
    def create_nominal_onehot_col_pipe(self):
        return ("ohe_encoder", OneHotEncoder(handle_unknown='ignore', drop="first"), self.one_hot_cols)
        
    def create_ordinal_col_pipe(self):
        return ("binary_encoder", LabelEncoder(), self.ordinal_cols)

    def create_col_transformer(self):
        if self.numerical_cols is not None:
            self.transformers.append(self.create_numerical_col_pipe())
        if self.one_hot_cols is not None:
            self.transformers.append(self.create_nominal_onehot_col_pipe())
        if self.ordinal_cols is not None:
            self.transformers.append(self.create_ordinal_col_pipe())
        
        if self.freq_encod_cols is not None:
            freq_encoder = CountFrequencyEncoder(
                encoding_method='frequency',
                variables=freq_encod_cols
            )
            freq_encoder.fit(self.x_train)
            self.x_train = freq_encoder.transform(self.x_train)
            self.frequency_encoder = freq_encoder
            

        self.col_transformer = ColumnTransformer(
            transformers=self.transformers
        )
        
    def get_estimator_by_name(self, estimator_name):
        for name, estimator in self.estimators:
            if name == estimator_name:
                return estimator
        return None
        
    def calculate_cv_f1(self, n_folds, scoring_average='f1_weighted'): # 'f1_micro' or 'f1_macro'
        for name, estimator in self.estimators:
            if len(self.transformers) == 0:
                f1_pipe = Pipeline(
                    [
                        ('ClassificationModel', estimator)
                    ]
                )
            else:
                f1_pipe = Pipeline(
                    [
                        ('ColumnTransformers', self.col_transformer), 
                        ('ClassificationModel', estimator)
                    ]
                )
            
            stratif_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
            
            f1_cv_score = cross_val_score(
                f1_pipe, 
                self.x_train, 
                self.encoded_y_train, 
                cv=stratif_cv, 
                scoring=scoring_average,
                n_jobs=-1
            )
            
            self.f1_cv_results[name] = f1_cv_score
            self.mean_result_f1[name] = f1_cv_score.mean()
        
        self.best_estimator_name_f1 = max(self.mean_result_f1, key=self.mean_result_f1.get)
        self.best_estimator_obj_f1 = self.get_estimator_by_name(self.best_estimator_name_f1)
        
        print("CV Results for Mean F1 Score:\n")
        for name, f1 in self.mean_result_f1.items():
            print(f"{name} = {f1:.6f}")
        print(f"\nBest Estimator (F1): {self.best_estimator_name_f1}")
        
    def calculate_cv_auc(self, n_folds, scoring_average='roc_auc_ovo_weighted'): #or 'roc_auc_ovr_weighted'

        for name, estimator in self.estimators:
            if len(self.transformers) == 0:
                auc_pipe = Pipeline(
                    [
                        ('ClassificationModel', estimator)
                    ]
                )
            else:
                auc_pipe = Pipeline(
                    [
                        ('ColumnTransformers', self.col_transformer), 
                        ('ClassificationModel', estimator)
                    ]
                )
            
            stratif_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
            
            auc_cv_score = cross_val_score(
                auc_pipe, 
                self.x_train, 
                self.encoded_y_train, 
                cv=stratif_cv, 
                scoring=scoring_average, 
                n_jobs=-1
            )
            
            self.auc_cv_results[name] = auc_cv_score
            self.mean_result_auc[name] = auc_cv_score.mean()
        
        self.best_estimator_name_auc = max(self.mean_result_auc, key=self.mean_result_auc.get)
        self.best_estimator_obj_auc = self.get_estimator_by_name(self.best_estimator_name_auc)
        
        print("CV Results for Mean AUC Score:\n")
        for name, auc in self.mean_result_auc.items():
            print(f"{name} = {auc:.6f}")
        print(f"\nBest Estimator (AUC): {self.best_estimator_name_auc}")
    
    def calculate_cv_accuracy(self, n_folds, scoring_average='accuracy'):
        for name, estimator in self.estimators:
            if len(self.transformers) == 0:
                accuracy_pipe = Pipeline(
                    [
                        ('ClassificationModel', estimator)
                    ]
                )
            else:
                accuracy_pipe = Pipeline(
                    [
                        ('ColumnTransformers', self.col_transformer), 
                        ('ClassificationModel', estimator)
                    ]
                )
            
            stratif_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
            
            accuracy_cv_score = cross_val_score(
                accuracy_pipe, 
                self.x_train, 
                self.encoded_y_train, 
                cv=stratif_cv, 
                scoring=scoring_average, 
                n_jobs=-1
            )
            
            self.accuracy_cv_results[name] = accuracy_cv_score
            self.mean_result_accuracy[name] = accuracy_cv_score.mean()
        
        self.best_estimator_name_accuracy = max(self.mean_result_accuracy, key=self.mean_result_accuracy.get)
        self.best_estimator_obj_accuracy = self.get_estimator_by_name(self.best_estimator_name_accuracy)
        
        print("CV Results for Mean Accuracy Score:\n")
        for name, acc in self.mean_result_accuracy.items():
            print(f"{name} = {acc:.6f}")
        print(f"\nBest Estimator (Accuracy): {self.best_estimator_name_accuracy}")

    def calculate_cost_matrix_cv2(self, n_folds=5):
        for name, estimator in self.estimators:
            if len(self.transformers) == 0:
                cost_matrix_pipe = Pipeline(
                    [
                        ('ClassificationModel', estimator)
                    ]
                )
            else:
                cost_matrix_pipe = Pipeline(
                    [
                        ('ColumnTransformers', self.col_transformer), 
                        ('ClassificationModel', estimator)
                    ]
                )
                
            stratif_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
            
            y_pred = cross_val_predict(cost_matrix_pipe, self.x_train, self.encoded_y_train, cv=stratif_cv)
            confusion_matrix_data = confusion_matrix(self.encoded_y_train, y_pred)

            """
            print(f'CV Confusion Matrix for {estimator}')
            print(self.target_label_mapping)
            plt.figure(figsize=(4, 3))
            sns.heatmap(confusion_matrix_data, annot=True, cmap='Reds', fmt='g')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()"""
            
            error_value = np.sum(confusion_matrix_data * self.cost_matrix) / len(self.encoded_y_train)            
            self.error_cost_matrix_results[name] = error_value
            
        self.best_estimator_name_cost_matrix = min(self.error_cost_matrix_results, key=self.error_cost_matrix_results.get)
        self.best_estimator_obj_cost_matrix = self.get_estimator_by_name(self.best_estimator_name_cost_matrix)
        
        print("CV Results for Cost Matrix Error Score:\n")
        for name, cost_err in self.error_cost_matrix_results.items():
            print(f"{name} = {cost_err:.6f}")
        print(f"\nBest Estimator (Cost Metric Error Score): {self.best_estimator_name_cost_matrix}")
  
    def calculate_cost_matrix_error_cv(self, custom_cost_func, n_folds=5):
        for name, estimator in self.estimators:
            if len(self.transformers) == 0:
                cost_matrix_pipe = Pipeline(
                    [
                        ('ClassificationModel', estimator)
                    ]
                )
            else:
                cost_matrix_pipe = Pipeline(
                    [
                        ('ColumnTransformers', self.col_transformer), 
                        ('ClassificationModel', estimator)
                    ]
                )
            
            stratif_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
            cost_matrix_cv_score = cross_val_score(
                cost_matrix_pipe, 
                self.x_train, 
                self.encoded_y_train, 
                cv=stratif_cv, 
                scoring=custom_cost_func, 
                n_jobs=-1,
                error_score="raise"
            )
            
            self.error_cost_matrix_results[name] = -cost_matrix_cv_score
            self.mean_result_cost_matrix[name] = -cost_matrix_cv_score.mean()    


        self.best_estimator_name_cost_matrix = min(self.mean_result_cost_matrix, key=self.mean_result_cost_matrix.get)
        self.best_estimator_obj_cost_matrix = self.get_estimator_by_name(self.best_estimator_name_cost_matrix)
        
        print("CV Results for Cost Matrix Error Score:\n")
        for name, cost_err in self.mean_result_cost_matrix.items():
            print(f"{name} = {cost_err:.6f}")
        print(f"\nBest Estimator (Cost Metric Error Score): {self.best_estimator_name_cost_matrix}")
        
        
    def print_classification_report_cv(self, estimator, n_folds=10):
        if len(self.transformers) == 0:
            pipe = Pipeline(
                [
                    ('ClassificationModel', estimator)
                ]
            )
        else:
            pipe = Pipeline(
                [
                    ('ColumnTransformers', self.col_transformer), 
                    ('ClassificationModel', estimator)
                ]
            )

        stratif_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        y_pred = cross_val_predict(pipe, self.x_train, self.encoded_y_train, cv=stratif_cv)
        print(f'CV Classification Report Result for {estimator}')
        print(self.target_label_mapping)
        print(classification_report(self.encoded_y_train, y_pred))

    def plot_confusion_matrix_cv(self, estimator, figsize=(5, 5), n_folds=10):
        if len(self.transformers) == 0:
            pipe = Pipeline(
                [
                    ('ClassificationModel', estimator)
                ]
            )
        else:
            pipe = Pipeline(
                [
                    ('ColumnTransformers', self.col_transformer), 
                    ('ClassificationModel', estimator)
                ]
            )
        
        stratif_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        y_pred = cross_val_predict(pipe, self.x_train, self.encoded_y_train, cv=stratif_cv)
        cm = confusion_matrix(self.encoded_y_train, y_pred)
        
        print(f'CV Confusion Matrix for {estimator}')
        print(self.target_label_mapping)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, cmap='Reds', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
    def apply_grid_cv(
        self, 
        estimator=None, 
        params=None, 
        cv=5, 
        scoring='f1_weighted'
    ):
        if len(self.transformers) == 0:
            pipe = Pipeline(
                [
                    ('ClassificationModel', estimator)
                ]
            )
        else:
            pipe = Pipeline(
                [
                    ('ColumnTransformers', self.col_transformer), 
                    ('ClassificationModel', estimator)
                ]
            )
            
        grid_search = GridSearchCV(
            estimator=pipe, 
            param_grid=params, 
            cv=cv, 
            n_jobs=-1, 
            verbose=2, 
            scoring=scoring,
            error_score="raise"
        )
        grid_search.fit(self.x_train, self.encoded_y_train)
        
        self.grid_object = grid_search
        self.grid_best_parameters = self.grid_object.best_params_
        self.grid_best_model = self.grid_object.best_estimator_
        self.grid_cv_results = self.grid_object.cv_results_
        self.grid_best_score = self.grid_object.best_score_
        
        print(f"Best parameters found by GridSearchCV ({scoring}):\n{self.grid_best_parameters}")
        print(f"\nBest score found by GridSearchCV ({scoring}):\n{self.grid_best_score}")
        
    def predict_on_test(self):
        self.test_preds = self.grid_best_model.predict(self.x_test)
        print(self.target_label_mapping)
        print(classification_report(self.target_label_encoder.transform(self.y_test.values.ravel()), self.test_preds))
        
        self.test_f1_score = f1_score(
            self.target_label_encoder.transform(self.y_test.values.ravel()), 
            self.test_preds, 
            average='weighted'
        )
        
        self.test_auc_score = roc_auc_score(
            self.target_label_encoder.transform(self.y_test.values.ravel()), 
            self.grid_best_model.predict_proba(self.x_test), 
            average='weighted',
            multi_class='ovo'
        )
        
        self.test_accuracy_score = accuracy_score(
            self.target_label_encoder.transform(self.y_test.values.ravel()),
            self.test_preds
        )
        
        cm = confusion_matrix(self.target_label_encoder.transform(self.y_test.values.ravel()), self.test_preds)
        print(self.target_label_mapping)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, cmap='Reds', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

def apply_one_hot_encoding(X_train, categorical_column):
    """
    Apply one-hot encoding to a categorical column in the DataFrame.

    Parameters:
        data (DataFrame): Input DataFrame containing features.
        categorical_column (str): Name of the categorical column to be one-hot encoded.

    Returns:
        ohe_encoder (OneHotEncoder): One-hot encoder object fitted on categorical data.
        X_encoded (DataFrame): DataFrame with categorical column replaced by one-hot encoded columns.
    """
    X_train_copy = X_train.copy()
    ohe_encoder = OneHotEncoder()

    X_train_copy_encoded = pd.DataFrame(
        ohe_encoder.fit_transform(X_train_copy[[categorical_column]]).toarray(),
        columns=ohe_encoder.get_feature_names_out([categorical_column])
    )

    X_train_encoded_last = pd.concat([X_train_copy.drop(columns=[categorical_column]).reset_index(drop=True), X_train_copy_encoded], axis=1)

    return X_train_encoded_last, ohe_encoder


def apply_smote(X, y, random_state=0):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

    Parameters:
        X (DataFrame): Input DataFrame containing features.
        y (Series): Target variable.
        random_state (int): Random state for reproducibility.

    Returns:
        smote_x (DataFrame): Resampled DataFrame with balanced features.
        smote_y (Series): Resampled target variable.
    """
    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X, y)

    smote_x = pd.DataFrame(X_smote, columns=X.columns)
    smote_y = y_smote

    return smote_x, smote_y


def apply_random_oversampling(X, y, random_state=0):
    """
    Apply RandomOverSampler to balance the dataset.

    Parameters:
        data (DataFrame): Input DataFrame containing features and target variable.
        random_state (int): Random state for reproducibility.

    Returns:
        oversampled_X (DataFrame): Resampled DataFrame with balanced features.
        oversampled_y (Series): Resampled target variable.
    """
    ros = RandomOverSampler(random_state=random_state)
    X_ros, y_ros = ros.fit_resample(X, y)

    oversampled_X = pd.DataFrame(X_ros, columns=X.columns)
    oversampled_y = y_ros

    return oversampled_X, oversampled_y


def apply_random_undersampling(X, y, random_state=0):
    """
    Apply RandomUnderSampler to balance the dataset.

    Parameters:
        data (DataFrame): Input DataFrame containing features and target variable.
        random_state (int): Random state for reproducibility.

    Returns:
        undersampled_X (DataFrame): Resampled DataFrame with balanced features.
        undersampled_y (Series): Resampled target variable.
    """
    rus = RandomUnderSampler(random_state=random_state)
    X_rus, y_rus = rus.fit_resample(X, y)

    undersampled_X = pd.DataFrame(X_rus, columns=X.columns)
    undersampled_y = y_rus

    return undersampled_X, undersampled_y



def apply_std_scaler(X, columns):
    """
    Apply standard scaling to specified columns of the data using a pre-fitted scaler.

    Parameters:
        data (DataFrame): Input DataFrame containing features.
        columns (list): List of column names to apply standard scaling to.

    Returns:
        scaled_data (DataFrame): DataFrame with specified columns scaled.
        scaler (StandardScaler): Fitted StandardScaler object.
    """
    std_scaler = StandardScaler()
    X_copy = X.copy()
    X_scaled = pd.DataFrame(std_scaler.fit_transform(X_copy[columns]), columns=X_copy[columns].columns).reset_index(drop=True)
    X_copy_cat = X_copy.loc[:, X_copy.columns.str.contains('Group')].reset_index(drop=True)
    X_scaled_last = pd.concat([X_scaled, X_copy_cat], axis=1)
    
    return X_scaled_last, std_scaler


def handle_missing_vals_simple(data, strategy: str = 'median'):
    """
    Handle missing values in a DataFrame by imputing them using the specified strategy.

    Parameters:
        data (DataFrame): Input DataFrame containing the data with missing values.
        strategy (str): The imputation strategy. Possible values are 'mean', 'median', 'most_frequent', or 'constant'.
            Defaults to 'median'.
        columns (list): A list of columns in which missing values should be handled. If None, missing values
            will be handled in all columns. Defaults to None.

    Returns:
        data_imputed (DataFrame): DataFrame with missing values imputed using the specified strategy.
        imputer (SimpleImputer): The fitted imputer instance used for imputation.
    """
    data_copy = data.copy()
    imputer = SimpleImputer(strategy=strategy)

    cat_df = data_copy.loc[:, data_copy.columns.str.contains('Group')].reset_index(drop=True)
    num_df = data_copy.loc[:, ~data_copy.columns.str.contains('Group')].reset_index(drop=True)
    
    num_df = pd.DataFrame(imputer.fit_transform(num_df), columns=num_df.columns)
    data_last = pd.concat([num_df, cat_df], axis=1)
    
    return data_last, imputer



def handle_missing_vals_knn(data, n_neighbors=5):
    """
    Handle missing values in a DataFrame by imputing them using KNNImputer.

    Parameters:
        data (DataFrame): Input DataFrame containing the data with missing values.
        n_neighbors (int): Number of neighboring samples to use for imputation.
        columns (list): A list of columns in which missing values should be handled. If None, missing values
            will be handled in all columns. Defaults to None.

    Returns:
        data_imputed (DataFrame): DataFrame with missing values imputed using KNNImputer.
        imputer (KNNImputer): The fitted KNNImputer instance used for imputation.
    """
    data_copy = data.copy()
    cat_df = data_copy.loc[:, data_copy.columns.str.contains('Group')].reset_index(drop=True)
    num_df = data_copy.loc[:, ~data_copy.columns.str.contains('Group')].reset_index(drop=True)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    num_df = pd.DataFrame(imputer.fit_transform(num_df), columns=num_df.columns)
    data_last = pd.concat([num_df, cat_df], axis=1)
    
    return data_last, imputer


def detect_outliers_with_lof(data, n_neighbors=20):
    """
    Detect outliers using Local Outlier Factor (LOF) algorithm.

    Parameters:
        data (DataFrame): Input DataFrame containing the data.
        columns (list): List of columns to consider for outlier detection.
        n_neighbors (int): Number of neighbors to consider for LOF algorithm.

    Returns:
        clean_data (DataFrame): DataFrame without the detected outliers.
        clf (LocalOutlierFactor): Fitted LOF model.
    """
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    outlier_labels = clf.fit_predict(data)

    clean_data = data.loc[outlier_labels != -1, :]

    return clean_data, clf

def replace_outliers_with_thresholds(dataframe, columns, target):    
    dataframe_copy = dataframe.copy()
    
    for variable in columns:
        if variable != target:
            quartile1 = dataframe_copy[variable].quantile(0.01)
            quartile3 = dataframe_copy[variable].quantile(0.99)
            interquantile_range = quartile3 - quartile1
            up_limit = quartile3 + 1.5 * interquantile_range
            low_limit = quartile1 - 1.5 * interquantile_range
            
            dataframe_copy.loc[(dataframe_copy[variable] < low_limit), variable] = low_limit
            dataframe_copy.loc[(dataframe_copy[variable] > up_limit), variable] = up_limit
    
    return dataframe_copy

def remove_collinear(dataframe, threshold=0.7):
    """
    Remove collinear features from the DataFrame based on the correlation coefficient.

    Collinear features are highly correlated features that may introduce redundancy in the data and negatively impact the performance and interpretability of machine learning models.

    Parameters:
        dataframe (DataFrame): The input DataFrame containing features.
        threshold (float): The threshold value for the correlation coefficient.
            Features with a correlation coefficient greater than or equal to this threshold will be considered collinear.
            Defaults to 0.7.

    Returns:
        DataFrame: DataFrame with collinear features removed.

    Notes:
        This function iteratively removes one feature from each pair of collinear features until no more collinear features remain above the specified threshold.
        It prints the names of the removed columns during each iteration.

    Example:
        # Remove collinear features from the DataFrame with a threshold of 0.8
        clean_data = remove_collinear(data, threshold=0.8)
    """
    dataf = dataframe.copy()
    while True:
        corr_matrix = dataf.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        if (upper >= threshold).values.any():
            to_drop = []
            for col in upper.columns:
                if any(upper[col] >= threshold):
                    to_drop.append(col)
                    break
                    
            dataf = dataf.drop(to_drop, axis=1)

            print("columns removed: ", to_drop)
        else:
            break
    return dataf


def select_features_rfecv(X, y, classifier, cv=5, scoring='f1_weighted'):
    """
    Select features using Recursive Feature Elimination with Cross-Validation (RFECV).

    Parameters:
        X : array-like or DataFrame
            The feature matrix.
        y : array-like
            The target variable.
        cv : int, cross-validation generator or an iterable, optional (default=5)
            Determines the cross-validation splitting strategy.
        scoring : str or callable, optional (default='f1_weighted')
            A scoring method to evaluate the performance of the estimator.

    Returns:
        DataFrame:
            DataFrame containing the selected features.
    """
    X_cop = X.copy()
    y_cop = y.copy()

    lab_enc = LabelEncoder()
    y_cop = lab_enc.fit_transform(y_cop)
    target_label_mapping = {
            label: category for label, category in enumerate(lab_enc.classes_)
        }
    print(target_label_mapping)

    rfecv_selector = RFECV(estimator=classifier, cv=cv, scoring=scoring)
    rfecv_selector.fit(X_cop, y_cop)

    selected_features = X_cop.columns[rfecv_selector.support_]

    return X_cop[selected_features]

def custom_error_cost_score(y_true, y_pred):
        error_cost_matrix = np.array([[0, 1, 2],
                                      [1, 0, 1],
                                      [2, 1, 0]])
        cm = confusion_matrix(y_true, y_pred)
        error_value = np.sum(cm * error_cost_matrix) / len(y_true)
        
        return error_value
matrix_error_function = make_scorer(custom_error_cost_score, greater_is_better=False)

def select_k_best(X_train, y_train, score_func, num_of_features):
    X_train_cop = X_train.copy()
    y_train_cop = y_train.copy()
    
    selector = SelectKBest(score_func=score_func, k=num_of_features) #f_classif, mutual_info_classif
    selector.fit(X_train_cop, y_train_cop)
    
    selected_feature_indices = selector.get_support(indices=True) #get feature indices
    selected_features = X_train_cop.columns[selected_feature_indices] #get feature names using indices
    feature_scores = selector.scores_[selected_feature_indices] #select scores using indices
    feature_importances = dict(zip(selected_features, feature_scores)) #zip the socres and corresponding feature names
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)) #sort the feature importances
    X_train_selected = selector.transform(X_train_cop) #transform x_train

    return X_train_selected, sorted_importances