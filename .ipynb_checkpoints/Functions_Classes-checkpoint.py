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

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

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
    
    def custom_error_cost_score(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        error_value = np.sum(cm * self.cost_matrix) / len(y_true)
        
        return -error_value
    
    def calculate_cost_matrix_error_cv(self, n_folds=5):
        cost_matrix_socrer = make_scorer(self.custom_error_cost_score, greater_is_better=True)
        
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
                scoring=cost_matrix_socrer, 
                n_jobs=-1
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
            scoring=scoring
        )
        grid_search.fit(self.x_train, self.encoded_y_train)
        
        self.grid_object = grid_search
        self.grid_best_parameters = self.grid_object.best_params_
        self.grid_best_model = self.grid_object.best_estimator_
        self.grid_cv_results = self.grid_object.cv_results_
        self.grid_best_score = self.grid_object.best_score_
        
        print(f"Best parameters found by GridSearchCV ({scoring}):\n{self.grid_best_parameters}")
        print(f"\nBest score found by GridSearchCV ({scoring}):\n{self.grid_best_score}")
        
    def predict_on_test(self, estimator):
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


def apply_one_hot_encoding(data, categorical_column):
    """
    Apply one-hot encoding to a categorical column in the DataFrame.

    Parameters:
        data (DataFrame): Input DataFrame containing features.
        categorical_column (str): Name of the categorical column to be one-hot encoded.

    Returns:
        ohe_encoder (OneHotEncoder): One-hot encoder object fitted on categorical data.
        X_encoded (DataFrame): DataFrame with categorical column replaced by one-hot encoded columns.
    """
    ohe_encoder = OneHotEncoder()

    X_encoded = pd.DataFrame(
        ohe_encoder.fit_transform(data[[categorical_column]]).toarray(),
        columns=ohe_encoder.get_feature_names_out([categorical_column])
    )

    X_encoded = pd.concat([data.drop(columns=[categorical_column]), X_encoded], axis=1)

    return X_encoded, ohe_encoder


def apply_smote(data, random_state=0):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

    Parameters:
        data (DataFrame): Input DataFrame containing features and target variable.
        categorical_column (str): Name of the categorical column to be one-hot encoded (optional).
        random_state (int): Random state for reproducibility.

    Returns:
        smote_df (DataFrame): Resampled DataFrame with balanced classes.
    """
    X = data.drop(columns=['Class'])
    y = data['Class']

    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X, y)

    smote_df = pd.concat([pd.DataFrame(X_smote, columns=X.columns), pd.Series(y_smote, name='Class')], axis=1)

    return smote_df


def apply_std_scaler(data, columns):
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
    scaled_data = data.copy()
    scaled_data[columns] = std_scaler.fit_transform(data[columns])
    
    return scaled_data, std_scaler

