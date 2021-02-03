from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV


def Clas_LogisticRegression(X_train, y_train):
   
    LR = LogisticRegression()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    LR_fit_time = scores['fit_time'].mean()
    LR_score_time = scores['score_time'].mean()
    LR_accuracy = scores['test_accuracy'].mean()
    LR_precision = scores['test_precision_macro'].mean()
    LR_recall = scores['test_recall_macro'].mean()
    LR_f1 = scores['test_f1_weighted'].mean()
    LR_roc = scores['test_roc_auc'].mean()
    
    return LR_fit_time, LR_score_time, LR_accuracy, LR_precision, LR_recall, LR_f1, LR_roc


def Clas_DecisionTree(X_train, y_train):
   
    decision_tree = DecisionTreeClassifier()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    dtree_fit_time = scores['fit_time'].mean()
    dtree_score_time = scores['score_time'].mean()
    dtree_accuracy = scores['test_accuracy'].mean()
    dtree_precision = scores['test_precision_macro'].mean()
    dtree_recall = scores['test_recall_macro'].mean()
    dtree_f1 = scores['test_f1_weighted'].mean()
    dtree_roc = scores['test_roc_auc'].mean()
    
    return dtree_fit_time, dtree_score_time, dtree_accuracy, dtree_precision, dtree_recall, dtree_f1, dtree_roc



def Clas_SVC(X_train, y_train):
   
    SVM = SVC(probability = True)
    scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    SVM_fit_time = scores['fit_time'].mean()
    SVM_score_time = scores['score_time'].mean()
    SVM_accuracy = scores['test_accuracy'].mean()
    SVM_precision = scores['test_precision_macro'].mean()
    SVM_recall = scores['test_recall_macro'].mean()
    SVM_f1 = scores['test_f1_weighted'].mean()
    SVM_roc = scores['test_roc_auc'].mean()
    
    return SVM_fit_time, SVM_score_time, SVM_accuracy, SVM_precision, SVM_recall, SVM_f1, SVM_roc



def Clas_LDA(X_train, y_train):
   
    LDA = LinearDiscriminantAnalysis()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    LDA_fit_time = scores['fit_time'].mean()
    LDA_score_time = scores['score_time'].mean()
    LDA_accuracy = scores['test_accuracy'].mean()
    LDA_precision = scores['test_precision_macro'].mean()
    LDA_recall = scores['test_recall_macro'].mean()
    LDA_f1 = scores['test_f1_weighted'].mean()
    LDA_roc = scores['test_roc_auc'].mean()
    
    return LDA_fit_time, LDA_score_time, LDA_accuracy, LDA_precision, LDA_recall, LDA_f1, LDA_roc



def Clas_QDA(X_train, y_train):
   
    QDA = QuadraticDiscriminantAnalysis()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    QDA_fit_time = scores['fit_time'].mean()
    QDA_score_time = scores['score_time'].mean()
    QDA_accuracy = scores['test_accuracy'].mean()
    QDA_precision = scores['test_precision_macro'].mean()
    QDA_recall = scores['test_recall_macro'].mean()
    QDA_f1 = scores['test_f1_weighted'].mean()
    QDA_roc = scores['test_roc_auc'].mean()
    
    return QDA_fit_time, QDA_score_time, QDA_accuracy, QDA_precision, QDA_recall, QDA_f1, QDA_roc



def Clas_RFOREST(X_train, y_train):
   
    random_forest = RandomForestClassifier()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    forest_fit_time = scores['fit_time'].mean()
    forest_score_time = scores['score_time'].mean()
    forest_accuracy = scores['test_accuracy'].mean()
    forest_precision = scores['test_precision_macro'].mean()
    forest_recall = scores['test_recall_macro'].mean()
    forest_f1 = scores['test_f1_weighted'].mean()
    forest_roc = scores['test_roc_auc'].mean()
    
    return forest_fit_time, forest_score_time, forest_accuracy, forest_precision, forest_recall, forest_f1, forest_roc


def Clas_KNN(X_train, y_train):
   
    KNN = KNeighborsClassifier()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    KNN_fit_time = scores['fit_time'].mean()
    KNN_score_time = scores['score_time'].mean()
    KNN_accuracy = scores['test_accuracy'].mean()
    KNN_precision = scores['test_precision_macro'].mean()
    KNN_recall = scores['test_recall_macro'].mean()
    KNN_f1 = scores['test_f1_weighted'].mean()
    KNN_roc = scores['test_roc_auc'].mean()
    return KNN_fit_time, KNN_score_time, KNN_accuracy, KNN_precision, KNN_recall, KNN_f1, KNN_roc


def Clas_GNB(X_train, y_train):
   
    bayes = GaussianNB()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    bayes_fit_time = scores['fit_time'].mean()
    bayes_score_time = scores['score_time'].mean()
    bayes_accuracy = scores['test_accuracy'].mean()
    bayes_precision = scores['test_precision_macro'].mean()
    bayes_recall = scores['test_recall_macro'].mean()
    bayes_f1 = scores['test_f1_weighted'].mean()
    bayes_roc = scores['test_roc_auc'].mean()
    return bayes_fit_time, bayes_score_time, bayes_accuracy, bayes_precision, bayes_recall, bayes_f1, bayes_roc

