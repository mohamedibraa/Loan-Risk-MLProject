from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
#from google.colab import files
#import io
import pickle

p=pickle
def Feature_Encoder(cols, filename):
    lbl = LabelEncoder()
    lbl.fit((cols.values))
    cols = lbl.transform((cols.values))
    # cols = LabelBinarizer().fit_transform(cols.values)
    # filename = 'finalized_model.sav'
    p.dump(lbl, open(filename, 'wb'))
    return cols


def normalizer(data, cols):
    for f in cols:
        normalize = StandardScaler()
        normalize.fit(np.array(data[f]).reshape(-1,1))
        data[f] = normalize.transform(np.array(data[f]).reshape(-1,1))
        p.dump(normalize, open((f+'n'+'.sav'), 'wb'))
    return data



def pre_train(data):
    print(data.info())

    data['EmploymentStatusDuration'] = data.loc[:, 'EmploymentStatusDuration'].fillna(value=0, inplace=False)
    #v=data['DebtToIncomeRatio'].mean()
    #data['DebtToIncomeRatio'] = data.loc[:, 'DebtToIncomeRatio'].fillna(value=data['DebtToIncomeRatio'].mean(),inplace=False)
    data = data.dropna(axis='columns', how='any', subset=None, inplace=False) #col DebtToIncomeRatio,CreditGrade

    print(data.info())
    data.boxplot(figsize=(20, 6), fontsize=20)
    plt.title("Whole data Box Plot")
    plt.show()
#    plt.boxplot(data['DebtToIncomeRatio'])
  #  plt.show()



    data = data.drop(columns=['StatedMonthlyIncome','AvailableBankcardCredit','RevolvingCreditBalance','TotalTrades'])
    print(data.info())
    #data.boxplot(figsize=(20, 6), fontsize=20)
    #plt.show()

    y = []
    count = 0
    for i in data['IncomeRange']:
        if (i == '$1-24,999'):
            y.append(0.20)
        elif (i == '$25,000-49,999'):
            y.append(0.40)
        elif (i == '$50,000-74,999'):
            y.append(0.60)
        elif (i == '$75,000-99,999'):
            y.append(0.80)
        elif (i == '$100,000+'):
            y.append(1.00)
        else:
            y.append(0.0)
        count += 1
    y = np.array(y)
    y = y.reshape(count, 1)
    data.loc[:, 'IncomeRange'] = y

    LoanStatus_filename = "LoanStatus.sav"
    BorrowerState_filename = "BorrowerState.sav"
    EmploymentStatus_filename = "EmploymentStatus.sav"

    data['LoanStatus'] = Feature_Encoder(data.loc[:, 'LoanStatus'], LoanStatus_filename)
    data['BorrowerState'] = Feature_Encoder(data.loc[:, 'BorrowerState'], BorrowerState_filename)
    data['EmploymentStatus'] = Feature_Encoder(data.loc[:, 'EmploymentStatus'], EmploymentStatus_filename)
    print(data.info())
    data=normalizer(data,data.columns)

    print(data.info())
    data.boxplot(figsize=(20, 6), fontsize=20)
    plt.title("Boxplots after drop & normalize")
    plt.show()

    return data




def best_of_tree():
    scores = []
    for i in range(1, 31):
        tree = DecisionTreeClassifier(random_state=1, min_samples_split=10, max_depth=i)
        tree.fit(X_train, y_train)
        score = tree.score(X_validate, y_validate)
        if i > 1 and (score > max(scores)):
            best_depth = i
            #break
        scores.append(score)
    plt.plot(range(1,31),scores)
    plt.title("tree scores with max_depth: "+str(best_depth))
    plt.show()

    scores = []
    for i in range(2, 31):
        tree = DecisionTreeClassifier(random_state=1, min_samples_split=i, max_depth=best_depth)
        tree.fit(X_train, y_train)
        score = tree.score(X_validate, y_validate)
        if i > 2 and (score > max(scores)):
            best_min_split=i
            #break
        scores.append(score)
    plt.plot(range(2, 31), scores)
    plt.title("tree scores with min_samples_split: "+str(best_min_split))
    plt.show()
    return [best_depth, best_min_split]



data = pd.read_csv('LoanRiskClassification.csv')
"""for x in data["ProsperRating (Alpha)"].unique():
    print(x, data.LoanStatus[data["ProsperRating (Alpha)"]==x].value_counts())"""

data = data.dropna(subset=['ProsperRating (Alpha)'])
X = data.iloc[:, 0:-1]
Y = data['ProsperRating (Alpha)']
Y_unique=Y.unique()



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=150)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.20, shuffle=True, random_state=150)

p.dump(X_test, open('X_test', 'wb'))
p.dump(y_test, open('y_test', 'wb'))


X_train = pre_train(X_train)
X_validate=pre_train(X_validate)

#y_train=Feature_Encoder(y_train,'y_train_trans')

lbl = LabelEncoder()
lbl.fit(y_train)
p.dump(lbl, open('y_train_trans', 'wb'))

y_train=pd.Series(lbl.transform(y_train))
y_validate=pd.Series(lbl.transform(y_validate))


#y_train=y_train[X_train['LoanStatus'] < 2]
#X_train = X_train[X_train['LoanStatus'] < 2]

#X_test = pre_test(X_test)
#y_test=y_test[X_test['LoanStatus'] < 2]
#X_test = X_test[X_test['LoanStatus'] < 2]

x = X_train.isna().sum()
corrl=X_train.corr()
sns.heatmap(corrl, annot=True)
plt.title("data columns correlations")
plt.show()





#print(X_train.info())



###############################################################################################
#                                  decision tree
###############################################################################################
from sklearn.tree import DecisionTreeClassifier
from time import process_time


[best_depth, best_min_split]=best_of_tree()
tree = DecisionTreeClassifier(random_state=1,min_samples_split=best_min_split,max_depth=best_depth)

tic= process_time()
tree.fit(X_train, y_train)
toc= process_time()
print("tree training time: ",toc-tic)
p.dump(tree, open('decision_tree.sav', 'wb'))

print(f'Decision Tree has {tree.tree_.node_count} nodes with a maximum depth of {tree.tree_.max_depth}')
print(f'decision tree Accuracy for train data: {r2_score(y_train,(tree.predict(X_train)))}')
print(f'decision tree Accuracy for validate data: {r2_score(y_validate,(tree.predict(X_validate)))}')

from sklearn.tree import plot_tree
plt.figure(figsize=(10,5))
decision_tree_plot = plot_tree(tree, filled=True, rounded=True, fontsize=8,max_depth=3,feature_names=X_train.columns,class_names=Y_unique)
plt.title("three levels of decision tree")
plt.show()

plt.plot(X_train.columns,tree.feature_importances_)
plt.title("feature importance with decision tree model")
plt.show()
############################################################################################################
#                                          random forest
##############################################################################################################
from sklearn.ensemble import RandomForestClassifier
scores=[]
max_n_estimators=120
for i in range(120,150,5):
    rf=RandomForestClassifier(random_state=1,min_samples_split=best_min_split,n_estimators=i, n_jobs=-1)
    rf.fit(X_train,y_train)
    score=rf.score(X_validate,y_validate)
    if i>120 and score>max(scores):
        max_n_estimators=i
    scores.append(score)

plt.plot(range(120,150,5),scores)
plt.title("max_n_estimators: "+str(max_n_estimators))
plt.show()

rf=RandomForestClassifier(random_state=1,min_samples_split=best_min_split,n_estimators=max_n_estimators, n_jobs=-1)
tic= process_time()
rf.fit(X_train, y_train)
toc= process_time()
print("random forest training time: ",toc-tic)
p.dump(rf, open('random_forest.sav', 'wb'))

print("Random forest train accuracy: ", r2_score(y_train,(rf.predict(X_train))))
print("Random forest validation accuracy: ", r2_score(y_validate,(rf.predict(X_validate))))

plt.plot(X_train.columns,rf.feature_importances_)
plt.title("feature importance with random forest model")

plt.show()

#################################################################################################################
#                                                  SVM
############################################################################################################
from sklearn import svm
#C = 0.1  # SVM regularization parameter

s=svm.SVC(random_state=1)
tic= process_time()
s.fit(X_train,y_train)
toc= process_time()

p.dump(s, open('normal_svc.sav', 'wb'))
print("normal svc time: ",toc-tic)
print('normal svc validation',r2_score(y_validate,s.predict(X_validate)))
print('normal svc train',r2_score(y_train,s.predict(X_train)))

scores=[]
C=0
for i in np.arange(0.01,0.7,0.1):
    svc = svm.SVC(kernel='linear', C=i)
    svc.fit(X_train, y_train)
    score=r2_score(y_validate, svc.predict(X_validate))
    if (i>0.01) and (score>max(scores)):
        C=i
    scores.append(score)
plt.plot(np.arange(0.01,0.7,0.1), scores)
plt.title("best regularization: "+str(C))
plt.show()


svc = svm.SVC(kernel='linear', C=C)
svc.fit(X_train, y_train)
print("validation linear kernal svc",r2_score(y_validate,svc.predict(X_validate)))
print("train linear kernal svc",r2_score(y_train,svc.predict(X_train)))
p.dump(svc, open('svc.sav', 'wb'))


lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(X_train, y_train)
print("validation lin_svc",r2_score(y_validate,lin_svc.predict(X_validate)))
print("train lin_svc",r2_score(y_train,lin_svc.predict(X_train)))
p.dump(lin_svc, open('lin_svc.sav', 'wb'))


rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C)
rbf_svc.fit(X_train, y_train)
print("validation rbf_svc",r2_score(y_validate,rbf_svc.predict(X_validate)))
print("train rbf_svc",r2_score(y_train,rbf_svc.predict(X_train)))
p.dump(rbf_svc, open('rbf_svc.sav', 'wb'))



sig_svc = svm.SVC(kernel='sigmoid', degree=3, C=C)
sig_svc.fit(X_train, y_train)
print("validation sig_svc",r2_score(y_validate,sig_svc.predict(X_validate)))
print("train sig_svc",r2_score(y_train,sig_svc.predict(X_train)))
p.dump(sig_svc, open('sig_svc.sav', 'wb'))



poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
poly_svc.fit(X_train, y_train)
print("validation poly_svc",r2_score(y_validate,poly_svc.predict(X_validate)))
print("train poly_svc",r2_score(y_train,poly_svc.predict(X_train)))
p.dump(poly_svc, open('poly_svc.sav', 'wb'))






