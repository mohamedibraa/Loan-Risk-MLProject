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
import pickle
from time import process_time

p=pickle

def Feature_encode_test(cols, filename):
    lbl = p.load(open(filename, 'rb'))
    cols = lbl.transform((cols.values))
    return cols

def normalizer_test(cols, filename):
    for f in filename:
        normalize = p.load(open((f+'n'+'.sav'), 'rb'))
        cols[f] = normalize.transform(np.array(cols[f]).reshape(-1,1))
    return cols

#TotalProsperPaymentsBilledn
def pre_test(data):

    data = data.drop(columns=['StatedMonthlyIncome', 'AvailableBankcardCredit', 'RevolvingCreditBalance','TotalTrades','CreditGrade','DebtToIncomeRatio','TotalProsperPaymentsBilled'])

    data['EmploymentStatusDuration'] = data.loc[:, 'EmploymentStatusDuration'].fillna(value=0, inplace=False)
    #data['DebtToIncomeRatio'] = data.loc[:, 'DebtToIncomeRatio'].fillna(value=data['DebtToIncomeRatio'].mean(),inplace=False)
    #data = data.dropna(axis='columns', how='any', subset=None, inplace=False)

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
    data['LoanStatus'] = Feature_encode_test(data.loc[:, 'LoanStatus'], LoanStatus_filename)
    data['BorrowerState'] = Feature_encode_test(data.loc[:, 'BorrowerState'], BorrowerState_filename)
    data['EmploymentStatus'] = Feature_encode_test(data.loc[:, 'EmploymentStatus'], EmploymentStatus_filename)

    data = normalizer_test(data,data.columns)
    return data

file="LoanRiskTestForTAsClassification_new_test.csv"
X_test=[]
y_test=[]
try:
    data = pd.read_csv(file)
    #data = data.dropna(subset=['ProsperRating (Alpha)'])
    X_test=data.iloc[:,:-1]
    y_test=data.iloc[:,-1]
    #y_test=y_test.fillna(np.median(y_test))
except:
    print("Exception")

#nan=y_test.isna().sum()
lbl = p.load(open('y_train_trans', 'rb'))
y_test=pd.Series(lbl.transform(y_test))    ## fit is not true use pikle   y_train_trans
y_test=y_test.fillna(np.median(y_test))    ###    record for ebra

X_test = pre_test(X_test)
models_accuracy=[]
models_names=["tree","decision tree","rbf_svm"]
###################################    tree     ##########################################################################
decision_tree = p.load(open(('decision_tree.sav'), 'rb'))
tic= process_time()
models_accuracy.append(r2_score(y_test,(decision_tree.predict(X_test))))

print(f'decision tree Accuracy for test data: {models_accuracy[-1]}')
toc= process_time()
print("tree test time: ",toc-tic,"\n")


##############################    decision tree      ###########################################################################
random_forest = p.load(open(('random_forest.sav'), 'rb'))
tic= process_time()
models_accuracy.append(r2_score(y_test,(random_forest.predict(X_test))))
print("Random forest test accuracy: ", models_accuracy[-1])
toc= process_time()
print("decision tree test time: ",toc-tic,"\n")

########################################## svm ##################################################

normal_svc=p.load(open(('normal_svc.sav'), 'rb'))
tic= process_time()
models_accuracy.append(r2_score(y_test,normal_svc.predict(X_test)))
print('normal svc test',models_accuracy[-1])
toc= process_time()
print("normal svm test time: ",toc-tic,"\n")

plt.plot(models_names,models_accuracy)
plt.show()

svc=p.load(open(('svc.sav'), 'rb'))
tic= process_time()
print('linear svc test',r2_score(y_test,svc.predict(X_test)))
toc= process_time()
print("linear svc test time: ",toc-tic,"\n")



lin_svc=p.load(open(('lin_svc.sav'), 'rb'))
tic= process_time()
print('lin_svc test',r2_score(y_test,lin_svc.predict(X_test)))
toc= process_time()
print("lin_svc test time: ",toc-tic,"\n")



rbf_svc=p.load(open(('rbf_svc.sav'), 'rb'))
tic= process_time()
print('rbf_svc test',r2_score(y_test,rbf_svc.predict(X_test)))
toc= process_time()
print("rbf_svc test time: ",toc-tic,"\n")



sig_svc=p.load(open(('sig_svc.sav'), 'rb'))
tic= process_time()
print('sig_svc test',r2_score(y_test,sig_svc.predict(X_test)))
toc= process_time()
print("sig_svc test time: ",toc-tic,"\n")



poly_svc=p.load(open(('poly_svc.sav'), 'rb'))
tic= process_time()
print('poly_svc test',r2_score(y_test,poly_svc.predict(X_test)))
toc= process_time()
print("poly_svc test time: ",toc-tic,"\n")

