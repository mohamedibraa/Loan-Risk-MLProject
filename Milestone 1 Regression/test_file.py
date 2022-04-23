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

p=pickle


def Feature_encode_test(cols, filename):
    lbl = pickle.load(open(filename, 'rb'))
    cols = lbl.transform((cols.values))
    return cols

def normalizer_test(cols, filename):
    for f in filename:
        normalize = p.load(open((f+'n'+'.sav'), 'rb'))
        cols[f] = normalize.transform(np.array(cols[f]).reshape(-1,1))
    return cols

def pre_test(data):
    value = p.load(open('value.sav', 'rb'))
    data['EmploymentStatusDuration'] = data.loc[:, 'EmploymentStatusDuration'].fillna(value=0, inplace=False)
    data['DebtToIncomeRatio'] = data.loc[:, 'DebtToIncomeRatio'].fillna(value=value, inplace=False)
    data = data.drop(columns=['CreditGrade','TotalProsperPaymentsBilled'])

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
    data = normalizer_test(data, data.columns)
    return data


file=""
X_test=[]
y_test=[]
try:
    data = pd.read_csv(file)
    #data = data.dropna(subset=['LoanRiskScore'])

    X_test=data.iloc[:,:-1]
    y_test=data.iloc[:,-1]

except:
    X_test=p.load(open('X_test.sav', 'rb'))
    y_test=p.load(open('y_test.sav', 'rb'))
    #print(X_test["LoanStatus"].value_counts())

#nan=y_test.isna().sum()

   ## fit is not true use pikle   y_train_trans
    ###    record for ebra

X_test = pre_test(X_test)
#pickle.dump(poly_features, open('poly_features.sav', 'wb'))
#pickle.dump(poly_model, open('poly_model.sav', 'wb'))

poly_features = p.load(open(('poly_features.sav'), 'rb'))
poly_model = p.load(open(('poly_model.sav'), 'rb'))

X_test_poly=poly_features.transform(X_test)
prediction=poly_model.predict(X_test_poly)
score=r2_score(y_test,prediction)

print('Polynomial Regression Accuracy for test data:  ',score*100)

#################################################################################3
#pickle.dump(SV_model, open('SV_model.sav', 'wb'))


SV_model = p.load(open(('SV_model.sav'), 'rb'))

prediction=SV_model.predict(X_test)
score=r2_score(y_test,prediction)
print("SVM test accuracy: ", score*100)


########################################## svm ##################################################
