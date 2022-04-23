from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
#from google.colab import files
#import io
import pickle


def Feature_Encoder(cols, filename):
    lbl = LabelEncoder()
    lbl.fit((cols.values))
    cols = lbl.transform((cols.values))
    # cols = LabelBinarizer().fit_transform(cols.values)
    # filename = 'finalized_model.sav'
    pickle.dump(lbl, open(filename, 'wb'))
    return cols


def Feature_encode_test(cols, filename):
    lbl = pickle.load(open(filename, 'rb'))
    cols = lbl.transform((cols.values))
    return cols

def normalizer(data, cols):
    for f in cols:
        normalize = StandardScaler()
        normalize.fit(np.array(data[f]).reshape(-1,1))
        data[f] = normalize.transform(np.array(data[f]).reshape(-1,1))
        pickle.dump(normalize, open((f+'n'+'.sav'), 'wb'))
    return data
def normalizer_test(cols, filename):
    for f in filename:
        normalize = pickle.load(open((f+'n'+'.sav'), 'rb'))
        cols[f] = normalize.transform(np.array(cols[f]).reshape(-1,1))
    return cols
def pre_train(data):

    data['EmploymentStatusDuration'] = data.loc[:, 'EmploymentStatusDuration'].fillna(value=0, inplace=False)
    data['DebtToIncomeRatio'] = data.loc[:, 'DebtToIncomeRatio'].fillna(value=value,inplace=False)
    data = data.dropna(axis='columns', how='any', subset=None, inplace=False)#TotalProsperPaymentsBilled CreditGrade
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
    data = normalizer(data, data.columns)
    return data


def pre_test(data):

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


data = pd.read_csv('LoanRiskScore.csv')

data = data.dropna(subset=['LoanRiskScore'])
X = data.iloc[:, 0:-1]
Y = data['LoanRiskScore']
#data=pre_train(data)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=150)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25,shuffle=True,random_state=150)
# transforms the existing features to higher degree features.
value=X_train["DebtToIncomeRatio"].mean()
pickle.dump(value,open('value.sav', 'wb'))
pickle.dump(X_test,open('X_test.sav', 'wb'))
pickle.dump(y_test,open('y_test.sav', 'wb'))

#print(X_train["LoanStatus"].value_counts())
X_train = pre_train(X_train)
X_test = pre_test(X_test)

#x = X_train.isna().sum()
corrl=X_train.corr()

sns.heatmap(corrl, annot=True)
#plt.show()

poly_features = PolynomialFeatures(degree=3)

X_train_poly = poly_features.fit(X_train)
X_train_poly = poly_features.transform(X_train)

X_test_poly = poly_features.transform(X_test)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

prediction = poly_model.predict((X_test_poly))

sc=poly_model.score(X_test_poly,y_test)

score=r2_score(y_test,prediction)

print('  score is',sc)
print('r2 score is',score)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))


############################################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25,shuffle=True,random_state=732)

X_train = pre_train(X_train)
X_test = pre_test(X_test)

SV_model = svm.SVR()
SV_model.fit(X_train, y_train)

svr_accuracy = SV_model.score(X_test, y_test)

svr_prediction1 = SV_model.predict(X_train)
svr_error1 = metrics.mean_squared_error(y_train, svr_prediction1)

svr_prediction2 = SV_model.predict(X_test)
svr_error2 = metrics.mean_squared_error(y_test, svr_prediction2)
#pickle.dump(SV_model, open('SV_model.sav', 'wb'))

print('Mean Square Error train', svr_error1)
print('Mean Square Error test', svr_error2)
print('SVR Accuracy: ', svr_accuracy * 100)
