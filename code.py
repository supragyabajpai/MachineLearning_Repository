#ML_Hackathon
'''
Contents:
1)Import Necessary Libraries
2)Read In and Explore the Historic Data
3)Data Analysis
4)Data Visualization
5)Cleaning Data
6)Choosing the Best Model
7)Creating Submission File

#-------------------------Answer to the 4th question----------------------------------
Recall should be on the higher side in this particular case beacuase ( or I need to minimize
the number of False Negative)
As Recall = True Positive / (true positive + false neagtive). Because as the false neagtive cases will suggest
the existence of potentially wrong customers in the company who tend to default on the loans poses a big threat
to copany's profit.

In our telecom data, test sensitivity is the ability of a test to correctly identify those who will pay back
the loan in specified time, whereas test specificity is the ability of the test to correctly those who will default
on the payment in specified time. So, in our opinion test specificity should be the target. As we need to minimize the
number of defaulters who are being predicted to be able to pay back the loan. i.e. we need to minimize
the number of cases where the model wrongly predicts a defaulter as the non-defaulter.

Same as we earlier said, That Recall should be targeted or the False Negatives should be reduced is the primary focus.

#-------------------------------------------Answer Completed -----------------------------------------------------------------------------



#-----------------------------------Steps Involved: Explained in words...----------------------------------------------------------------

The flow of the program was as follows:
1) We dropped the variables of 'object type' at first to create a correlation matrix plot to see the relationship among variables
2) Removed one of the variables having high collinearity ( or having high yellow color intensity in the plot )
3) We gain looked at the correlation plot to remove few more variables and restricted ourselves to 25 variables
4) In the data pre-processing, we converted alphanumeric values to Null Values and later imputed them by mean technique
5) We did Negative Value treatment and Outlier Treatment for the data
6) After the data was ready (cleaned, imputed non-textual data), we checked the feature selection method 'Recursive Feature Elimination' and calculated the accuracies
7) We switched to PCA as the recursive feature elimination was not giving satisfying results
8) Then we applied Principal Component Analysis to extract 10 new features which are a linear combination of old features
9) We did exactly the same operations on test data and applied the spot check algorithms after splitting the train data
10) We collected the results of all spot check algorithms and found out that Random Forest is giving the highest accuracy
11)Changing the parameters of RandomForestClassifier to target the maximum Recall value, we freezed our results.
12)Printed Classification report to see the precision and recall values





'''

#data analysis libraries
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 30)
pd.set_option('precision', 2)
import matplotlib.pyplot as plt
import seaborn as sbn
import warnings
warnings.filterwarnings('ignore')

#import train CSV files
train = pd.read_csv('train.csv', index_col=0)
# you need to enter scoring dataset as (test.csv) and it must contain the column 'label'
test = pd.read_csv('test.csv', index_col=0)
##filename = pd.read_csv('test.csv')

#take a look at the training data .....

print( train.describe()[:]  )
print( "\n"  )
print( train.describe(include="all")  )
print(  "\n"  )
print( pd.isnull(train).sum()  )
print (train.dtypes)

#--------------Dropping off the 'object type variables to see the initial correlations by correlation matrix-----------
train1 = train.drop(['label'], axis = 1)
train1 = train1.drop(['msisdn'], axis = 1)
train1 = train1.drop(['aon'], axis = 1)
train1 = train1.drop(['daily_decr30'], axis = 1)
train1 = train1.drop(['daily_decr90'], axis = 1)
train1 = train1.drop(['rental30'], axis = 1)
train1 = train1.drop(['rental90'], axis = 1)
train1 = train1.drop(['pcircle'], axis = 1)
train1 = train1.drop(['pdate'], axis = 1)
print (train1.head())


test1 = test.drop(['label'], axis = 1)
test1 = test1.drop(['msisdn'], axis = 1)
test1 = test1.drop(['aon'], axis = 1)
test1 = test1.drop(['daily_decr30'], axis = 1)
test1 = test1.drop(['daily_decr90'], axis = 1)
test1 = test1.drop(['rental30'], axis = 1)
test1 = test1.drop(['rental90'], axis = 1)
test1 = test1.drop(['pcircle'], axis = 1)
test1 = test1.drop(['pdate'], axis = 1)

#----------------plot correlation matrix to see the relationship among variables --------------------

correlations = train1.corr()
print( correlations  )
print (train1.dtypes)

fig = plt.figure()
#Following will add matrix and side bar in entire area
subFig = fig.add_subplot(111)

cax = subFig.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)

#------------------------------------------------------------
ticks = np.arange(0,27)     # It will generate values from 0....27
subFig.set_xticks(ticks)
subFig.set_yticks(ticks)
subFig.set_xticklabels(train1.columns)
subFig.set_yticklabels(train1.columns)
#plt.show()
#------------------------------------ Plotting Done ---------------------------


#----------------------------Converting alpha-numeric values to Null-----------

train2 = train.drop(['pdate'], axis = 1)
test2 = test.drop(['pdate'], axis = 1)


cols = train2.columns
train2[cols] = train2[cols].apply(pd.to_numeric, errors='coerce')

cols1 = test2.columns
test2[cols1] = test2[cols1].apply(pd.to_numeric, errors='coerce')


print(train2.dtypes)
train2 = train2.drop(['pcircle'], axis = 1)
train2 = train2.drop(['msisdn'], axis = 1)

test2 = test2.drop(['pcircle'], axis = 1)
test2 = test2.drop(['msisdn'], axis = 1)



print( pd.isnull(train2).sum())

print (train2.columns )

#-----------------------All the alpha numeric values converted to Null Values -------------------------------------------


#----------------------------------------Imputation for Null values---------------------------------------------------

train3 = train2.fillna(train2.mean())

test3 = test2.fillna(train2.mean())

print( pd.isnull(train3).sum())

#-------------------------------------Imputed new values-----------------------------------

#=-----------------------------Converting to absolute values -----------------------------------

train3 = abs(train3)
print (train3.head())
c =list( train3.columns)
train4 = pd.DataFrame({})
for i in c :
    train4[i] = train3[i].abs()
print (train4)




test3 = abs(test3)
print (test3.head())
c =list( test3.columns)
test4 = pd.DataFrame({})
for i in c :
    test4[i] = test3[i].abs()
print (test4)
#print (sum(n<0 for n in train4.values.flatten()))
print ('No negative values in the Data')

#-----------------------------ALL the negative values converted into positive-------------------------


#----------------------- ----------Outlier Treatment for Train------------------------------------
train4.describe(include='all')
c =list( train4.columns)


from scipy import stats
print (train4['aon'].mean())
#train4['aon'].replace(train4.aon> (3*train4['aon'].mean()),train4['aon'].mean(),inplace=True)
print (train4['aon'].mean())
print (train4.describe(include='all'))

for i in c:
    train4[i] = np.where(train4[i] >(3*train4[i].mean()),train4[i].mean(),train4[i])
print (train4.describe(include='all'))

#--------------------------------------Outliers Removed -------------------------------------------

#-----------------------------------Outliers Treatment for test--------------

c =list( test4.columns)

from scipy import stats
print (test4['aon'].mean())
#train4['aon'].replace(train4.aon> (3*train4['aon'].mean()),train4['aon'].mean(),inplace=True)
print (test4['aon'].mean())
print (test4.describe(include='all'))

for i in c:
    test4[i] = np.where(test4[i] >(3*test4[i].mean()),test4[i].mean(),test4[i])
print (test4.describe(include='all'))


#------------------------------Outliers test data treatment done --------------------

#------------------Dropping Correlated Variables in train data  -------------------------

train4 = train4.drop(['cnt_ma_rech90'], axis = 1)
train4 = train4.drop(['sumamnt_ma_rech90'], axis = 1)
train4 = train4.drop(['medianamnt_ma_rech90'], axis = 1)
train4 = train4.drop(['amnt_loans90'], axis = 1)
train4 = train4.drop(['medianamnt_loans90'], axis = 1)

print (train4.describe(include='all'))

correlations = train4.corr()
print( correlations  )

fig = plt.figure()
#Following will add matrix and side bar in entire area
subFig = fig.add_subplot(111)

cax = subFig.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)

ticks = np.arange(0,27)   # It will generate values from 0....8
subFig.set_xticks(ticks)
subFig.set_yticks(ticks)
subFig.set_xticklabels(train4.columns)
subFig.set_yticklabels(train4.columns)
#plt.show()

train4 = train4.drop(['daily_decr90'], axis = 1)
train4 = train4.drop(['rental90'], axis = 1)
train4 = train4.drop(['cnt_loans30'], axis = 1)

print (train4.describe(include='all'))



#-------------------Dropped Correlated Variables in train data ---------------------

#-------------------Dropping correlated variables in Test data --------------------

test4 = test4.drop(['cnt_ma_rech90'], axis = 1)
test4 = test4.drop(['sumamnt_ma_rech90'], axis = 1)
test4 = test4.drop(['medianamnt_ma_rech90'], axis = 1)
test4 = test4.drop(['amnt_loans90'], axis = 1)
test4 = test4.drop(['medianamnt_loans90'], axis = 1)

test4 = test4.drop(['daily_decr90'], axis = 1)
test4 = test4.drop(['rental90'], axis = 1)
test4 = test4.drop(['cnt_loans30'], axis = 1)

#---------------------Dropped correlated variables in test data ------------------------------

'''
#--------------------------Recursive Feature Elimination---------------------------------

array = train4.values
X = array[:,1:]
Y = array[:,0]
print (X)
print (Y)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

rfe = RFE(model, 10)
fit = rfe.fit(X, Y)
result = fit.transform(X)

print( "Num Features:      ",  fit.n_features_  )
print( "Selected Features: ",  fit.support_  )
print( "Feature Ranking:   ",  fit.ranking_  )

print (train4.describe(include='all'))
print (type(fit.ranking_))
print ((fit.ranking_).ndim)
d=list(train4.columns)
train5=pd.DataFrame({})
train6=train4

print(d)
print(len(fit.ranking_))

for i in range(len(fit.ranking_)):
    if fit.ranking_[i]==1:
        train5[(train6.columns)[i]]=train4.values[i]

#train5[0]=train4['label']
print(train5)


#---------------------------Recursive Fearture Elimination---------------------
'''

#---------------------Principal Component Analysis on train data ---------------------------------

array =train4.values
X = array[:,1:]
Y = array[:,0]

# feature extraction
from sklearn.decomposition import PCA
pca = PCA(n_components=10)

fit = pca.fit(X)

resultX = pca.transform(X)
resultdf = pd.DataFrame(data=resultX,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
#finaldf = pd.concat([resultdf,train4[['label']]],axis=1)

resultdf['label'] = train4 ['label'].values
print (train4)
print(resultdf)
train5=resultdf
print (test4)
#------------------------Principal component applied on train data---------------------------------

#------------------------Principal compaonent in test data ---------------------

array1 =test4.values
X1 = array1[:,1:]
Y1 = array1[:,0]

# feature extraction
from sklearn.decomposition import PCA
pca = PCA(n_components=10)

fit = pca.fit(X1)

resultX1 = pca.transform(X1)
resultdf1 = pd.DataFrame(data=resultX1,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
#finaldf = pd.concat([resultdf,train4[['label']]],axis=1)

resultdf1['label'] = test4 ['label'].values
print (test4)
print(resultdf1)
test5=resultdf1

#------------------------Principal Component analysis done in test data ---------------




#----------------------------Splitting the test_train-----------------------------------

from sklearn.model_selection import train_test_split


input_predictors = train5.drop(['label'], axis=1)
ouptut_target = train5['label']

x_train, x_val, y_train, y_val=train_test_split(
    input_predictors, ouptut_target, test_size = 0.01, random_state = 7)



#------------------------------------------------------------------------------------------
'''
#------------------------------------Spiltting Done ------------------------------------------------------------

#---------------------------------Spot Check Algorithms ----------------------------------------------------


#****************************************
#Choosing the Best Model
#****************************************

#Testing Different Models
#I will be testing the following models with my training data (got the list from here):

#1) Logistic Regression
#2) Gaussian Naive Bayes
#3) Support Vector Machines
#4) Linear SVC
#5) Perceptron
#6) Decision Tree Classifier
#7) Random Forest Classifier
#8) KNN or k-Nearest Neighbors
#9) Stochastic Gradient Descent
#10) Gradient Boosting Classifier

from sklearn.metrics import accuracy_score

#MODEL-1) LogisticRegression
#------------------------------------------
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-1: Accuracy of LogisticRegression : ", acc_logreg  )


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


model = LogisticRegression()
num_folds = 10
#seed = 7
kfold = KFold(n_splits=num_folds)
results = cross_val_score(model, x_train, y_train, cv = kfold )

print( "results : " , results  )
print(  )
print(  "Accuracy: %.3f%% "  % ( results.mean()*100.0 )  )


#MODEL-2) Gaussian Naive Bayes
#------------------------------------------
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-2: Accuracy of GaussianNB : ", acc_gaussian  )



#OUTPUT:-
#MODEL-2: Accuracy of GaussianNB :


#MODEL-4) Linear SVC
#------------------------------------------
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-4: Accuracy of LinearSVC : ",acc_linear_svc  )


#OUTPUT:-
#MODEL-4: Accuracy of LinearSVC :  





#MODEL-5) Perceptron
#------------------------------------------
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-5: Accuracy of Perceptron : ",acc_perceptron  )


#OUTPUT:-
#MODEL-5: Accuracy of Perceptron :


#MODEL-6) Decision Tree Classifier
#------------------------------------------
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-6: Accuracy of DecisionTreeClassifier : ", acc_decisiontree  )



#OUTPUT:-
#MODEL-6: Accuracy of DecisionTreeClassifier :





#MODEL-7) Random Forest
#------------------------------------------
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(max_depth = 10, max_features=10, criterion= 'entropy', min_samples_leaf=3,random_state=10)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-7: Accuracy of RandomForestClassifier : ",acc_randomforest  )


#OUTPUT:-
#MODEL-7: Accuracy of RandomForestClassifier :





#MODEL-8) KNN or k-Nearest Neighbors
#------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-8: Accuracy of k-Nearest Neighbors : ",acc_knn  )



#OUTPUT:-
#MODEL-8: Accuracy of k-Nearest Neighbors :







#MODEL-9) Stochastic Gradient Descent
#------------------------------------------
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-9: Accuracy of Stochastic Gradient Descent : ",acc_sgd )



#MODEL-10) Gradient Boosting Classifier
#------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk )


#OUTPUT:-
#MODEL-10: Accuracy of Stochastic Gradient Descent :
'''

#-------------------------------Spot check Algoriths applied ---------------------------------------------------------

#I decided to use the Random Forest model for the testing because of its highest accuracy of 89.15.



#-------------------------------Classification Report ----------------------------#


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
model = RandomForestClassifier()
model.fit(x_train, y_train)
predicted = model.predict(test5.drop('label',axis=1))
report = classification_report(test5['label'], predicted)
print(report)


#------------------------------------Random Forest on test data ----------------------------------------------



randomforest = RandomForestClassifier(max_depth = 10, max_features=10, criterion= 'entropy', min_samples_leaf=5,random_state=10)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(test5.drop('label',axis=1))


acc_randomforest = round(accuracy_score(y_pred,test5['label'] ) * 100, 2)
print( "MODEL-7: Accuracy of RandomForestClassifier by predictions on Scoring Dataset : ",acc_randomforest  )

#By changing the values of parameters, we were able to achieve higher precision and recall


#--------------------------------Gradient boosing Classifier------------------------------------------
#--------------------------Gradient boosting applied and checked the resulte-----------------
'''
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(test5.drop('label',axis=1))
acc_gbk = round(accuracy_score(y_pred,test5['label']) * 100, 2)
print( "MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk )
'''
# Accuracy of gradient boosting was 87.2 percent only


#---------------------------------Submitting to output file  -----------------------------------------
#set ids as Phone numbers and predict probability of  repayment

ids = test['msisdn']
#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'msisdn' : ids, 'label': y_pred })
output.to_csv('submission.csv', index=False)

print( "All survival predictions done." )
print( "All predictions exported to submission.csv file." )

#-----------------------------------------End of the project --------------------------------------------------------------------



