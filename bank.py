import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
data = pd.read_csv(r"C:\Users\pcpc\Downloads\BankChurners.csv")
data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',"CLIENTNUM"], axis=1, inplace=True)
allcols= data.columns
numerical_cols = data._get_numeric_data().columns
categorical = list(set(allcols) - set(numerical_cols))
data.Attrition_Flag.replace({'Existing Customer': 1, 'Attrited Customer': 0}, inplace=True)
data.Gender.replace({'F': 1, 'M': 0}, inplace=True)
data.Education_Level.replace({'Graduate': 0, 'High School': 1, 'Unknown': 2,'Uneducated':3,'College':4,'Post-Graduate':5,'Doctorate':6}, inplace=True)
data.Marital_Status.replace({'Married': 0, 'Single': 1,'Unknown':3,'Divorced':2}, inplace=True)
data.Income_Category.replace({'Less than $40K': 0,'$80K - $120K': 2, '$60K - $80K': 1, 'Unknown': 3, '$40K - $60K': 4, '$120K +': 5}, inplace=True)
data.Card_Category.replace({'Blue': 0, 'Silver': 1,'Gold':2,'Platinum':3}, inplace=True)
x=data.drop(['Attrition_Flag'],axis='columns')
y=data['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
""" comment:/ this is for finding the best model:
models = [("SGD",SGDClassifier()),\
            ('KNN',KNeighborsClassifier()),\
             ('Random Forest',RandomForestClassifier()),\
             ('Logistic Regression',LogisticRegression()),\
             ('Decision Tree',tree.DecisionTreeClassifier())]
for name, model in models:
    print(name)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test,predictions))"""
model = RandomForestClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print("Classification report:")
print(classification_report(y_test,predictions))
cm = confusion_matrix(y_test,predictions)
print("Confusion Matrix:")
print(cm)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
print("AUC:"+str(metrics.auc(fpr,tpr)))

