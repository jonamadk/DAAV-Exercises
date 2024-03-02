import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB


dataframe = pd.read_csv("ClassificationLabData.csv")


# DATA PRE-PROCESSING
# 1. Finding NA/NUll
print("Checking NA value:", dataframe.isnull().sum())


# 2. Overviewing columns unique data value
column_unique_value_dict = dict()
for column in dataframe.columns.tolist():
    column_unique_value_dict[column] = dataframe[column].unique()


#3 Finding '?'
features_with_question_mark = []
for column in dataframe.columns.tolist():
    if any(dataframe[column] == " ?"):
        features_with_question_mark.append(column)


#3. Performing MODE Imputation on the 'worktype' and 'CurrentOccupation'
for column in features_with_question_mark:
    dataframe[column]= dataframe[column].replace(" ?", pd.NA)
    dataframe[column] = dataframe[column].fillna(dataframe[column].mode()[0])


#Label Encoding on Taget Variable 'Label' Column
label_encoder = LabelEncoder() 
dataframe['Label'] = label_encoder.fit_transform(dataframe['Label'])

# One Hot Encoding on the Features
dataframe = pd.get_dummies(dataframe, dtype=int)
dataframe.replace({True:1,False:0}, inplace=True)

X = dataframe.drop(['Label'], axis=1)
y = dataframe['Label']


#Splitting data to train-test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)


#Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)



print("Confusion Matrix for Decision Tree:\n",confusion_matrix(y_test, y_pred))
print("Accuracy Decision Tree:\n",classification_report(y_test, y_pred))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)


#Naive Bayes Classifier
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print("Confusion Matrix for NB:\n", confusion_matrix(y_test, y_pred))
print("Accuracy NB:\n",classification_report(y_test, y_pred))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)