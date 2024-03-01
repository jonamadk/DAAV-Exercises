import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt




dataframe = pd.read_csv("ClassificationLabData.csv")



#Data EDA
# print(dataframe.info())
# print(dataframe.head())

print("Numeric Columns List", dataframe.select_dtypes(include = ['number']).columns.tolist())
print("Categorical Columns List", dataframe.select_dtypes(exclude=['number']).columns.tolist())


# DATA PRE-PROCESSING
# 1. Finding NA/NUll
print(dataframe.isnull().sum())


# 2. Overviewing columns unique data value
column_unique_value_dict = dict()
for column in dataframe.columns.tolist():
    column_unique_value_dict[column] = dataframe[column].unique()


#3 Finding '?'
features_with_question_mark = []
for column in dataframe.columns.tolist():
    if any(dataframe[column] == " ?"):
        features_with_question_mark.append(column)


# #3. Performing MODE Imputation on the 'worktype' and 'CurrentOccupation'
# for column in features_with_question_mark:
#     dataframe[column]= dataframe[column].replace(" ?", pd.NA)
#     dataframe[column] = dataframe[column].fillna(dataframe[column].mode()[0])


#Label Encoding on Taget Variable 'Label' Column
label_encoder = LabelEncoder()
dataframe['Label'] = label_encoder.fit_transform(dataframe['Label'])

# One Hot Encoding on the Features
dataframe = pd.get_dummies(dataframe)
dataframe = dataframe.replace({True:1, False:0})


X = dataframe.drop(['Label'], axis=1)
y = dataframe['Label']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)
dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train, y_train)


y_pred = dt_clf.predict(X_test)


print(accuracy_score(y_pred, y_test))


false_positive_rate, true_positive_rate, threshold = roc_curve(y_test,y_pred)
plt.figure(figsize=(10,6))
plt.title('ROC for decision tree')
plt.plot(false_positive_rate, true_positive_rate, linewidth=5, color='green')
plt.plot([0,1],ls='--',linewidth=5)
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.text(0.2,0.6,'AUC: {:.2f}'.format(roc_auc_score(y_test,y_pred)),size= 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()