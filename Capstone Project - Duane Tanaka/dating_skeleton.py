import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

#Create your df here:
df = pd.read_csv('profiles.csv')
df = df[df.income != -1]
ethnicities = ['white', 'hispanic / latin', 'black', 'asian']
df = df[df.ethnicity.isin(ethnicities)]
education_level = ['graduated from high school', 'graduated from two-year college', 
  'graduated from college/university', 'graduated from masters program', 
  'graduated from ph.d program']
df = df[df.education.isin(education_level)]
df.reset_index(drop=True, inplace=True)

#values
print(df.columns)
print(df.education.head())
print(df.education.value_counts())
print(df.income.head())
print(df.income.value_counts())
print(df.ethnicity.head())
print(df.ethnicity.value_counts())
print(df.shape)

#histogram of income distribution
plt.hist(df.income, bins=20, range=(0, 200000))
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

#scatter plot for income versus education
plt.scatter(df['income'], df['education'], alpha=0.1)
plt.xlim(0, 200000)
plt.title('Income versus Education')
plt.xlabel('Income')
plt.ylabel('Education')
plt.show()

#scatter plot for income versus ethnicity
plt.scatter(df['income'], df['ethnicity'], alpha=0.1)
plt.xlim(0, 200000)
plt.title('Income versus Ethnicity')
plt.xlabel('Income')
plt.ylabel('Ethnicity')
plt.show()

#processing essays
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6",
              "essay7","essay8","essay9"]
# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df["essay_len"] = all_essays.apply(lambda x: len(x))
df['essay_includes_passionate'] = all_essays.apply(lambda x: 'passionate' in x)

#mapping for education
education_mapping = {'graduated from high school': 0, 'graduated from two-year college': 1, 
  'graduated from college/university': 2, 'graduated from masters program': 3, 
  'graduated from ph.d program': 4}
df["education_code"] = df.education.map(education_mapping)

#mapping for ethnicity
ethnicity_mapping = {'white': 0, 'asian': 1, 'black':2, 'hispanic / latin': 3}
df['ethnicity_code'] = df.ethnicity.map(ethnicity_mapping)

#histogram of education distribution
plt.hist(df.education_code, bins=20)
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.show()

#histogram of ethnicity distribution
plt.hist(df.ethnicity_code, bins=20)
plt.xlabel('Ethnicity')
plt.ylabel('Frequency')
plt.show()

feature_data = df[['income', 'ethnicity_code', 'education_code', 'essay_len', 
                   'essay_includes_passionate']]

#scale values
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
feature_data.dropna(inplace=True)

print(feature_data.head())
print('includes passionate')
print(feature_data.essay_includes_passionate.head())
print('ethnicity')
print(feature_data.ethnicity_code.head())
print('income')
print(feature_data.income.head())
print('essay length')
print(feature_data.essay_len.head())
print('education')
print(feature_data.education_code.head())

#multiple linear regression
features = feature_data[['income', 'education_code', 'essay_len']]
ethnicity_choice = feature_data[['ethnicity_code']]
X_train, X_test, y_train, y_test = train_test_split(features, ethnicity_choice,
                                                    test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)
print('Linear Regression')
print('Training Score')
print(model.score(X_train, y_train))
print('Testing Score')
print(model.score(X_test, y_test))

print(sorted(list(zip(['income','education_code', 'essay_len'],model.coef_)),
       key = lambda x: abs(x[1]),reverse=True))

y_predicted = model.predict(X_test)
print('linear regression prediction')
print(y_predicted)

#multiple linear regression scatter plot
plt.scatter(y_test, y_predicted)
plt.title('Linear Regression')
plt.xlabel('Ethnicity')
plt.ylabel('Predicted Ethnicity')
plt.ylim(0, 1)
plt.show()

#K-Nearest Neighbors regressor
regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')
regressor.fit(X_train, y_train)
print('K-Neighbors Regression')
print('Training Score')
print(regressor.score(X_train, y_train))
print('Testing Score')
print(regressor.score(X_test, y_test))

print('predictions using K-Nearest Neighbors Regression')
y_predicted = regressor.predict(X_test)
print(y_predicted)

#K-Nearest Neighbors regression scatter plot
y_predicted = model.predict(X_test)
plt.scatter(y_test, y_predicted)
plt.title('K-Nearest Neighbors Regression')
plt.xlabel('Ethnicity')
plt.ylabel('Predicted Ethnicity')
plt.ylim(0, 1)
plt.show()

#K-Nearest Neighbors classification
features = feature_data[['income', 'education_code', 'ethnicity_code']]
passionate_choice = feature_data[['essay_includes_passionate']]
X_train, X_test, y_train, y_test = train_test_split(features, passionate_choice,
                                                    test_size=0.2, random_state=1)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
print('K-Neighbors Classification')
print('Training Score')
print(classifier.score(X_train, y_train))
print('Testing Score')
print(classifier.score(X_test, y_test))

print('predictions using K-Nearest Neighbors Classification')
y_predicted = classifier.predict(X_test)
print(y_predicted)
print('accuracy')
print(accuracy_score(y_test, y_predicted))
print('precision')
print(precision_score(y_test, y_predicted))
print('recall')
print(recall_score(y_test, y_predicted))

#K-Nearest Neighbors
plt.scatter(y_test, y_predicted)
plt.title('K-Nearest Neighbors Classification')
plt.xlabel('Essay Includes "passion"')
plt.ylabel('Predicted Essay Includes "passion"')
plt.ylim(0, 1)
plt.show()

#SVM classification
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
print('SVM Classification')
print('Training Score')
print(classifier.score(X_train, y_train))
print('Testing Score')
print(classifier.score(X_test, y_test))

print('predictions using SVC Classification')
y_predicted = classifier.predict(X_test)
print(y_predicted)
print('accuracy')
print(accuracy_score(y_test, y_predicted))
print('precision')
print(precision_score(y_test, y_predicted))
print('recall')
print(recall_score(y_test, y_predicted))

plt.scatter(y_test, y_predicted)
plt.title('SVM Classification')
plt.xlabel('Essay Includes "passion"')
plt.ylabel('Predicted Essay Includes "passion"')
plt.ylim(0, 1)
plt.show()



