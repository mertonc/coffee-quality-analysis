# Coffee Quality Data Analysis from CQI Database
**Data Analysis of CQI Database** 

# Description
This project focuses on cleaning, analyzing, and interpreting coffee bean data from Coffee Quality Institute on Kaggle (https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi). The code performs simple machine learning testing on predicting made labels of 'good' and 'not good' coffee. Data illustrations are also done to better understand the data and find other interesting information. 

# Code
* Clean and uniform data
```python
# select columns that we want from the data
columns = ["Species", "Country.of.Origin", "Processing.Method", "Aroma", "Flavor", "Aftertaste",
           "Acidity", "Body", "Balance", "Uniformity", "Moisture", "Total.Cup.Points", "unit_of_measurement", "altitude_mean_meters"]

# cleaning data and making format uniform
df = df[columns]
df.columns = df.columns.str.replace('.', '_')
df.columns = df.columns.str.lower()

# make sure metrics are all in meters
convert = df['unit_of_measurement'].eq('feet')
df.loc[convert, ['altitude_mean_meters']] /= 3.281

df = df.drop(columns="unit_of_measurement")

# clean data from impossible altitudes
# number is highest altitude in the world
df = df[~(df["altitude_mean_meters"] > 8848)]
df.isnull().sum()
df = df.dropna()

df.describe()  # show overview of stats. See that there is altitude with meters of 1 or 4000+
# need to find outliers

# find outliers that are greater than 3
df = df[~(np.abs(stats.zscore(df["altitude_mean_meters"])) > 3)]

#shows boxplot of outliers and data of altitude
sns.boxplot(data=df, y="altitude_mean_meters")
plt.show()
```

* Prepare and process data
```python 
#replace processing methodss with quantified numbers 0 and 1
df['processing_method'] = df['processing_method'].replace(
    ['Washed / Wet'], '0')
df['processing_method'] = df['processing_method'].replace(
    ['Natural / Dry'], '1')

#create x values
correlation_columns = ['aroma', 'flavor', 'aftertaste', 'acidity',
                       'body', 'balance', 'uniformity', 'moisture', 'altitude_mean_meters']

#Reframe input columns after observing correlation heat map. Can retest using the below columns 
#correlation_columns_2 = ['aroma', 'uniformity', 'moisture', 'altitude_mean_meters']

X = df[correlation_columns]

scaler = StandardScaler()

new_df = df[correlation_columns]

x_std = scaler.fit_transform(X)
x_outlier = pd.DataFrame(x_std, columns=correlation_columns)

# sns.pairplot(x_outlier_checking)
corr = x_outlier.corr()
sns.heatmap(corr, annot=True)

#create labels for cup points that are above 80 meaning good coffee
df['label'] = np.where(df['total_cup_points'] > 80, 1, 0)
#df['label_2'] = np.where(df['total_cup_points'] > 85, 1, 0)
#df['label_3'] = np.where(df['total_cup_points'] > 90, 1, 0)

Y = df.label
```

* Create training and testing split. Implement different ML classifiers
```python
#create train test split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#create number of k values for knn
k = [1, 3, 5, 7, 9, 11, 13, 15]

#knn classifier for optimal k (5)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train, y_train)

y_pred = knn_classifier.predict(x_test)
knn_classifier_score = accuracy_score(y_test, y_pred)
knn_error = np.mean(y_pred != y_test)
knn_cm = confusion_matrix(y_test, y_pred)


print("KNN Accuracy:", knn_classifier_score)
print("KNN Error Rate:", knn_error)

#used to find lowest error rate for optimal k 
error_rate = []
for i, k in enumerate(k):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)

    pred_k = knn_classifier.predict(x_test)

    k_error = accuracy_score(y_test, pred_k)
    error_rate.append(k_error)

k_rate = [1], [3], [5], [7], [9], [11], [13], [15]

#code in comment prints out plot for error rate by k
#sns.set()
#plt.plot(k_rate, error_rate, color='red', marker = 'o')
#plt.title('Error Rate by k')
#plt.xlabel('k')
#plt.ylabel('Error Rate')
#plt.show

# Naiive Bayesian Classifier
NB_classifier = GaussianNB().fit(x_train, y_train)

nb_pred = NB_classifier.predict(x_test)

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_cm = confusion_matrix(y_test, nb_pred)
nb_error = np.mean(nb_pred != y_test)
print('Naiive Bayesian Accuracy:', nb_accuracy)
print('Naiive Bayesian Error Rate:', nb_error)

# Decision Tree Classifier
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(x_train, y_train)
tree_predict = clf.predict(x_test)

dt_error = np.mean(tree_predict != y_test)
dt_accuracy = metrics.accuracy_score(y_test, tree_predict)
dt_cm = confusion_matrix(y_test, tree_predict)
print('Decision Tree Accuracy:', dt_accuracy)
print('Decision Tree Error Rate:', dt_error)

# Random Forest Classifier
n_estimators = 10
max_depth = 5

model = RandomForestClassifier(criterion='entropy')

#find optimal n_estimators and max_depth
error_rate2 = []
for i in range(1, n_estimators + 1):  # max_depth + 1
    model.set_params(n_estimators=i)  # max_depth=i
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    error_rate2 = np.mean(prediction != y_test)
    print(i, error_rate2)

n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
error = [0.065, 0.045, 0.05, 0.05, 0.035, 0.05,
         0.04, 0.04, 0.035, 0.04]

plt.plot(n, error)

rf_model = RandomForestClassifier(
    n_estimators=5, max_depth=3, criterion='entropy')
rf_model.fit(x_train, y_train)

rf_pred = rf_model.predict(x_test)
rf_error = np.mean(rf_pred != y_test)
rf_accuracy = metrics.accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Error Rate:", rf_error)

#use all confusion matrixs from different classifiers to calculate TPR and TNR 
rf_cm = confusion_matrix(y_test, rf_pred)
#TPR = TP/(TP+FN)
#TNR = TN/(TN+FP)
```








# My Findings

1. Scatterplot of Altitude and Processing Method

![Scatterplot of Altitude and Processing Method](/imag/altitude_processingmethod.png)

2. Correlation Matrix of Different Factors

![Correlation Matrix](/imag/correlation_matrix.png)

3. Comparison of Different Classifiers

![Comparison of Different ML Classifiers](/imag/coffee-quality-comparison-classifiers.png)

# Libraries
- sklearn, pandas, numpy, matplotlib, seaborn, scipy

# Authors
- Merton Chen

# Version History
- 0.1
  - Initial Release

# Acknowledgement
