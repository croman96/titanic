'''

Author: Carlos Roman Rivera
Date:   21-Feb-2019

Kaggle: Titanic
Type: 	Logistic Regression

'''

import random

import math
import pandas as pd
import numpy as np
import random as rnd

import seaborn as sbn
import matplotlib.pyplot as plt

# For testing data, ups.
from sklearn.linear_model import LogisticRegression

# Parse data input.

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
combine = [train_data, test_data]

# Describing data.

print(train_data.columns.values)
print("\n")
print(train_data.head())
print("\n")
print(train_data.describe())
print("\n")
print(train_data.describe(include=['O']))
print("\n")

# Data statistics.

print(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
print(train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
print(train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
print(train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
print(train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")

# Data graphs.

graph_age = sbn.FacetGrid(train_data, col='Survived')
graph_age.map(plt.hist, 'Age', bins=10)

graph_pclass = sbn.FacetGrid(train_data, col='Survived')
graph_pclass.map(plt.hist, 'Pclass', bins=10)

# Drop cabin.

train_data = train_data.drop(['Cabin'], axis=1)
test_data = test_data.drop(['Cabin'], axis=1)
combine = [train_data, test_data]

# Drop ticket.

train_data = train_data.drop(['Ticket'], axis=1)
test_data = test_data.drop(['Ticket'], axis=1)
combine = [train_data, test_data]

# Extract title from name before dropping it.

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])

# Normalize titles into mr, miss, mrs, master, other.

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Map title from categorical to ordinal values.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Drop name.

train_data = train_data.drop(['Name', 'PassengerId'], axis=1)
test_data = test_data.drop(['Name'], axis=1)
combine = [train_data, test_data]

# Map sex from categorical to ordinal value.

sex_mapping = {'female': 1, 'male': 0}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)

# Calculate age based on sex and pclass to complete missing values.

guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

# Group ages to get upper and lower values.

train_data['AgeGroup'] = pd.cut(train_data['Age'], 5)

# Change Age for its given group calculated above.

for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

# Drop auxiliary column AgeBand.

train_data = train_data.drop(['AgeGroup'], axis=1)
combine = [train_data, test_data]

# Merge SibSp and Parch into one feature.

for dataset in combine:
    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1

train_data[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Check if passenger traveled alone or with family.

for dataset in combine:
    dataset['Alone'] = 0
    dataset.loc[dataset['Family'] == 1, 'Alone'] = 1

train_data[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean() # Damn, should've brought family.

# Drop Parch, SipSp and Family. Keep Alone.

train_data = train_data.drop(['Parch', 'SibSp', 'Family'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'Family'], axis=1)
combine = [train_data, test_data]

# Combine age and class.

for dataset in combine:
    dataset['AgeClass'] = dataset.Age * dataset.Pclass

# Fill embark column (2 values) with most popular value.

go_to_port = train_data.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(go_to_port)

train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False) # Damn, should've travelled from C.

# Embark from categorical to ordinal value.

for dataset in combine:
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)

# Fill Fare missing values with median.

test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

# Group fares.

train_data['FareGroup'] = pd.qcut(train_data['Fare'], 4)

# Convert Fare to ordinal.

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_data = train_data.drop(['FareGroup'], axis=1)
combine = [train_data, test_data]

# Logistic Regression, Cost Function and Gradient Descent

# Evaluate hypothesis given parameters and the sample.
def hypothesis(params, sample):
    aux = 0
    # Sum all parameters, used for sigmoid function.
    for i in range(len(params)):
        aux = aux + params[i] * sample[i]
    aux = aux*(-1);
    # Sigmoid function.
    aux = 1/(1+ math.exp(aux));
    return aux;

# Evaluate the error of the current parameters.
def display_error(params, samples, labels):
	global graph_errors
	acum_error = 0
	error = 0
	for i in range(len(samples)):
        # Get predicted output with current parameters.
		hyp = hypothesis(params,samples[i])
        # Cross-entropy function.
		if(labels[i] == 1):
			if(hyp == 0): # Avoid log(0)
				hyp = .0001;
			error = (-1)*math.log(hyp);
		if(labels[i] == 0):
			if(hyp == 1): # Avoid log(0)
				hyp = .9999;
			error = (-1)*math.log(1-hyp);
        # Accumulative error.
		acum_error += error
    # Error for current sample.
	mean_error = acum_error/len(samples);
	graph_errors.append(mean_error)
	return mean_error;

# Gradient descend used to optimize hypothesis.
def gradient_descent(params, samples, labels, learning_rate):
    temp = list(params)
    general_error = 0
    for j in range(len(params)):
        acum = 0
        acum_error = 0
        for i in range(len(samples)):
            error = hypothesis(params,samples[i]) - labels[i]
            acum = acum + error * samples[i][j]
        # Update parameters given the learning rate and the gradient.
        temp[j] = params[j] - learning_rate * (1/len(samples)) * acum
    return temp
'''
# Train

# Guys we can play with.
learning_rate = .05
epochs = 3000

# Random parameters.
params = []
for i in range(0,8):
    params.append(random.uniform(0,1))

print(params)

# Training set.
features = train_data.drop("Survived", axis=1)
features = features.apply(lambda y: y.tolist(), axis=1)
labels = train_data["Survived"]

# Auxiliary variables.
current_epoch = 0
graph_errors = [];

while True:
    oldparams = list(params)
    params = gradient_descent(params,features,labels,learning_rate)
    error = display_error(params,features,labels)
    if(oldparams == params or current_epoch > epochs):
        print("Final Parameters:")
        print(params)
        break
    print("epoch:\t" + str(current_epoch) + "/" + str(epochs) + "\terror:\t" + str(error))
    current_epoch+=1

plt.figure()
plt.plot(graph_errors)
plt.title('Training Error')
plt.ylabel('error')
plt.xlabel('epoch')
'''
# Test

test_parameters = [-0.7559236308729694, 2.255160806671083, 0.28939615321724843, -0.08266902275063291, 0.26843528110325704, 0.39879200609912974, 0.1442149360189284, -0.31317974627250483]
test_set = test_data.drop("PassengerId", axis=1).copy()
test_set_rows = test_set.apply(lambda y: y.tolist(), axis=1)

print(len(test_set_rows))
results = []
for test in test_set_rows:
    results.append(round(hypothesis(test_parameters,test)))

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": results
})

submission.to_csv('data/submission.csv', index=False)

print(submission)

# Show graphs.

plt.show()
