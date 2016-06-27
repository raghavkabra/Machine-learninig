## 1. Classification ##

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# We will use pandas to work with the data
import pandas

# Read file
credit = pandas.read_csv("credit.csv")

# Dataframe of our data
# credit["model_score"] is the probability provided by the model
# credit["paid"] is the observed payments
# .head(10) shows the first 10 rows of the dataframe
print(credit.head(10))
plt.scatter(credit["model_score"], credit["paid"])
plt.show()

## 2. Introduction to the data ##

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
admissions = pd.read_csv("admissions.csv")
plt.scatter(admissions['gpa'], admissions['admit'])
plt.show()

## 3. Logistic regression ##

# prediction with discrimination threshold at 0.50
pred = credit["model_score"] > 0.5

# number of true positives
TP = sum(((pred == 1) & (credit["paid"] == 1)))
print(TP)
FN = sum(((pred == 0) & (credit["paid"] == 1)))

## 4. Logit function ##

import numpy as np

# Logit Function
def logit(x):
    # np.exp(x) raises x to the exponential power, ie e^x. e ~= 2.71828
    return np.exp(x)  / (1 + np.exp(x)) 
    
# Generate 50 real values, evenly spaced, between -6 and 6.
x = np.linspace(-6,6,50, dtype=float)

# Transform each number in t using the logit function.
y = logit(x)

# Plot the resulting data.
plt.plot(x, y)
plt.ylabel("Probability")
plt.show()


## 5. Training a logistic regression model ##

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(admissions[["gpa"]], admissions["admit"])
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(admissions[["gpa"]], admissions["admit"])

## 6. Plotting probabilities ##

logistic_model = LogisticRegression()
logistic_model.fit(admissions[["gpa"]], admissions["admit"])
pred_probs = logistic_model.predict_proba(admissions[["gpa"]])
plt.scatter(admissions["gpa"], pred_probs[:,1])

## 7. Predict labels ##

logistic_model = LogisticRegression()
logistic_model.fit(admissions[["gpa"]].values, admissions["admit"])
fitted_labels = logistic_model.predict(admissions[["gpa"]])
plt.scatter(admissions["gpa"].values, fitted_labels)

## 8. Next steps ##

from sklearn.metrics import roc_auc_score

probs = [ 0.98200848,  0.92088976,  0.13125231,  0.0130085,   0.35719083,  
         0.34381803, 0.46938187,  0.53918899,  0.63485958,  0.56959959]
obs = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]

testing_auc = roc_auc_score(obs, probs)
print("Example AUC: {auc}".format(auc=testing_auc))
auc = roc_auc_score(credit["paid"], credit["model_score"])