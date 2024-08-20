
# Standard operational package imports.
import numpy as np
import pandas as pd

# Important imports for preprocessing, modeling, and evaluation.
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

# Visualization package imports.
import matplotlib.pyplot as plt
import seaborn as sns



df_original = pd.read_csv("Invistico_Airline.csv")

df_original.head(n = 10)


# ### Explore the data


df_original.dtypes


df_original['satisfaction'].value_counts(dropna = False)


# ### Check for missing values

df_original.isnull().sum()

# ### Drop the rows with missing values

df_subset = df_original.dropna(axis=0).reset_index(drop = True)

df_subset = df_subset.astype({"Inflight entertainment": float})


# Convert the categorical column `satisfaction` into numeric through one-hot encoding.



df_subset['satisfaction'] = OneHotEncoder(drop='first').fit_transform(df_subset[['satisfaction']]).toarray()



# ### Output the first 10 rows of `df_subset`
# 
# To examine what one-hot encoding did to the DataFrame, output the first 10 rows of `df_subset`.



df_subset.head(10)


# ### Create the training and testing data
# 
# Put 70% of the data into a training set and the remaining 30% into a testing set. Create an X and y DataFrame with only the necessary variables.

X = df_subset[["Inflight entertainment"]]
y = df_subset["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# ## Step 3: Model building

# ### Fit a LogisticRegression model to the data

clf = LogisticRegression().fit(X_train,y_train)

# ### Obtain parameter estimates

clf.coef_

clf.intercept_


# ### Create a plot 

sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None)


# Save predictions.

y_pred = clf.predict(X_test)

# Use predict_proba to output a probability.


clf.predict_proba(X_test)

clf.predict(X_test)


print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))


# ### Produce a confusion matrix


cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = clf.classes_)
disp.plot()


