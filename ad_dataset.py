# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

from factor_analyzer import FactorAnalyzer
# %%
"""# Dataset"""

df = pd.read_csv('FB AD Campaign_Modify_Final.csv')
df.head()
# %%
"""# EDA"""

# Converting numbers into float type
cov_col = ['Reach','Impressions','ThruPlays','Amount spent (INR)','Post engagement','Video plays at 25%','Video plays at 50%','Video plays at 75%','Video plays at 100%']

for col in cov_col :
  df[col] = df[col].replace(',','',regex=True).astype(float)

df.head()

# %%
"""### Converting Post engagement into binary"""

# Mean of Post engagement column
mean = df['Post engagement'].mean()
mean

# Labeling
cat = []
for i in df['Post engagement']:
  if i < mean:
    cat.append('Less than mean')
  else:
    cat.append('More than mean')

df['Post engagement (categorical)'] = cat

# Converting to binary data
df_one = pd.get_dummies(df['Post engagement (categorical)'])
df_one = df_one.drop(['Less than mean'], axis=1)

# Combinig two dataframe
df_two = df.join(df_one)
df_two = df_two.drop(['Post engagement','Post engagement (categorical)'], axis = 1)
df_two.rename(columns = {'More than mean':'Post engagement'}, inplace = True)
df_two.head()

df_two['Post engagement'] = df_two['Post engagement'].astype(float)

# %%
"""### Box plots"""
for i in df_two.columns:
  if df_two[i].dtype == 'float64':
    box = df_two.boxplot(column=[i])
    plt.show()

# %%
"""Converting Platform to binary"""
df_two['Platform'] = df_two['Platform'].apply(lambda x: 1 if x.strip()=='FB' else 0)

# %%
"""### Treating outlier"""

# Number of outliers in each column
Q1 = df_two.quantile(0.25)
Q3 = df_two.quantile(0.75)
IQR = Q3 - Q1
print(((df_two < (Q1 - 1.5 * IQR)) | (df_two > (Q3 + 1.5 * IQR))).sum())

# Removing the first 4 coulmns
columns = list(df_two.columns)
del columns[:4]

# Winsorizing the outlier
for col in columns:
  df_two[col] = winsorize(df_two[col], limits=[0.1, 0.3])

# Value count of target variable
freq = df_two['Post engagement'].value_counts()
print(freq)
print((freq / (freq.sum()))*100)

# %%
"""# Logistics Regression"""

# Select feature set and target column
X = df_two.drop(['Campaign name','Month','Ad name','Post engagement'], axis = 1)
y = df_two['Post engagement']

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# Calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

print(vif_data)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
"""### Logistic Regressin using sklearn"""

# Build the logistic regression model
lr = LogisticRegression(random_state=0)
lr.fit(X, y)

# Evaluate the model using accuracy, confusion matrix, and classification report
y_pred = lr.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Compute ROC curve and ROC area for each class
y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC: {roc_auc}")

# Plot ROC curve
plt.figure()
plt.plot(fpr,tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# %%
"""### Logistic Regression using statsmodel"""

# Building the model and fitting the data
log_reg = sm.Logit(y_train, X_train).fit()

# Printing the summary table
print(log_reg.summary())

# performing predictions on the test dataset
yhat = log_reg.predict(X_test)
prediction = list(map(round, yhat))

# Confusion matrix
cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix : \n", cm)

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test, prediction))

# %%
"""# PCA + Logistics Regression

### PCA
"""

X = df_two.drop(['Campaign name','Month','Ad name','Post engagement'], axis = 1)
X.head()

# Mean
X_mean = X.mean()

# Standard deviation
X_std = X.std()

# Standardization
Z = (X - X_mean) / X_std

# Covariance
c = Z.cov()

# Plot the covariance matrix
sns.heatmap(c)
plt.show()

eigenvalues, eigenvectors = np.linalg.eig(c)
print('Eigen values:\n', eigenvalues)
print('Eigen values Shape:', eigenvalues.shape)
print('Eigen Vector Shape:', eigenvectors.shape)

# Index the eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]

# Sort the eigenvalues in descending order
eigenvalues = eigenvalues[idx]

# sort the corresponding eigenvectors accordingly
eigenvectors = eigenvectors[:,idx]

explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
explained_var

n_components = np.argmax(explained_var >= 0.90) + 1
n_components

# PCA component or unit matrix
u = eigenvectors[:,:n_components]
pca_component = pd.DataFrame(u, index = X.columns, columns = ['PC1','PC2','PC3','PC4','PC5'])

# plotting heatmap
plt.figure(figsize =(5, 7))
sns.heatmap(pca_component)
plt.title('PCA Component')
plt.show()

# Matrix multiplication or dot Product
Z_pca = Z @ pca_component
# Rename the columns name
Z_pca.rename({'PC1': 'PCA1', 'PC2': 'PCA2', 'PC3': 'PCA3', 'PC4': 'PCA4', 'PC5': 'PCA5'}, axis=1, inplace=True)
# Print the  Pricipal Component values
print(Z_pca)

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = Z_pca.columns

# Calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(Z_pca.values, i)
                          for i in range(len(Z_pca.columns))]

print(vif_data)

# %%
"""### PCA with sklearn"""

# Select feature set and target column
X = df_two.drop(['Campaign name','Month','Ad name','Post engagement','Post engagement'], axis = 1)
y = df_two['Post engagement']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to the Data
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
"""### Logistics Regression after PCA using sklearn"""

# Build the logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train_pca, y_train)

# Evaluate the model using accuracy, confusion matrix, and classification report
y_pred = lr.predict(X_test_pca)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Compute ROC curve and ROC area for each class
y_pred_proba = lr.predict_proba(X_test_pca)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC: {roc_auc}")

# Plot ROC curve
plt.figure()
plt.plot(fpr,tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# %%
"""### Logistics Regression after PCA using statsmodel"""

# Building the model and fitting the data
log_reg = sm.Logit(y_train, X_train_pca).fit()

# Printing the summary table
print(log_reg.summary())

# performing predictions on the test dataset
yhat = log_reg.predict(X_test_pca)
prediction = list(map(round, yhat))

# Confusion matrix
cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix : \n", cm)

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test, prediction))


# %%
"""# Hotelling T-Squared Test"""

# %%
"""# Factor Analysis + Logistic Regression"""
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

"""# Checking Assumptions"""
chi_square_value,p_value=calculate_bartlett_sphericity(X)
print(chi_square_value, p_value)

kmo_all,kmo_model=calculate_kmo(X)
print(kmo_model)

# %%
# Correlation Matrix and heatmap
X.corr()
sns.heatmap(data=X.corr())
# %%
# Create factor analysis object and perform factor analysis

fa = FactorAnalyzer(n_factors=3, rotation='varimax')
x_fa = fa.fit_transform(X)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

# %%
# Create scree plot using matplotlib
plt.scatter(range(1,X.shape[1]+1),ev)
plt.plot(range(1,X.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# %%
# print factor loadings
print(fa.loadings_)

# print factor communalites
print(fa.get_communalities())
load = pd.DataFrame(fa.loadings_, index=X.columns)
load
x_fa = pd.DataFrame(x_fa, columns= ['Factor 1', 'Factor 2', 'Factor 3'])
x_train_fa, x_test_fa, y_train, y_test = train_test_split(x_fa, y, test_size = 0.2, random_state=42)


# %%
"""# Logistic Regression using sklearn on factor analysis"""

lr = LogisticRegression(random_state=42)
lr.fit(x_train_fa, y_train)


# printing evaluation metrics
y_pred = lr.predict(x_test_fa)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))


# %%
"""# Logistic Regression using statsmodels after factor analysis"""
log_reg = sm.Logit(y_train, x_train_fa).fit()

# Printing the summary table
print(log_reg.summary())

# performing predictions on the test dataset
yhat = log_reg.predict(x_test_fa)
prediction = list(map(round, yhat))

# Confusion matrix
cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix : \n", cm)

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test, prediction))


  # %%
"""# Discriminant Analysis"""

# %%
