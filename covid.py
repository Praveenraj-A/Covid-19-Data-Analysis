""" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load a smaller sample of your dataset for testing
# Adjust nrows to load a smaller subset for testing
covid_df = pd.read_csv('covid.csv', nrows=1000)

# Step 2: Preprocess your data
numerical_cols = covid_df.select_dtypes(include=['int', 'float']).columns
categorical_cols = covid_df.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder()
encoded_categorical_cols = encoder.fit_transform(covid_df[categorical_cols])
encoded_categorical_cols = pd.DataFrame(encoded_categorical_cols.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate numerical and encoded categorical columns
X = pd.concat([covid_df[numerical_cols], encoded_categorical_cols], axis=1)
y = covid_df["Resident (Y/N)"]

# Step 3: Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Instantiate and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Adjust n_neighbors for faster training
knn.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
 """

""" import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv("covid.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=['Resident']).values  # Features
y = df['Resident'].values  # Target variable

# Encode categorical features to numerical values
label_encoders = [LabelEncoder() for _ in range(X.shape[1])]
for i in range(X.shape[1]):
    X[:, i] = label_encoders[i].fit_transform(X[:, i])

gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy) """


import matplotlib.pyplot as plt
import pandas as pd
covid_df = pd.read_csv('covid.csv')

covid_df.BHRCode.plot.hist()
plt.show()
print("Unique values in ' Borough code':", covid_df.Boroughcode.unique())
print("Value counts for ' Borough code':")
print(covid_df.Boroughcode.value_counts())                 
covid_df.Boroughcode.value_counts().plot.bar()
plt.show()