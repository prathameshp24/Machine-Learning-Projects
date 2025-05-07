import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    'fixed acidity': [7.4, 7.8, 7.8, 11.2, 7.4, 7.4, 7.9, 7.3, 7.8, 6.7, 5.6, 7.8, 8.5, 7.9],
    'volatile acidity': [0.7, 0.88, 0.76, 0.28, 0.7, 0.66, 0.6, 0.65, 0.58, 0.58, 0.615, 0.61, 0.28, 0.32],
    'citric acid': [0.0, 0.0, 0.04, 0.56, 0.0, 0.0, 0.06, 0.0, 0.02, 0.08, 0.0, 0.29, 0.56, 0.51],
    'residual sugar': [1.9, 2.6, 2.3, 1.9, 1.9, 1.8, 1.6, 1.2, 2.0, 1.8, 1.6, 1.6, 1.8, 1.8],
    'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076, 0.075, 0.069, 0.065, 0.073, 0.09699999999999999, 0.089, 0.114, 0.092, 0.341],
    'free sulfur dioxide': [11.0, 25.0, 15.0, 17.0, 11.0, 13.0, 15.0, 15.0, 9.0, 15.0, 16.0, 9.0, 35.0, 17.0],
    'total sulfur dioxide': [34.0, 67.0, 54.0, 60.0, 34.0, 40.0, 59.0, 21.0, 18.0, 65.0, 59.0, 29.0, 103.0, 56.0],
    'density': [0.9978, 0.9968, 0.997, 0.998, 0.9978, 0.9978, 0.9964, 0.9946, 0.9968, 0.9959, 0.9943, 0.9974, 0.9969, 0.9969],
    'pH': [3.51, 3.2, 3.26, 3.16, 3.51, 3.51, 3.3, 3.39, 3.36, 3.28, 3.58, 3.26, 3.3, 3.04],
    'sulphates': [0.56, 0.68, 0.65, 0.58, 0.56, 0.56, 0.46, 0.47, 0.57, 0.54, 0.52, 1.56, 0.75, 1.08],
    'alcohol': [9.4, 9.8, 9.8, 9.8, 9.4, 9.4, 9.4, 10.0, 9.5, 9.2, 9.9, 9.1, 10.5, 9.2],
    'quality': [5, 5, 5, 6, 5, 5, 5, 7, 7, 5, 5, 5, 7, 6],
}


df = pd.DataFrame(data)


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="quality", palette="viridis")
plt.title("Distribution of Wine Quality", fontsize=14)
plt.xlabel("Quality", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


X = df.drop("quality", axis=1)
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


clf = DecisionTreeClassifier(max_depth=10, random_state=1234)
clf.fit(X_train, y_train)


predictions = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc:.2f}")


plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree Visualization", fontsize=16)
plt.show()
