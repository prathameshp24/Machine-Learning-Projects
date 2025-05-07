import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("WineQT.csv")


plt.figure(figsize=(8, 5))
sns.countplot(data=data, x="quality", palette="viridis")
plt.title("Distribution of Wine Quality", fontsize=14)
plt.xlabel("Quality", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


X = data.drop("quality", axis=1)
y = data["quality"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


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
