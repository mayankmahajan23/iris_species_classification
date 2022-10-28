import numpy as np # numerical python (scientific computing)
import pandas as pd # for data manipulation and analysis
import matplotlib.pyplot as plt # for plotting
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import msvcrt

x = pd.read_csv("iris.csv")
data = np.array(x)
features = data[:,:4]
target =  data[:,4:]

initial_max_depth = 2
final_max_depth = 8
depth_increment = 2

initial_split = 0.1
final_split = 0.8
split_increment = 0.01

split_history = []
split = initial_split
while split <= final_split :
    split_history.append(split)
    split += split_increment

depth = initial_max_depth
while depth <= final_max_depth :
    score_history = []
    split = initial_split
    while split <= final_split :
        X_train, X_test, y_train, y_test = train_test_split (features, target, random_state = 0, test_size = split)

        clf = DecisionTreeClassifier(max_depth = depth)
        clf = clf.fit(X_train, y_train.ravel())
        y_prediction = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, y_prediction)
        score_history.append(score)

        split += split_increment

    plt.plot(split_history, score_history, label = depth)
    depth += depth_increment

plt.legend()
plt.show()
