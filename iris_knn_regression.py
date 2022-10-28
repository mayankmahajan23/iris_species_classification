import numpy as np # numerical python (scientific computing)
import pandas as pd # for data manipulation and analysis
import matplotlib.pyplot as plt # for plotting
from sklearn.datasets import load_iris # sci-kit learn : machine learning library
    # load_iris loads and returns the iris dataset (a bunch object)
import msvcrt

mainData = load_iris() # bunch onject - strings, numeric values, etc. (multiple data types in hybrid)
# every machine learning data has two parts - input and target
# system will learn with these, then we will provide only input and get target from computer
print(mainData)
print(type(mainData))

irisData = mainData['data'] # extracting data part of load_iris
irisTarget = mainData['target'] # extracting target part of load_iris
irisFeatureNames = mainData['feature_names'] # extracting Feature Names part of load_iris

from sklearn.preprocessing import StandardScaler # normalization
scaler = StandardScaler()
# note: this only takes 2d arrays as parameters
scaler.fit(irisData) # compute the mean and std to be used for later scaling
# fit_transform: fit and then transform, inverse_transform: scale back data to the original form
print(scaler.mean_)
irisData = scaler.transform(irisData) # perform standardization by centering and scaling
msvcrt.getch()

print(irisData)
print(irisTarget)
print(type(irisData)) # gives type
print(type(irisTarget))
print(irisData.shape) # gives dimensions
print(irisTarget.shape)
# shape of the data is number of samples multiplied by number of features
msvcrt.getch()

'''
# method 2 of reading file
x = pd.read_csv("iris.csv") # comma seperated values
print(x.head())
print(x.tail())

x1 = np.array(x)
x2 = x1[:,:-1] # slicing the array, taking all (:) rows, but taking all-1 (:-1) columns. negative values start from the back
# similarly x2 = x1[:,:+1] means all rows and all till 1 columns. positve values start from the front
'''
# inspect your data
d1 = irisData[:50,:]
d2 = irisData[50:100,:]
d3 = irisData[100:150,:]

plt.subplot(2,3,1)
plt.scatter(d1[:,0], d1[:,1], 10, color = 'r')
plt.scatter(d2[:,0], d2[:,1], 10, color = 'g')
plt.scatter(d3[:,0], d3[:,1], 10, color = 'b')

plt.subplot(2,3,2)
plt.scatter(d1[:,0], d1[:,2], 10, color = 'r')
plt.scatter(d2[:,0], d2[:,2], 10, color = 'g')
plt.scatter(d3[:,0], d3[:,2], 10, color = 'b')

plt.subplot(2,3,3)
plt.scatter(d1[:,0], d1[:,3], 10, color = 'r')
plt.scatter(d2[:,0], d2[:,3], 10, color = 'g')
plt.scatter(d3[:,0], d3[:,3], 10, color = 'b')

plt.subplot(2,3,4)
plt.scatter(d1[:,1], d1[:,2], 10, color = 'r')
plt.scatter(d2[:,1], d2[:,2], 10, color = 'g')
plt.scatter(d3[:,1], d3[:,2], 10, color = 'b')

plt.subplot(2,3,5)
plt.scatter(d1[:,1], d1[:,3], 10, color = 'r')
plt.scatter(d2[:,1], d2[:,3], 10, color = 'g')
plt.scatter(d3[:,1], d3[:,3], 10, color = 'b')

plt.subplot(2,3,6)
plt.scatter(d1[:,2], d1[:,3], 10, color = 'r')
plt.scatter(d2[:,2], d2[:,3], 10, color = 'g')
plt.scatter(d3[:,2], d3[:,3], 10, color = 'b')

plt.clf()
# scatter plot puts one feature on the x axis and other on the y axis. limitation - 2 dimensions only, maybe 3 dimensions
# pair plot - looks at all possible pairs of features. if small number of features like 4 in our case, but not with large number of feature
# still doesn't show all at once, so some interesting aspects of data may not be shown this way
# first convert numpy array into pandas dataframe, because - pandas has a function to plot scatter mattrix,
# there are total 12 scatter plots for 4 features (4C2 = 6: hence only 6 of them are unique)
    # the diagonals of this matrix are histograms of each feature
import mglearn
irisDataFrame = pd.DataFrame(irisData, columns = irisFeatureNames)
pd.plotting.scatter_matrix(irisDataFrame, c = irisTarget, figsize = (15,15), marker = 'o', hist_kwds = {'bins':20}, s = 60, alpha = .8, cmap = mglearn.cm3)
plt.show()
plt.clf()
msvcrt.getch()

# split data into train and test
# usually data is denoted by X and labels are denoted by y (sci-kit learn)
from sklearn.model_selection import train_test_split

accuracy = []
neighbors = []

i = 3 # number of neighbors
while i <= 27 :
    neighbors.append(i)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = i)

    j = 0.02
    result = []
    while j <= 0.70 : # 0.78 corresponding to neighbors = 30
        X_train, X_test, y_train, y_test = train_test_split (irisData, irisTarget, test_size = j) # shuffle = 'true' by default
        # test_size = None (default) and if train_size is also none, then 0.25. train_size = None (default) and the value is automatically set to complement of test_size if not mentioned

        knn.fit(X_train, y_train) # fit the model using training data and it's corresponding target values
        # making predictions
        X_new = np.array([[5,2.9,1,0.2], [6.2,3.4,5.4,2.3]]) # sending feature values for prediction
        prediction = knn.predict(X_new)
        #print(prediction)
        #print(mainData['target_names'][prediction])

        # testing
        y_predict = knn.predict(X_test)
        result.append( knn.score(X_test, y_test) )
        #print( np.mean(y_predict == y_test) ) # same
        from sklearn import metrics
        #print(metrics.accuracy_score(y_test, y_predict)) # same

        j += 0.03

    i += 2 # becuase i must be odd
    accuracy.append(result)

testFraction = []
j = 0.02
while j <= 0.70 :
    testFraction.append(j)
    j += 0.03

for i in range(len(neighbors)) :
    plt.plot(testFraction, accuracy[i], 'b')
plt.show()
msvcrt.getch()
plt.clf()

# dbt - how in the world it gives great accuracy for insane test sizes like 145 out of 150 - logistic regression
from sklearn.linear_model import LogisticRegression
i = 0.02
testFraction.clear()
accuracy.clear()
while i <= 0.98 :
    X_train, X_test, y_train, y_test = train_test_split (irisData, irisTarget, random_state = 0, test_size = i)
        # becuase messed up train and test split upstairs for plotting purposes
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    #prediction = logreg.predict(X_new)
    #print(prediction)

    score = logreg.score(X_test, y_test)
    print(score)
    print(y_test.shape)
    accuracy.append(score)
    testFraction.append(i)
    i += 0.02

plt.plot(testFraction, accuracy, 'b')
plt.show()
plt.clf()

# we had a convergence warning (lbfgs failed to converge, max iterations reached): when an optimization algorithm fails to converge, it is becuase the
# problem is not problem is not well-conditioned - 1. normalization, 2. set max_iter to a larger value (default is 1000)
    # our problem solved after normalizing the data

''' notes:
pip - package installer/manager for python

requirements for working with sci-kit learn
    features and response
        are seperate objects
        should be numeric
        should be numpy array
        should have specific shapes

# classification - response is categorical (supervised learning)
# regression - response is ordered and continous

syntaxNeighborsClassifier (n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)
    n_neighbors : number of neighbors use by default for kneighbor queries (defualt = 5)
    weight function used in prediction. Possible values: ‘uniform’, ‘distance’, [callable]
    Algorithm : algo used to compute the nearest neighbors: ‘ball_tree’, ‘kd_tree’, ‘brute’, ‘auto’
        Note: fitting on sparse input will override the setting of this parameter, using brute force
    leaf_size : default = 30
    p - integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used
    metric: string or callable, default ‘minkowski’
    metric_params
    n_jobs

kneighbor - finds k neighbors of a point
'''
