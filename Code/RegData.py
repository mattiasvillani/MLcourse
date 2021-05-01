# %%
# Trying out some ML models on simulated regression data
# Author: Mattias Villani, https://mattiasvillani.com

# %% Imports and settings
import numpy as np
import pandas as pd
import seaborn as sns;sns.set();sns.set_style("darkgrid")
import matplotlib.pylab as plt
sns.set_context('talk')
np.random.seed(seed=123) # Set the seed for reproducibility

# Set plot defaults
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %% Simulate data
n = 200
sigmaEps = 0.2
w = np.array([0,1])
x = np.random.uniform(0,1,(n,1))
xBasis = x*np.sin(6*x)*np.exp(x)
y = w[0] + w[1]*xBasis + sigmaEps*np.random.standard_t(df=3, size=(n,1))
y[x[:,0]<0.3,0] = 1 + sigmaEps*np.random.standard_t(df=3, size=np.sum(x[:,0]<0.3))

# %% Split the data into training and testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.5, random_state = 123)
print('Number of obs for training:',len(yTrain))
print('Number of obs for testing:',len(yTest))

# %% Plot the data
f = plt.figure(figsize=(8, 6))
plt.scatter(xTrain,yTrain,s=10)
plt.scatter(xTest,yTest,s=10)
plt.xlabel("x");plt.ylabel("y");
plt.title('Data', fontsize=12)
plt.legend(labels=['Training data','Test data'], loc = 'upper right', fontsize = 12);
f.savefig("RegData.pdf")

# %% Fitting a linear model
from sklearn import linear_model # submodule with linear models
regModel = linear_model.LinearRegression() # Instantiating the Linear regression object
regModel.fit(X = xTrain, y = yTrain);
print('w_0 = ',regModel.intercept_) #
print('w_1 = ',regModel.coef_)

# Plotting the fit
xGrid = np.linspace(np.min(xTrain),np.max(xTrain),1000)
xGrid = xGrid.reshape(-1,1) # Convert it to matrix, as required by the predict method.
f = plt.figure(figsize=(8, 6))
plt.title('Linear', fontsize=12)
plt.scatter(xTrain, yTrain, s = 10)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
yFit = regModel.predict(xGrid)
plt.plot(xGrid, yFit, 'C2');
f.savefig("RegDataLinear.pdf")


# %% Fitting polynomial models
from sklearn.preprocessing import PolynomialFeatures # Not construct polynomials
from sklearn.metrics import mean_squared_error # Simple function that computes MSE

regModel = linear_model.LinearRegression()
f = plt.figure(figsize=(15,15))
polyOrders = (1,2,3,5,7,10,15,20) # Orders of the fitted polynomials
RMSEtrain = np.zeros(len(polyOrders))
RMSEtest = np.zeros(len(polyOrders))
for count,polyDegree in enumerate(polyOrders):
    
    # Fitting the polynomial model on the training data
    poly = PolynomialFeatures(degree=polyDegree, include_bias=False)
    xBasisTrain = poly.fit_transform(xTrain)
    regModel.fit(X = xBasisTrain, y = yTrain)
    yPredTrain = regModel.predict(X = xBasisTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))

    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    xGridBasis = poly.fit_transform(xGrid)
    yFit = regModel.predict(X = xGridBasis)
    plt.plot(xGrid, yFit, 'C2');
  
    # Prediction on test data
    xBasisTest = poly.fit_transform(xTest)
    yPredTest = regModel.predict(X = xBasisTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('Degree = '+str(polyDegree), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
   
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(polyOrders,RMSEtrain);
plt.plot(polyOrders,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("Polynomial order", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'upper right', fontsize = 12);
f.savefig("RegDataPoly.pdf")



# %% Ridge regression models with alpha = 1
alpha = 1
regModel = linear_model.Ridge(alpha=alpha)
f = plt.figure(figsize=(15,15))
polyOrders = (1,2,3,5,7,10,15,20) # Orders of the polynomials
RMSEtrain = np.zeros(len(polyOrders))
RMSEtest = np.zeros(len(polyOrders))
for count,polyDegree in enumerate(polyOrders):
    
    # Set up the polynomial basis in the training data
    poly = PolynomialFeatures(degree=polyDegree, include_bias=False)
    XBasisTrain = poly.fit_transform(xTrain)
    regModel.fit(X = XBasisTrain, y = yTrain)
    yPredTrain = regModel.predict(X = XBasisTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))
    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    xGridBasis = poly.fit_transform(xGrid)
    yFit = regModel.predict(X = xGridBasis)
    plt.plot(xGrid, yFit, 'C2');
   
    # Prediction on test data
    xBasisTest = poly.fit_transform(xTest)
    yPredTest = regModel.predict(X = xBasisTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('Degree = '+str(polyDegree), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
    
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(polyOrders,RMSEtrain);
plt.plot(polyOrders,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("Polynomial order", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'best', fontsize = 12);
f.savefig("RegDataRidgeAlpha1.pdf")
# %% Ridge regression with alpha determined by cross-validation
alphas = np.linspace(10**(-20),1,100)
regModel = linear_model.RidgeCV(alphas = alphas, cv = 5)
f = plt.figure(figsize=(15,15))
polyOrders = (1,2,3,5,7,10,15,20) # Maximal order of polynomial
RMSEtrain = np.zeros(len(polyOrders))
RMSEtest = np.zeros(len(polyOrders))
for count,polyDegree in enumerate(polyOrders):
    
    # Set up the polynomial basis in the training data
    poly = PolynomialFeatures(degree=polyDegree, include_bias=False)
    XBasisTrain = poly.fit_transform(xTrain)
    regModel.fit(X = XBasisTrain, y = yTrain)
    yPredTrain = regModel.predict(X = XBasisTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))

    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    xGridBasis = poly.fit_transform(xGrid)
    yFit = regModel.predict(X = xGridBasis)
    plt.plot(xGrid, yFit, 'C2');
    
    # Prediction on test data
    xBasisTest = poly.fit_transform(xTest)
    yPredTest = regModel.predict(X = xBasisTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('Degree = '+str(polyDegree) + ' and alphaCV = ' + str(round(regModel.alpha_,5)), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
    
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(polyOrders,RMSEtrain);
plt.plot(polyOrders,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("Polynomial order", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'best', fontsize = 12);
f.savefig("RegDataRidgeAlphaCV.pdf")

# %% Lasso regularized regression 
regModel = linear_model.LassoLars(alpha = 0.01) # Numerical more stable than linear_model.LassoCV
f = plt.figure(figsize=(15,15))
polyOrders = (1,2,3,5,7,10,15,20) # Maximal order of polynomial
RMSEtrain = np.zeros(len(polyOrders))
RMSEtest = np.zeros(len(polyOrders))
for count,polyDegree in enumerate(polyOrders):
    # Set up the polynomial basis in the training data
    poly = PolynomialFeatures(degree=polyDegree, include_bias=False)
    XBasisTrain = poly.fit_transform(xTrain)
    regModel.fit(X = XBasisTrain, y = np.ravel(yTrain))
    yPredTrain = regModel.predict(X = XBasisTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))

    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    xGridBasis = poly.fit_transform(xGrid)
    yFit = regModel.predict(X = xGridBasis)
    plt.plot(xGrid, yFit, 'C2');
    plt.xlabel('', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Prediction on test data
    xBasisTest = poly.fit_transform(xTest)
    yPredTest = regModel.predict(X = xBasisTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('Degree = '+str(polyDegree), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
    
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(polyOrders,RMSEtrain);
plt.plot(polyOrders,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("Polynomial order", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'best', fontsize = 12);
f.savefig("RegDataLassoAlpha001.pdf")


# %% LASSO with alpha determined by 5-fold CV
regModel = linear_model.LassoLarsCV(cv = 5) # Numerical more stable than linear_model.LassoCV
f = plt.figure(figsize=(15,15))
polyOrders = (1,2,3,5,7,10,15,20) # Maximal order of polynomial
RMSEtrain = np.zeros(len(polyOrders))
RMSEtest = np.zeros(len(polyOrders))
for count,polyDegree in enumerate(polyOrders):
    # Set up the polynomial basis in the training data
    poly = PolynomialFeatures(degree=polyDegree, include_bias=False)
    XBasisTrain = poly.fit_transform(xTrain)
    regModel.fit(X = XBasisTrain, y = np.ravel(yTrain))
    yPredTrain = regModel.predict(X = XBasisTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))

    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    xGridBasis = poly.fit_transform(xGrid)
    yFit = regModel.predict(X = xGridBasis)
    plt.plot(xGrid, yFit, 'C2');
    plt.xlabel('', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Prediction on test data
    xBasisTest = poly.fit_transform(xTest)
    yPredTest = regModel.predict(X = xBasisTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('Degree = '+str(polyDegree) + ' and alphaCV = ' + str(round(regModel.alpha_,5)), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
    
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(polyOrders,RMSEtrain);
plt.plot(polyOrders,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("Polynomial order", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'best', fontsize = 12);
f.savefig("RegDataLassoAlphaCV.pdf")

# %% Regression trees
from sklearn.tree import DecisionTreeRegressor
f = plt.figure(figsize=(15,15))
treeDepths = (1,2,3,4,5,6,7,8)
RMSEtrain = np.zeros(len(treeDepths))
RMSEtest = np.zeros(len(treeDepths))
for count,treeDepth in enumerate(treeDepths):
    
    regModel = DecisionTreeRegressor(max_depth=treeDepth)
    regModel.fit(X = xTrain, y = np.ravel(yTrain))
    yPredTrain = regModel.predict(X = xTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))

    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    yFit = regModel.predict(X = xGrid)
    plt.plot(xGrid, yFit, 'C2');
    plt.xlabel('', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Prediction on test data
    yPredTest = regModel.predict(X = xTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('Tree Depth = ' + str(treeDepth), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
    
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(treeDepths,RMSEtrain);
plt.plot(treeDepths,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("Max tree depth", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'best', fontsize = 12);
f.savefig("RegDataRegTrees.pdf")

# %% Random forest
from sklearn.ensemble import RandomForestRegressor
f = plt.figure(figsize=(15,15))
max_depths = (1,2,3,4,5,6,7,8)
RMSEtrain = np.zeros(len(max_depths))
RMSEtest = np.zeros(len(max_depths))
for count,max_depth in enumerate(max_depths):
    
    regModel = RandomForestRegressor(max_depth = max_depth, n_estimators = 100)
    regModel.fit(X = xTrain, y = np.ravel(yTrain))
    yPredTrain = regModel.predict(X = xTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))

    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    yFit = regModel.predict(X = xGrid)
    plt.plot(xGrid, yFit, 'C2');
    plt.xlabel('', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Prediction on test data
    yPredTest = regModel.predict(X = xTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('Max tree depth = ' + str(max_depth), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
    
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(max_depths,RMSEtrain);
plt.plot(max_depths,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("Max tree depth", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'best', fontsize = 12);
f.savefig("RegDataRandomForest.pdf")
# %% XGBoost
import xgboost as xgb
f = plt.figure(figsize=(15,15))
max_depths = (1,2,3,4,5,6,7,8)
RMSEtrain = np.zeros(len(max_depths))
RMSEtest = np.zeros(len(max_depths))
for count,max_depth in enumerate(max_depths):
    
    regModel = xgb.XGBRegressor(objective ='reg:squarederror', max_depth = max_depth, subsample = 0.1, learning_rate = 0.3, reg_lambda = 10, gamma = .1)
    regModel.fit(X = xTrain, y = np.ravel(yTrain))
    yPredTrain = regModel.predict(xTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))

    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    yFit = regModel.predict(xGrid)
    plt.plot(xGrid, yFit, 'C2');
    plt.xlabel('', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Prediction on test data
    yPredTest = regModel.predict(xTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('Max tree depth = ' + str(max_depth), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
    
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(treeDepths,RMSEtrain);
plt.plot(treeDepths,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("Max tree depth", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'best', fontsize = 12);
f.savefig("RegDataXGBoost.pdf")

# %%k-NN
from sklearn.neighbors import KNeighborsRegressor
f = plt.figure(figsize=(15,15))
kGrid = (1,2,3,4,5,6,7,8)
RMSEtrain = np.zeros(len(kGrid))
RMSEtest = np.zeros(len(kGrid))
for count,k in enumerate(kGrid):
    
    regModel = KNeighborsRegressor(n_neighbors=k)
    regModel.fit(X = xTrain, y = np.ravel(yTrain))
    yPredTrain = regModel.predict(xTrain)
    RMSEtrain[count] = np.sqrt(mean_squared_error(yTrain, yPredTrain))

    # Plotting the fit
    plt.subplot(3,3,count+1)
    plt.scatter(xTrain, yTrain, s = 10)
    yFit = regModel.predict(xGrid)
    plt.plot(xGrid, yFit, 'C2');
    plt.xlabel('', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Prediction on test data
    yPredTest = regModel.predict(xTest)
    RMSEtest[count] = np.sqrt(mean_squared_error(yTest, yPredTest))
    plt.scatter(xTest, yTest, s = 10)
    plt.title('k = ' + str(k), fontsize=12)
    plt.legend(labels=['Training fit', 'Training data', 'Test data'], loc = 'best', fontsize = 8);
    
# Plotting the RMSE on training and test   
plt.subplot(3,3,9)
plt.plot(kGrid,RMSEtrain);
plt.plot(kGrid,RMSEtest);
plt.title('RMSE - training vs test', fontsize = 12)
plt.ylabel("RMSE", fontsize = 12);
plt.xlabel("k", fontsize = 12);
plt.legend(labels=['RMSE training', 'RMSE test'], loc = 'best', fontsize = 12);
f.savefig("RegDatakNN.pdf")
