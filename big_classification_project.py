#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition'
                 '/master/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')


# In[ ]:





# In[3]:


df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PRATIO', 'B', 'LSTAT', 'MEDV']


# In[4]:


df.head()


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


from mlxtend.plotting import scatterplotmatrix


# In[7]:


cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']


# In[ ]:





# In[8]:


scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                 names=cols, alpha=0.5)
plt.tight_layout()
plt.show()


# In[9]:


from mlxtend.plotting import heatmap


# In[10]:


import numpy as np


# In[11]:


cm = np.corrcoef(df[cols].values.T)


# In[12]:


hm = heatmap(cm,
            row_names=cols,
            column_names=cols)
plt.show()


# In[13]:


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
                self.eta = eta
                self.n_iter = n_iter
        
    def fit(self, X, y):
            self.w_ = np.zeros(1 + X.shape[1])
            self.cost_ = []
        
            for i in range(self.n_iter):
                output = self.net_input(X)
                errors = (y - output)
                self.w_[1:] += self.eta * X.T.dot(errors)
                self.w_[0] += self.eta * errors.sum()
                cost = (errors**2).sum() / 2.0
                self.cost_.append(cost)
            return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return self.net_input(X)
        


# In[14]:


X = df[['RM']].values


# In[15]:


y = df[['MEDV']].values


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


sc_x = StandardScaler()


# In[18]:


sc_y = StandardScaler()


# In[19]:


X_std = sc_x.fit_transform(X)


# In[20]:


y_std = sc_y.fit_transform(y, np.newaxis).flatten()


# In[21]:


lr = LinearRegressionGD()


# In[22]:


lr.fit(X_std, y_std)


# In[23]:


plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('EPOCH')
plt.show()


# In[24]:


def line_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70,)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None
line_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()


# In[25]:


num_rooms_std = sc_x.transform(np.array([[5.0]]))


# In[26]:


price_std = lr.predict(num_rooms_std)


# In[27]:


print("Price in $1000s: %.3f" %        sc_y.inverse_transform(price_std))


# In[28]:


print('Slope: %.3f' % lr.w_[1])


# In[29]:


print('Intercept: %.3f' % lr.w_[0])


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


import seaborn as sns


# In[ ]:





# In[32]:


slr = LinearRegression()


# In[33]:


slr.fit(X, y)


# In[34]:


y_predict = slr.predict(X)


# In[35]:


print('Slope: %.3f' % slr.intercept_)


# In[36]:


sns.regplot(X, y, slr)


# In[37]:


from sklearn.linear_model import RANSACRegressor


# In[38]:


ransac = RANSACRegressor(LinearRegression(),
                        max_trials=100,
                        min_samples=50,
                        loss='absolute_loss',
                        residual_threshold=5.0,
                        random_state=0)


# In[39]:


ransac.fit(X,y)


# In[40]:


inlier_mask = ransac.inlier_mask_


# In[41]:


outlier_mask = np.logical_not(inlier_mask)


# In[42]:


line_X = np.arange(3, 10, 1)


# In[43]:


line_y_ransac = ransac.predict(line_X[:, np.newaxis])


# In[44]:


plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of roooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()


# In[45]:


print('Slope: %.3f' % ransac.estimator_.coef_[0])


# In[46]:


print('Intercept: %3f' % ransac.estimator_.intercept_)


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X = df.iloc[:, :-1].values


# In[49]:


y = df['MEDV'].values


# In[50]:


X_train, X_test, y_train, y_test, = train_test_split(
       X, y, test_size=0.3, random_state=0)


# In[51]:


slr = LinearRegression()


# In[52]:


slr.fit(X_train, y_train)


# In[53]:


y_train_pred = slr.predict(X_train)


# In[54]:


y_test_pred = slr.predict(X_test)


# In[55]:


plt.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()


# In[56]:


from sklearn.metrics import mean_squared_error


# In[57]:


print('MSE train: %.3f, test: %.3f' % (
       mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))


# In[58]:


from sklearn.metrics import r2_score


# In[59]:


print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))


# In[60]:


from sklearn.linear_model import Ridge


# In[61]:


ridge = Ridge(alpha=1.0)


# In[62]:


from sklearn.linear_model import Lasso


# In[63]:


lasso = Lasso(alpha=1.0)


# In[64]:


from sklearn.linear_model import ElasticNet


# In[65]:


elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)


# In[66]:


from sklearn.preprocessing import PolynomialFeatures


# In[67]:


X = np.array([ 258.0, 270.0, 294.0, 320.0, 342.0,
               368.0, 396.0, 446.0, 480.0, 586.0])\
             [:, np.newaxis]


# In[68]:


y = np.array([ 236.4, 234.4, 252.8, 298.6, 314.2,
               342.2, 360.8, 368.0, 391.2, 390.8])


# In[69]:


lr = LinearRegression()


# In[70]:


pr = LinearRegression()


# In[71]:


quadratic = PolynomialFeatures(degree=2)


# In[72]:


X_quad = quadratic.fit_transform(X)


# In[73]:


lr.fit(X, y)


# In[74]:


X_fit = np.arange(250, 600, 10)[:, np.newaxis]


# In[75]:


y_lin_fit = lr.predict(X_fit)


# In[76]:


pr.fit(X_quad, y)


# In[77]:


y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))


# In[78]:


plt.scatter(X, y, label='Training points')
plt.plot(X_fit, y_lin_fit,
       label='Linear fit', linestyle= '--')
plt.plot(X_fit, y_quad_fit,
         label='Quadratic fit')
plt.xlabel('Explanatory variable')
plt.ylabel('Predicted or known target values')
plt.tight_layout()
plt.show()


# In[79]:


y_lin_pred =  lr.predict(X)


# In[80]:


y_quad_pred = pr.predict(X_quad)


# In[81]:


print('Training MSE linear: %.3f, quadratic: %.3f' % (
      mean_squared_error(y, y_lin_pred),
      mean_squared_error(y, y_quad_pred)))


# In[82]:


print('Training R^2 linear: %.3f, quadtratic: %.3f' %(
       r2_score(y, y_lin_pred),
       r2_score(y, y_quad_pred)))


# In[83]:


X = df[['LSTAT']].values


# In[84]:


y = df[['MEDV']].values


# In[85]:


regr = LinearRegression()


# In[86]:


quadratic = PolynomialFeatures(degree=2)


# In[87]:


cubic = PolynomialFeatures(degree=3)


# In[88]:


X_quad = quadratic.fit_transform(X)


# In[89]:


X_cubic = cubic.fit_transform(X)


# In[90]:


X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]


# In[91]:


regr = regr.fit(X, y)


# In[92]:


y_lin_fit = regr.predict(X_fit)


# In[93]:


linear_r2 = r2_score(y, regr.predict(X))


# In[94]:


regr = regr.fit(X_quad, y)


# In[95]:


y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))


# In[96]:


quadratic_r2 = r2_score(y, regr.predict(X_quad))


# In[97]:


regr = regr.fit(X_cubic, y)


# In[98]:


y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))


# In[99]:


cubic_r2 = r2_score(y, regr.predict(X_cubic))


# In[100]:


plt.scatter(X, y, label='Training points', color='lightgray')
plt.plot(X_fit, y_lin_fit,
         label='Linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue',
         lw=2,
         linestyle=':')
plt.plot(X_fit, y_quad_fit,
         label='Quadratic (d=2), $R^2%.2f$' % quadratic_r2,
         color='red',
         lw=2,
         linestyle='--')
plt.plot(X_fit, y_cubic_fit,
         label='Cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green',
         lw=2,
         linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')
plt.show()


# In[101]:


X_log = np.log(X)


# In[102]:


y_sqrt = np.sqrt(y)


# In[103]:


X_fit = np.arange(X_log.min()-1,
                  X_log.max()+1, 1)[:, np.newaxis]


# In[104]:


regr =  regr.fit(X_log, y_sqrt)


# In[105]:


y_lin_fit = regr.predict(X_fit)


# In[106]:


linear_r2 = r2_score(y_sqrt, regr.predict(X_log))


# In[107]:


plt.scatter(X_log, y_sqrt,
            label='Training points',
            color='lightgray')
plt.plot(X_fit, y_lin_fit,
         label='Linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue',
         lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# In[108]:


from sklearn.tree import DecisionTreeRegressor


# In[109]:


X = df[['LSTAT']].values


# In[110]:


y = df['MEDV'].values


# In[112]:


tree = DecisionTreeRegressor(max_depth=3)


# In[113]:


tree.fit(X, y)


# In[114]:





# In[122]:


sort_idx = X.flatten().argsort()
sns.regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()


# In[123]:


X = df.iloc[:, :-1].values


# In[124]:


y = df['MEDV'].values


# In[125]:


X_train, X_test, y_train , y_test =       train_test_split(X, y,
                        test_size=0.4,
                        random_state=1)


# In[126]:


from sklearn.ensemble import RandomForestRegressor


# In[128]:


forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)


# In[132]:


print('MSE train: %.3f, test: %.3f' % (
      mean_squared_error(y_train, y_train_pred),
      mean_squared_error(y_test, y_test_pred)))


# In[133]:


print('R^2 train: %.3f, test: %.3f' % (
       r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))


# In[136]:


plt.scatter(y_train_pred,
            y_train_pred - y_train,
            c='steelblue',
            edgecolor='white',
            marker='o',
            s=35,
            alpha=0.9)
plt.scatter(y_test_pred,
            y_test_pred - y_test,
            c='limegreen',
            marker='s',
            s=35,
            alpha=0.9,
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.tight_layout()
plt.show()


# In[ ]:




