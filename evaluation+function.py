
# coding: utf-8

# In[2]:

#The normal imports
import numpy as np
from numpy.random import randn
import pandas as pd

#import the stats libarary from numpy
from scipy import stats

# These are the plotting module and libraries we'll use:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np  
import scipy as sp  
from scipy.stats import norm  
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn import linear_model  
get_ipython().magic(u'matplotlib inline')


# In[3]:

def newtest(data):
    #creat some empty list for further use
    even_list=[]
    odd_list=[]
    avg = []#average list
    std = []#standard deviation list
    sbr = []#output 
    cv = []#output
    number_of_test = len(data)
    #split the data from the array into two list because they are all paired
    for i in range(0,number_of_test):
        if i%2:
            even_list.append(data[i])
        else:
            odd_list.append(data[i])
    #calculate the average and the standard deviation of the paired data
    half_number = int(number_of_test/2)
    for i in range(0,half_number):
        avg_number = (even_list[i][1]+odd_list[i][1])/2
        avg.append(avg_number)#avg = sum of xi/i
        std_number = (even_list[i][1])**2+(odd_list[i][1])**2-((even_list[i][1]+odd_list[i][1])**2)/2
        std.append(std_number)#std = sum of xi^2- (sum of xi)^2/i
    #calculate cv
    for i in range(0,half_number):
        cv.append(std[i]/avg[i])
    for i in range(0,number_of_test):
        sbr.append(data[i][1]/data[0][1])
    return cv,sbr
        


# In[4]:

#read csv file 
df = pd.read_csv("/Users/zhshpe_40/Downloads/test_01.csv",error_bad_lines=False)
sample_concentration=df['sample_concentration_1']
ample = df['current_amp']

#turn two data frame into list
s=sample_concentration.tolist()
a=ample.tolist()

#create a new 2d list for the newtest function
data=[]
for i in range(len(ample)):
    new_data=[sample_concentration[i],ample[i]]
    data.append(new_data)
cv,std = newtest(data)
#create a list for the result whether cv is less than 10
true_false=[]
for i in cv:
    if i <10:
        true_false.append([i,'True'])
    elif i>10:
        true_false.append([i,'False'])
print(true_false)
#Since data are all paired, split the data into two part for calculating the average 
x1=[]
y1=[]
x2=[]
y2=[]
k=0
for i in range(len(ample)):
    if k%2 ==0:
        x1.append(s[i])
        y1.append(a[i])
    elif k%2 !=0:
        x2.append(s[i])
        y2.append(a[i])
    k+=1
#create a new list of average of x and y for plotting and finding the evaluation function
avg_x=[]
avg_y=[]
for i in range(len(x1)):
    x_avg=(x1[i]+x2[i])/2
    y_avg=(y1[i]+y2[i])/2
    avg_x.append(x_avg)
    avg_y.append(y_avg)
plt.plot(avg_x,avg_y,'d')
plt.figure()

''''' 数据 '''  
x=np.zeros(len(avg_x))
y=np.zeros(len(avg_y))
for i in range(len(avg_x)):
    x[i]=avg_x[i]
for i in range(len(avg_y)):
    y[i]=avg_y[i]

''''' 均方误差根 '''  
def rmse(y_test, y):  
    return sp.sqrt(sp.mean((y_test - y) ** 2))  
  
''''' 与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测.这个版本的实现是参考scikit-learn官网文档  '''  
def R2(y_test, y_true):  
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()  
  
  
''''' 这是Conway&White《机器学习使用案例解析》里的版本 '''  
def R22(y_test, y_true):  
    y_mean = np.array(y_true)  
    y_mean[:] = y_mean.mean()  
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)  
  
  
plt.scatter(x, y, s=5)  
degree = [1,2,30]  
y_test = []  
y_test = np.array(y_test)  
  
  
for d in degree:  
    clf = Pipeline([('poly', PolynomialFeatures(degree=d)),  
                    ('linear', LinearRegression(fit_intercept=False))])  
    clf.fit(x[:, np.newaxis], y)  
    y_test = clf.predict(x[:, np.newaxis])  
  
    print(clf.named_steps['linear'].coef_)  
    print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f' %  
      (rmse(y_test, y),  
       R2(y_test, y),  
       R22(y_test, y),  
       clf.score(x[:, np.newaxis], y)))
    if d ==1:
        print('y=%.2f*x+%.2f'%
          (clf.named_steps['linear'].coef_[1],
            clf.named_steps['linear'].coef_[0]))
    elif d==2:
        print('y=%.2f*x^2+%.2fx+%.2f'%
             (clf.named_steps['linear'].coef_[2],
            clf.named_steps['linear'].coef_[1],
             clf.named_steps['linear'].coef_[0]))
    plt.plot(x, y_test, linewidth=2)  
      
plt.grid()  
plt.legend(['1','2','30'], loc='upper left')  
plt.show() 

