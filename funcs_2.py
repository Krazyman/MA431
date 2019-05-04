from __future__ import print_function
from datetime import datetime
from math import sqrt
from math import e
from statistics import stdev
from scipy.stats import t
from statistics import mean
import pandas as pd
import statsmodels.api as sm
from calendar import monthrange
from statsmodels.graphics.api import abline_plot
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
# from statsmodels.stats.outliers_influence import summary_table
# from summary_table import *
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#copy this class
# RainCount is the count of number of raindays per month with year
# - RainCount(date, oni, inch)
class RainCount:

    def __init__(self, date, oni, lvl1, lvl2, lvl3, temp):
        self.date = datetime.strptime(date, "%b-%Y") # date Ex: Jan-50
        self.oni = float(oni) # oceanic index
        self.lvl1 = float(lvl1) # number of raindays 0.25
        self.lvl2 = float(lvl2) # number of raindays 0.5
        self.lvl3 = float(lvl3) # number of raindays 1.0
        self.season = None
        self.temp = float(temp) # avg temp
        
    def __eq__(self, other):
        return type(other) == RainCount and \
               self.date == other.date and \
               self.oni == other.oni and \
               self.lvl1 == other.lvl1 and \
               self.lvl2 == other.lvl2 and \
               self.lvl3 == other.lvl3 and \
               self.season == other.season and \
               self.temp == other.temp

    def __repr__(self):
        return "{!r}-{!r}: {!r}: {!r} ({!r}, {!r}, {!r}, {!r})\n".format(
            self.date.month, self.date.year, self.oni, self.season, 
            self.lvl1, self.lvl2, self.lvl3, self.temp)

#copy this function
def read_date_from_file(file1):
    inFile1 = open(file1, "r")
    rains = []
    z = inFile1.readlines()
    inFile1.close()

    for i in range(len(z)):
        info = []
        info.append(z[i].split(","))
        n = [info[0][0], info[0][1], info[0][2], info[0][3], info[0][4], info[0][5].strip('"\n"')]
        new_info = RainCount(*n)
        rains.append(new_info)
    return rains

def group_oni(data):
    for item in data:
        if item.oni >= 0.5:
            item.season = "El Nino"
        elif item.oni <= -0.5:
            item.season = "La Nina"
        else:
            item.season = "Neutral"

def group_month(data):
    data.sort(key=myfunc)
    # print(data)

def myfunc(e):
    return e.date.month

# copy this
# filters the months in each season Ex. All Jan. in El for 0.25
def filt(data, lvl, season, month, temp):
    dates = []
    for item in data:
        # print(item.date.month)
        if (item.season == season and item.date.month == month and item.temp == temp) or\
         (season is None and item.date.month == month and temp is None):
            # print(item)
            if lvl == 1:
                dates.append(item)
            elif lvl == 2:
                dates.append(item)
            elif lvl == 3:
                dates.append(item)
    return dates

# gets the avg for the all the months
def go_getter(data, lvl, season):
    date_set = []
    for i in range(1, 13):
        arr = filt(data, lvl, season, i)
        if lvl == 1:
            avg = mean(stuff.lvl1 for stuff in arr)
            std = stdev(stuff.lvl1 for stuff in arr)
            df = len(arr) -1
            tvalue = t.ppf(0.95, df=df)
            value = ttest(avg, tvalue, std, len(arr))
            date_set.append((avg, value))
        elif lvl == 2:
            avg = mean(stuff.lvl2 for stuff in arr)
            std = stdev(stuff.lvl2 for stuff in arr)
            df = len(arr) -1
            tvalue = t.ppf(0.95, df=df)
            value = ttest(avg, tvalue, std, len(arr))
            date_set.append((avg, value))
        elif lvl == 3:
            avg = mean(stuff.lvl3 for stuff in arr)
            std = stdev(stuff.lvl3 for stuff in arr)
            df = len(arr) -1
            tvalue = t.ppf(0.95, df=df)
            value = ttest(avg, tvalue, std, len(arr))
            date_set.append((avg, value))
    return date_set

def stat(data):
    s = ["La Nina", "El Nino", "Neutral"]
    for i in range(1, 4): # number of lvls
        for j in range(3): # number of seasons
            x = go_getter(data, i, s[j])
            print(s[j], x)
            print()

# avg tvalue, stdev, number of elements
def ttest(avg, tvalue, std, n):
    return (avg - (tvalue*std/sqrt(n)), avg + (tvalue*std/sqrt(n)))

count = read_date_from_file("count.csv")
group_oni(count)
# print(count)
# stat(count)
# print(filt(count, 3, "El Nino", 4))
# x = go_getter(count, 3, "La Nina")
# for i in range(len(x)):
#     print(x[i][0])
# print(filt(count, 1, None, 1)) # all of jan

#-------------------------------------------------------------------------
month = filt(count, 1, None, 1, None) #copy this
# print(month)
df1 = pd.DataFrame({"Oni": [item.oni for item in month],    #copy this variable
                    "Temp": [item.temp for item in month]})
df2 = pd.DataFrame({"Rain": [item.lvl1 for item in month]})#,
    # "Non": [monthrange(item.date.year, item.date.month)[1]-item.lvl1 for item in month]})

model = sm.formula.glm("df2 ~ df1",
                       family=sm.families.Gaussian(), data=df1).fit()
print(model.summary())
print(model.bse)
print(model.fittedvalues)
# print(sm.formula.glm.summary(model))

# print(model.params)
# y = [(e**y)/(1+e**y) for y in model.mu] # probability 
# y = [y for y in model.mu] # perfect fit values
y = [stuff.lvl1 for stuff in month] # 0.25 raindays
x = [item.oni for item in month] # oni lvl for everything for the month

#--------------------------------------------------------------------------------
# plot residuals vs fitted values
# resid = model.resid_pearson
# # print(resid)
# x = [x for x in model.mu]
# y = [y for y in resid]
#--------------------------------------------------------------------------------
# X = sm.add_constant(x)
# proba = model.predict()
# cov = model.cov_params()
# gradient = (proba * (1 - proba) * X.T).T # matrix of gradients for each observation
# std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
# c = 1.96 # multiplier for confidence interval
# upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
# lower = np.maximum(0, np.minimum(1, proba - std_errors * c))

# plt.plot(x, proba)
# plt.plot(x, lower, color='g')
# plt.plot(x, upper, color='g')
# plt.show()
#--------------------------------------------------------------------------------
# fittedvalues = model.fittedvalues
# tppf = stats.t.isf(0.05/2., model.df_resid)
# tmp = wls_prediction_std(model, alpha=0.05)
# predict_se, predict_ci_low, predict_ci_upp = tmp
# predict_mean_ci = np.column_stack([
#                     model.fittedvalues - tppf * predict_mean_se,
#                     model.fittedvalues + tppf * predict_mean_se])
# predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
# predict_ci_low, predict_ci_upp = data[:, 6:8].T

# # Check we got the right things
# print(np.max(np.abs(re.fittedvalues - fittedvalues)))
# print(np.max(np.abs(iv_l - predict_ci_low)))
# print(np.max(np.abs(iv_u - predict_ci_upp)))

# plt.plot(x, y, 'o')
# plt.plot(x, fittedvalues, '-', lw=2)
# plt.plot(x, predict_ci_low, 'r--', lw=2)
# plt.plot(x, predict_ci_upp, 'r--', lw=2)
# plt.plot(x, predict_mean_ci_low, 'r--', lw=2)
# plt.plot(x, predict_mean_ci_upp, 'r--', lw=2)
#--------------------------------------------------------------------------------

# fig, ax = plt.subplots()
# # print(fig, ax)
# # ax.plot(y, x)
# ax.scatter(x, y)
# # plt.plot(x, y)
# # line_fit = sm.GLM(y, sm.add_constant(x, prepend=True)).fit()
# # print(str(line_fit))
# slope,  intercept, r_value, p_value, std_err = stats.linregress(x,y)

# abline_plot(model_results=model, ax=ax, label="{!r}x+{!r}".format(slope, intercept))

# # print(y)

# ax.set_title('Month')
# ax.set_ylabel('# of Rain Days')
# ax.set_xlabel('Oceanic Nino Index');
# plt.ylim([-6,6]) 

# # resid = model.resid_deviance.copy()
# # resid_std = stats.zscore(resid)
# # print(max(resid_std))
# # print(resid)
# plt.legend(loc="upper right")
# plt.show()