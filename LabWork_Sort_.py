>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.preprocessing import MinMaxScaler
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.feature_selection import SelectKBest,chi2,RFE
>>> from pandas import read_excel
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> import openpyxl
>>> from sklearn import datasets
>>> from sklearn.manifold import TSNE
>>> import matplotlib.pyplot as plt
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.naive_bayes import MultinomialNB
>>> from sklearn.model_selection import KFold
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.pipeline import Pipeline
>>> from sklearn import model_selection

--РЕАЛЬНЫЙ Dataset--
>>> df = pd.read_excel(io=r'C:\универ\Специальные главы CS\file.xlsx', engine='openpyxl', usecols='B:P')
>>> X = df.iloc[:, 3:12].values
>>> y = df.iloc[:, 13].values
>>> X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=10)
>>> X_train.shape,X_test.shape
((36, 9), (147, 9))

--СИНТЕЗИРОВАННЫЙ Dataset--
>>> df2 = pd.read_excel(io=r'C:\универ\Специальные главы\CS\ЛР5готов\ПунктКонОкругСрЗначПризнакКласс.xlsx', engine='openpyxl', usecols='B:O')
>>> X2 = df2.iloc[:, 2:11].values
>>> y2 = df2.iloc[:, 12].values
>>> X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,train_size=0.2,random_state=10)

--Метод случайный лес--
>>> model = RandomForestClassifier()
>>> model.fit(X_train, y_train)
RandomForestClassifier()
>>> xs1=model.score(X_test, y_test)
>>> xs1
0.1292517006802721
>>> xs2=model.score(X2_test, y2_test)
>>> xs2
0.32653061224489793

--Метод ближайших соседей--
>>> model2= KNeighborsClassifier()
>>> model2.fit(X_train, y_train)
KNeighborsClassifier()
>>> xs4=model2.score(X2_test, y2_test)
>>> xs3=model2.score(X_test, y_test)
>>> xs3
0.1292517006802721
>>> xs4
0.3469387755102041

--Метод наивный байес--
>>> model3= MultinomialNB()
>>> model3.fit(X_train, y_train)
MultinomialNB()
>>> xs5=model3.score(X_test, y_test)
>>> xs6=model3.score(X2_test, y2_test)
>>> xs5
0.11564625850340136
>>> xs6
0.32653061224489793

--График по первому методу--
>>> plt.plot([0, xs1])
[<matplotlib.lines.Line2D object at 0x000002592FBA66D0>]
>>> plt.plot([0, xs2])
[<matplotlib.lines.Line2D object at 0x000002592CDC8040>]
>>> plt.show()

--График по второму методу--
>>> plt.plot([0, xs3])
[<matplotlib.lines.Line2D object at 0x000002592852AC70>]
>>> plt.plot([0, xs4])
[<matplotlib.lines.Line2D object at 0x000002592852A460>]
>>> plt.show()

--График по третьему методу--
>>> plt.plot([0, xs5])
[<matplotlib.lines.Line2D object at 0x000002592836AA30>]
>>> plt.plot([0, xs6])
[<matplotlib.lines.Line2D object at 0x000002592881BA90>]
>>> plt.show()

--Отображение методов на одном графике--
>>> plt.plot([0, xs1], color = "blue")
[<matplotlib.lines.Line2D object at 0x000002592B6152E0>]
>>> plt.plot([0, xs2], color = "blue")
[<matplotlib.lines.Line2D object at 0x000002592B6156A0>]
>>> plt.plot([0, xs3], color = "red")
[<matplotlib.lines.Line2D object at 0x000002592B615A30>]
>>> plt.plot([0, xs4], color = "red")
[<matplotlib.lines.Line2D object at 0x000002592B615DC0>]
>>> plt.plot([0, xs5], color = "green")
[<matplotlib.lines.Line2D object at 0x00000259284F7190>]
>>> plt.plot([0, xs6], color = "green")
[<matplotlib.lines.Line2D object at 0x00000259284F7520>]
>>> plt.show()

--Разница между ошибками оказалась большая. Проведем еще один эксперимент, взяв за X лучшие 3 значения(столбцы)--
>>> X = df.iloc[:, 10:12].values
>>> y = df.iloc[:, 13].values

>>> X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=10)

>>> X2 = df2.iloc[:, 9:11].values
>>> y2 = df2.iloc[:, 12].values

>>> X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,train_size=0.2,random_state=10)

>>> model.fit(X_train, y_train)
RandomForestClassifier()
>>> xs1=model.score(X_test, y_test)
>>> xs2=model.score(X2_test, y2_test)
>>> xs1
0.4557823129251701
>>> xs2
0.20408163265306123

>>> model2= KNeighborsClassifier()
>>> model2.fit(X_train, y_train)
KNeighborsClassifier()
>>> xs3=model2.score(X_test, y_test)
>>> xs4=model2.score(X2_test, y2_test)
>>> xs3
0.47619047619047616
>>> xs4
0.20408163265306123

>>> model3= MultinomialNB()
>>> model3.fit(X_train, y_train)
MultinomialNB()
>>> xs5=model3.score(X_test, y_test)
>>> xs6=model3.score(X2_test, y2_test)
>>> xs5
0.46258503401360546
>>> xs6
0.20408163265306123

--Проведем еще один эксперимент, исключив 3 наихудших значения(столбца)--
>>> df = pd.read_excel(io=r'C:\универ\Специальные главы CS\ЛР5готов\Почтиконец2.xlsx', engine='openpyxl', usecols='B:M')
>>> df2 = pd.read_excel(io=r'C:\универ\Специальные главы CS\ЛР5готов\ПунктКонОкругСрЗначПризнакКласс2.xlsx', engine='openpyxl', usecols='B:L')

>>>X = df.iloc[:, 3:9].values
>>>y = df.iloc[:, 10].values

>>> X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=10)

>>> X2 = df2.iloc[:, 2:8].values
>>> y2 = df2.iloc[:, 9].values

>>> X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,train_size=0.2,random_state=10)

>>> model = RandomForestClassifier()
>>> model.fit(X_train, y_train)
RandomForestClassifier()
>>> xs2=model.score(X2_test, y2_test)
>>> xs1=model.score(X_test, y_test)
>>> xs1
0.6802721088435374
>>> xs2
0.17006802721088435

>>> model2= KNeighborsClassifier()
>>> model2.fit(X_train, y_train)
KNeighborsClassifier()
>>> xs3=model2.score(X_test, y_test)
>>> xs4=model2.score(X2_test, y2_test)
>>> xs3
0.5986394557823129
>>> xs4
0.20408163265306123

>>> model3= MultinomialNB()
>>> model3.fit(X_train, y_train)
MultinomialNB()
>>> xs5=model3.score(X_test, y_test)
>>> xs6=model3.score(X2_test, y2_test)
>>> xs5
0.46258503401360546
>>> xs6
0.20408163265306123
