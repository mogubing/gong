# -*- coding: utf-8 -*-
_author_ = 'huihui.gong'
_date_ = '2020/2/4'
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn import metrics
## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
digits=datasets.load_digits()
data=digits.data
# print(data.shape)
print(data[0])
# print(digits.images[0])
# print(digits.target[0])
plt.imshow(digits.images[0])
plt.show()
x,y=digits.data,digits.target
# print(x,y)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)
ss=StandardScaler()
train_x=ss.fit_transform(train_x,train_y)
test_x=ss.transform(test_x)
print(train_x.shape,train_y.shape)
# 使用交叉验证(10交叉验证)得到最佳K值。
# accScoreMean=[]
    # for i in range(1,10):
    #     knn=KNeighborsClassifier(n_neighbors=i)
    #     knnclf=knn.fit(train_x,train_y)
    #     predict_y=knnclf.predict(test_x)
    #     cross_score = cross_val_score(knnclf, train_x, train_y, scoring='accuracy', cv=10)
    #     acc_score=np.mean(cross_score)
    #     accScoreMean.append(acc_score)
    #     # print(accScoreMean)
    # plt.figure(figsize=(6,8))
    # plt.plot(range(1,10),accScoreMean,marker='o',label='准确率',c='g')
    # plt.xlabel("n_neighbors")
    # plt.ylabel("准确率")
    # plt.title('准确率与n_neighbors的关系')
    # plt.legend()
    # plt.show()
# 由画图结果可知，当n_neighbors为3时，准确率最高。但是还要结合混合矩阵的值。
knclf=KNeighborsClassifier(n_neighbors=3)
knclf.fit(train_x,train_y)
predict_y1=knclf.predict(test_x)
preds=knclf.predict_proba(test_x)
score=knclf.score(test_x,test_y)
precision_score=metrics.precision_score(test_y,predict_y1,average='weighted')
recall_score=metrics.recall_score(test_y,predict_y1,average='weighted')
f1_score=metrics.f1_score(test_y,predict_y1,average='weighted')
print("准确率为:%0.4f"%(score))
print("精确率为:%0.4f"%(precision_score))
print("召回率为:%0.4f"%(recall_score))
print("f1_score为:%0.4f"%(f1_score))
# 综上，各个指标皆理想。再画auc曲线。
# auc值：即roc曲线下方的面积
print(test_y)
# 将test_y二值化
test_y1=label_binarize(test_y,classes=[0,1,2,3,4,5,6,7,8,9])
print(test_y1)
print(predict_y1)
print(preds)
# 混淆矩阵怎么看？
print("混淆矩阵：",metrics.confusion_matrix(test_y,predict_y1))
# 这样简单地求roc曲线，只针对二分类问题，目前是10分类
# 可以将横轴纵轴改成:tpr(tp/tp+fn),fpr(fp/tn+fp)true,false,positive,nagitive
# 计算某一类的roc
tpr=dict()
fpr=dict()
roc_auc=dict()
for i in range(0,10):
    fpr[i],tpr[i],_=metrics.roc_curve(test_y1[:,i],preds[:,i])
    roc_auc[i]=metrics.auc(fpr[i],tpr[i])
print(tpr)
# 画图roc曲线(特征9)
plt.figure(figsize=(8,6))
plt.plot(fpr[9],tpr[9],marker='o',c='g',label='roc curve(area=%0.4f)'%(roc_auc[9]))
plt.plot([0,1],[0,1],c='gray',linestyle='--')
plt.xlim([0,1])
plt.xlim([0,1.05])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc curve')
plt.legend()
plt.show()








