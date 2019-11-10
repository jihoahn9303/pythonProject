import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import tkinter as tk 
from sklearn import linear_model

class Regression(object):
    def __init__(self, df, column_name_list):
        self.df = df                              # 데이터프레임
        self.column_name_list = column_name_list  # 컬럼 명 리스트
        self.epochs = 100000                      # 에폭 수
        self.min_grad = 0.0001                    # 학습 종료 기준 미분 값
        self.learning_rate = 0.1                 # 학습률

    def one_column_regression(self):
        # 기울기와 절편
        m = 0.0
        c = 0.0
        x = np.asarray(self.df[self.column_name_list[0]])
        y = np.asarray(self.df[self.column_name_list[-1]])
        n, count = x.size, 0
        
        # linear regression의 속도를 향상시키기 위해, 각 컬럼 데이터에 대해 정규화
        # 1. 각 컬럼에 대해 평균 값 계산
        x_mean= np.mean(x, axis = None)
        # 2. 각 컬럼에 대해 (최대값 - 최소값)을 계산
        x_differ = np.max(x) - np.min(x)
        # 3. 각 컬럼의 데이터에 대해 -1 ~ 1의 값을 가지도록 정규화
        x = (x - x_mean) / x_differ

        for epoch in range(self.epochs):
            c_grad = np.zeros(n)
            m_grad = np.zeros(n)

            # 오차함수에 대한 편미분 값 구하기
            y_pred = m*x + c*np.ones(n)
            m_grad = np.dot(np.subtract(y_pred, y), x) * 2 / n
            c_grad = np.sum(np.subtract(y_pred, y)) * 2 / n

            # y-intercept와 기울기 값 갱신
            m -= self.learning_rate  * m_grad
            c -= self.learning_rate  * c_grad

            count += 1
            if (abs(m_grad) < self.min_grad) and (abs(c_grad) < self.min_grad):
                break

        # 알고리즘으로 추정한 parameter 값 출력
        print("After %d iteration : m = %f, c = %f" % (count, m, c))
        # sklearn 패키지를 활용한 회귀 분석
        src = np.array([x]).transpose()
        obj = np.array(y)
        intercept, coefficient = self.MVLR_library(src, obj)  # sklearn 패키지가 추정한 절편 및 파라미터 값 리턴
        coefficient_list = [m]  # 직접 구현한 알고리즘으로 추정한 파라미터 리스트
        error_list  = [c - intercept]  # 직접 추정 값 - 패키지 추정 값
        for idx in range(coefficient.size):
            error_list.append(coefficient_list[idx] - coefficient[idx])
        print(error_list)  # 오차 리스트 출력

    def two_column_regression(self):
        # parameter 선언
        m1, m2, c = 0.0, 0.0, 0.0
        # 컬럼에 대응하는 데이터 추출
        x1, x2 = np.asarray(self.df[self.column_name_list[0]]), np.asarray(self.df[self.column_name_list[1]])
        y = np.asarray(self.df[self.column_name_list[-1]])
        n, count = x1.size, 0

        # linear regression의 속도를 향상시키기 위해, 각 컬럼 데이터에 대해 정규화
        # 1. 각 컬럼에 대해 평균 값 계산
        x1_mean, x2_mean= np.mean(x1, axis = None), np.mean(x2, axis = None)
        # 2. 각 컬럼에 대해 (최대값 - 최소값)을 계산
        x1_differ, x2_differ = np.max(x1)-np.min(x1), np.max(x2)-np.min(x2)
        # 3. 각 컬럼의 데이터에 대해 -1 ~ 1의 값을 가지도록 정규화
        x1_norm, x2_norm = (x1-x1_mean) / x1_differ, (x2-x2_mean) / x2_differ

        for epoch in range(self.epochs):
            # 미분 값을 저장할 배열
            c_grad, m1_grad, m2_grad = np.zeros(n), np.zeros(n), np.zeros(n)
            # 각 parameter 마다 편미분 값 계산
            y_pred = m1*x1_norm + m2*x2_norm + c*np.ones(n)
            m1_grad = np.dot(np.subtract(y_pred, y), x1_norm) * 2 / n
            m2_grad = np.dot(np.subtract(y_pred, y), x2_norm) * 2 / n
            c_grad = np.sum(np.subtract(y_pred, y)) * 2 / n
            # parameter 갱신
            m1 -= (self.learning_rate * m1_grad)
            m2 -= (self.learning_rate * m2_grad)
            c -= (self.learning_rate * c_grad)
            count += 1
            if (abs(m1_grad) < self.min_grad) and (abs(m2_grad) < self.min_grad) and (abs(c_grad) < self.min_grad):
                break
        
        print("After %d iteration: m1= %f, m2 = %f, c = %f" % (count, m1, m2, c))
        # sklearn 패키지를 활용한 회귀 분석
        src = np.array([x1_norm, x2_norm]).transpose()
        obj = np.array(y)
        intercept, coefficient = self.MVLR_library(src, obj)  # sklearn 패키지가 추정한 절편 및 파라미터 값 리턴
        coefficient_list = [m1, m2]  # 직접 구현한 알고리즘으로 추정한 파라미터 리스트
        error_list  = [c - intercept]  # 직접 추정 값 - 패키지 추정 값
        for idx in range(coefficient.size):
            error_list.append(coefficient_list[idx] - coefficient[idx])
        print(error_list)  # 오차 리스트 출력


        # 3D 시각화
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_axis = x1_norm    # x축에 입력될 데이터
        y_axis = x2_norm    # y축에 입력될 데이터
        z_axis = y
        X, Y = np.meshgrid(x_axis, y_axis)     # 행렬을 좌표로 변환
        Z = m1 * X + m2 * Y + c       # 선형 회귀를 통해 얻은 방정식으로 z축에 pointing
        
        # 그래프 그리기
        ax.plot(x_axis, y_axis, z_axis, linestyle="none", marker="^", mfc="none", markeredgecolor="red")   
        ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.4, cmap=cm.jet)
        # 라벨링
        ax.set_xlabel(self.column_name_list[0])
        ax.set_ylabel(self.column_name_list[1])
        ax.set_zlabel(self.column_name_list[-1])
        plt.show()

    def three_column_regression(self):
        # constant 선언
        m1, m2, m3, c = 0.0, 0.0, 0.0, 0.0
        # 각 컬럼에 대응하는 데이터 추출
        x1, x2, x3 = self.df[self.column_name_list[0]], self.df[self.column_name_list[1]], self.df[self.column_name_list[2]]
        y = self.df[self.column_name_list[-1]]
        n, count = x1.size, 0
        # linear regression의 속도를 향상시키기 위해, 각 컬럼 데이터에 대해 정규화
        x1_mean, x2_mean, x3_mean = np.mean(x1, axis = None), np.mean(x2, axis = None), np.mean(x3, axis = None)
        x1_differ, x2_differ, x3_differ = np.max(x1)-np.min(x1), np.max(x2)-np.min(x2), np.max(x3)-np.min(x3)
        x1_norm, x2_norm, x3_norm = (x1-x1_mean) / x1_differ, (x2-x2_mean) / x2_differ, (x3-x3_mean) / x3_differ

        # 에폭마다 경사하강법을 통해 선형 회귀 실시
        for epoch in range(self.epochs):
            c_grad, m1_grad, m2_grad, m3_grad = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            
            # 각 feature에 대해 편미분 실시
            y_pred = m1*x1_norm + m2*x2_norm + m3*x3_norm + c*np.ones(n)
            m1_grad = np.dot(np.subtract(y_pred, y), x1_norm) * 2 / n
            m2_grad = np.dot(np.subtract(y_pred, y), x2_norm) * 2 / n
            m3_grad = np.dot(np.subtract(y_pred, y), x3_norm) * 2 / n
            c_grad = np.sum(np.subtract(y_pred, y)) * 2 / n

            # constant 갱신
            m1 -= (self.learning_rate * m1_grad)
            m2 -= (self.learning_rate * m2_grad)
            m3 -= (self.learning_rate * m3_grad)
            c -= (self.learning_rate * c_grad)
            count += 1
            if (abs(m1_grad) < self.min_grad and abs(m2_grad) < self.min_grad
                    and abs(m3_grad) < self.min_grad and abs(c_grad) < self.min_grad):
                break
 
        print("After %d iteration: m1= %f, m2 = %f, m3 = %f, c = %f" % (count, m1, m2, m3, c))
        # sklearn 패키지를 활용한 회귀 분석
        src = np.array([x1_norm, x2_norm, x3_norm]).transpose()
        obj = np.array(y)
        intercept, coefficient = self.MVLR_library(src, obj)  # sklearn 패키지가 추정한 절편 및 파라미터 값 리턴
        coefficient_list = [m1, m2, m3]  # 직접 구현한 알고리즘으로 추정한 파라미터 리스트
        error_list  = [c - intercept]  # 직접 추정 값 - 패키지 추정 값
        for idx in range(coefficient.size):
            error_list.append(coefficient_list[idx] - coefficient[idx])
        print(error_list)  # 오차 리스트 출력

    def four_column_regression(self):
        # constant 선언
        m1, m2, m3, m4, c = 0.0, 0.0, 0.0, 0.0, 0.0
        # 각 컬럼에 대응하는 데이터 추출
        x1, x2, x3, x4 = self.df[self.column_name_list[0]], self.df[self.column_name_list[1]], self.df[self.column_name_list[2]], self.df[self.column_name_list[3]]
        y = self.df[self.column_name_list[-1]]
        n, count = x1.size, 0
        # linear regression의 속도를 향상시키기 위해, 각 컬럼 데이터에 대해 정규화
        x1_mean, x2_mean, x3_mean, x4_mean = np.mean(x1), np.mean(x2), np.mean(x3), np.mean(x4)
        x1_differ, x2_differ, x3_differ, x4_differ = np.max(x1)-np.min(x1), np.max(x2)-np.min(x2), np.max(x3)-np.min(x3), np.max(x4)-np.min(x4)
        x1_norm, x2_norm, x3_norm, x4_norm = (x1-x1_mean) / x1_differ, (x2-x2_mean) / x2_differ, (x3-x3_mean) / x3_differ, (x4-x4_mean) / x4_differ

        # 에폭마다 경사하강법을 통해 선형 회귀 실시
        for epoch in range(self.epochs):
            c_grad, m1_grad, m2_grad, m3_grad, m4_grad = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            
            y_pred = m1*x1_norm + m2*x2_norm + m3*x3_norm + m4*x4_norm + c*np.ones(n)
            m1_grad = np.dot(np.subtract(y_pred, y), x1_norm) * 2 / n
            m2_grad = np.dot(np.subtract(y_pred, y), x2_norm) * 2 / n
            m3_grad = np.dot(np.subtract(y_pred, y), x3_norm) * 2 / n
            m4_grad = np.dot(np.subtract(y_pred, y), x4_norm) * 2 / n
            c_grad = np.sum(np.subtract(y_pred, y)) * 2 / n

            m1 -= (self.learning_rate * m1_grad)
            m2 -= (self.learning_rate * m2_grad)
            m3 -= (self.learning_rate * m3_grad)
            m4 -= (self.learning_rate * m4_grad)
            c -= (self.learning_rate * c_grad)
            count += 1
            if (abs(m1_grad) < self.min_grad and abs(m2_grad) < self.min_grad
                    and abs(m3_grad) < self.min_grad and abs(m4_grad) < self.min_grad and abs(c_grad) < self.min_grad):
                break
        
        print("After %d iteration: m1= %f, m2 = %f, m3 = %f, m4 = %f, c = %f" % (count, m1, m2, m3, m4, c))       
        # sklearn 패키지를 활용한 회귀 분석
        src = np.array([x1_norm, x2_norm, x3_norm, x4_norm]).transpose()
        obj = np.array(y)
        intercept, coefficient = self.MVLR_library(src, obj)  # sklearn 패키지가 추정한 절편 및 파라미터 값 리턴
        coefficient_list = [m1, m2, m3, m4]  # 직접 구현한 알고리즘으로 추정한 파라미터 리스트
        error_list  = [c - intercept]  # 직접 추정 값 - 패키지 추정 값
        for idx in range(coefficient.size):
            error_list.append(coefficient_list[idx] - coefficient[idx])
        print(error_list)  # 오차 리스트 출력

    def all_column_regression(self):
        # constant 선언
        m1, m2, m3, m4, m5, c = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # 각 컬럼에 대응하는 데이터 추출
        x1, x2, x3, x4 = self.df[self.column_name_list[0]], self.df[self.column_name_list[1]], self.df[self.column_name_list[2]], self.df[self.column_name_list[3]]
        x5 = self.df[self.column_name_list[4]]
        y = self.df[self.column_name_list[-1]]
        n, count = x1.size, 0
        # linear regression의 속도를 향상시키기 위해, 각 컬럼 데이터에 대해 정규화
        x1_mean, x2_mean, x3_mean, x4_mean, x5_mean = np.mean(x1), np.mean(x2), np.mean(x3), np.mean(x4), np.mean(x5)
        x1_differ, x2_differ, x3_differ, x4_differ = np.max(x1)-np.min(x1), np.max(x2)-np.min(x2), np.max(x3)-np.min(x3), np.max(x4)-np.min(x4)
        x5_differ = np.max(x5)-np.min(x5)
        x1_norm, x2_norm, x3_norm, x4_norm = (x1-x1_mean) / x1_differ, (x2-x2_mean) / x2_differ, (x3-x3_mean) / x3_differ, (x4-x4_mean) / x4_differ
        x5_norm = (x5-x5_mean) / x5_differ

        # 에폭마다 경사하강법을 통해 선형 회귀 실시
        for epoch in range(self.epochs):
            c_grad, m1_grad, m2_grad, m3_grad, m4_grad, m5_grad = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            
            y_pred = m1*x1_norm + m2*x2_norm + m3*x3_norm + m4*x4_norm + m5*x5_norm + c*np.ones(n)
            m1_grad = np.dot(np.subtract(y_pred, y), x1_norm) * 2 / n
            m2_grad = np.dot(np.subtract(y_pred, y), x2_norm) * 2 / n
            m3_grad = np.dot(np.subtract(y_pred, y), x3_norm) * 2 / n
            m4_grad = np.dot(np.subtract(y_pred, y), x4_norm) * 2 / n
            m5_grad = np.dot(np.subtract(y_pred, y), x5_norm) * 2 / n
            c_grad = np.sum(np.subtract(y_pred, y)) * 2 / n

            m1 -= (self.learning_rate * m1_grad)
            m2 -= (self.learning_rate * m2_grad)
            m3 -= (self.learning_rate * m3_grad)
            m4 -= (self.learning_rate * m4_grad)
            m5 -= (self.learning_rate * m5_grad)
            c -= (self.learning_rate * c_grad)
            count += 1
            if (abs(m1_grad) < self.min_grad and abs(m2_grad) < self.min_grad
                    and abs(m3_grad) < self.min_grad and abs(m4_grad) < self.min_grad 
                        and abs(m5_grad) < self.min_grad and abs(c_grad) < self.min_grad):
                break

        print("After %d iteration: m1= %f, m2 = %f, m3 = %f, m4 = %f, m5 = %f, c = %f" % (count, m1, m2, m3, m4, m5, c))   
        # sklearn 패키지를 활용한 회귀 분석
        src = np.array([x1_norm, x2_norm, x3_norm, x4_norm, x5_norm]).transpose()
        obj = np.array(y)
        intercept, coefficient = self.MVLR_library(src, obj)  # sklearn 패키지가 추정한 절편 및 파라미터 값 리턴
        coefficient_list = [m1, m2, m3, m4, m5]  # 직접 구현한 알고리즘으로 추정한 파라미터 리스트
        error_list  = [c - intercept]  # 직접 추정 값 - 패키지 추정 값
        for idx in range(coefficient.size):
            error_list.append(coefficient_list[idx] - coefficient[idx])
        print(error_list)  # 오차 리스트 출력

    # sklearn 패키지를 활용한 회귀 분석 함수
    def MVLR_library(self, X, Y):
        """
        parameter : 1) X : 독립변수
                    2) Y : 종속변수(score 컬럼 데이터)
        """
        regr = linear_model.LinearRegression(fit_intercept = True)
        regr.fit(X, Y)
        return regr.intercept_, regr.coef_
      
if __name__ == "__main__":
    data = pd.read_excel('D:/Datamining/dbscore.xlsx')
    column_name_list = [x for x in input("Input column names. : ").split(', ')]
    column_name_list.append('score')
    df = pd.DataFrame(data, columns = column_name_list)
    # 클래스 생성자 호출
    regress = Regression(df, column_name_list)
    # 컬럼 갯수에 따라 서로 다른 선형회귀 함수 호출
    if len(column_name_list) == 2:
        regress.one_column_regression()
    elif len(column_name_list) == 3:
        regress.two_column_regression()
    elif len(column_name_list) == 4:
        regress.three_column_regression() 
    elif len(column_name_list) == 5:
        regress.four_column_regression() 
    elif len(column_name_list) == 6:
        regress.all_column_regression()     
