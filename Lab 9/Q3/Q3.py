from utils import *
import numpy as np


############ QUESTION 3 ##############
class KR:
	def __init__(self, x,y,b=1):
		self.x = x
		self.y = y
		self.b = b
	
	def gaussian_kernel(self, z):
		'''
		Implement gaussian kernel
		'''
		temp = (1/np.sqrt(2*np.pi))*(np.exp(-1*z*z/2))
		return temp 	
		
	def predict(self, x_test):
		'''
		returns predicted_y_test : numpy array of size (x_train, ) 
		'''
		n = self.x.shape[0]
		pred = np.zeros((x_test.shape[0]))
		x_train = np.reshape(self.x,(self.x.shape[0],1))
		x_test = np.reshape(x_test,(x_test.shape[0],1))
		x_temp = x_train - x_test.T

		for i in range(x_test.shape[0]):
			s = np.zeros((n))
			for j in range(n):
				s[j] = self.gaussian_kernel(x_temp[j][i]/self.b)
			temp = np.sum(s)
			for j in range(n):
				s[j] = (n*s[j])/temp
			pred[i] = np.dot(s,self.y)/n

		
		return pred

		
def q3():
	#Kernel Regression
	x_train, x_test, y_train, y_test = get_dataset()
	
	obj = KR(x_train, y_train)
	
	y_predicted = obj.predict(x_test)
	
	print("Loss = " ,find_loss(y_test, y_predicted))
