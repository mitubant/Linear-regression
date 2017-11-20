import pandas as pd, numpy as np
import scipy
from PIL import Image
import os
import time
import timeit
from sympy import *



#---create data using random
def CreateData():	
	X_array = np.random.rand(100)
	
	#---target a = 10000, target b = 4
	Y_array = 10000 * X_array + 4
	return X_array, Y_array

#---compute Mean squared error
def compute_error(a, b,X_array, Y_array):

    totalError = 0

    x = X_array
    y = Y_array
    totalError = (y-a*x-b)**2
    totalError = np.sum(totalError,axis=0)

    return totalError/float(len(X_array))

#---Gradient function
def Gradient(X_array, Y_array, a_now, b_now, LR):
	a_gradient = 0
	b_gradient = 0

	for i in range(len(X_array)):

		N = float(len(X_array))

		x = X_array[i]
		y = Y_array[i]

		#---calculate Partial derivative for a, b 
		b_gradient += -(2/N) * (y-((a_now * x) + b_now))  
		a_gradient += -(2/N) * (y-((a_now * x) + b_now)) * x
		
	#---update the value of a and b	
	new_b = b_now - (LR * b_gradient)	
	new_a = a_now - (LR * a_gradient)



	return new_a, new_b


#---train data
def train(X_array, Y_array, LR):
	
	a = 0
	b = 0


	for i in range(1000000):

		a,b = Gradient(X_array, Y_array, a, b, LR)
		error = compute_error(a, b, X_array, Y_array)
		
		if i%100 == 0:
			print('iter {0} : error={1} : a={2} : b={3}'.format(i, error, a, b))

		if error <= 1.0e-15:
			print('early ended... iter {0} : error={1} : a={2} : b={3}'.format(i, error, a, b))
			break


#---main function
def main():
	
	X_array, Y_array = CreateData()

	print(X_array)
	print(Y_array)

	train(X_array, Y_array, 0.01)


if __name__ == '__main__':
	main()

