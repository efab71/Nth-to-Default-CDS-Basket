''' The project's aim is to price a kth-to default basket consisting of 
5 entities.  For the project the entities chosen were: Barclays, BP, 
Exxon, Pfizer and HSBC.

A kth-to default basket swap is an exotic product in which one of the 
two counterparties (the buyer) buys protection against the default of 
one of the entities in the portfolio, whereas the second counterparty 
(the seller) provides protection  in the form of a payment to the buyer 
when a default  occurs.  
The payment time depends on the type of product: in the first-to-default
 the payment by the seller happens when the first reference entity 
defaults. 
After the payment the contract terminates and no more premium payments 
are due.

In the second-to-default, the default payment, as described above, 
happens when the second entity defaults.  This means that the buyer is 
not covered by any payment by the seller after the first default, but 
will have to wait until the second default to receive a default payment.  
However, after the first default the amount of the premium paid is 
reduced to take into account the fact that the derivative is not giving 
protection on all entities.  

In the third-to-default the mechanism described above is the same except
that the default payment happens after the third entity in the basket 
has defaulted, for the fourth-to-default this happens on the fourth 
default, and so on for the kth-to default.
As a general rule the premium of the kth-1-to default should be more 
expensive than the premium of the kth-to default, as the probability of 
k assets to default is greater than k-1 assets.
'''

''' The Monte Carlo simulation implemented here uses a Gaussian
Copula.
In the simulations we will not apply a closed-form formula for the 
copula, but correlation is imposed between the five elements 
of the basket, which will be equivalent to a factorisation of the 
copula into as set of linear equations.
  
To be more specific, a Cholesky factorisation of the correlation matrix 
is found and the correlation is imposed by multiplying the Cholesky 
factorisation matrix A to the five random vectors Z. '''

''' Some of the details of the projects have been hard-coded, however 
other versions of the project will enable the user to enter the details 
of the problem directly''' 

import scipy.stats

import heapq
import random
from math import exp, sqrt, log
from random import gauss, seed

seed(1)
I = 10000 # Number of simulations
recovery_rate = 0.40 

# The cholesky decomposition of the correlation matrix between the five 
# companies is hardcoded below. The decomposition used a correlation 
# matrix of weekly average returns from 28 December 2007 to 12 December 
# 2014 sourced from yahoo.com/finance 
cholesky11=1
cholesky21=-0.000676
cholesky22=1
cholesky31=0.179108
cholesky32=0.01259
cholesky33=0.983749
cholesky41=0.179108
cholesky42=-0.054535
cholesky43=0.026661
cholesky44=0.981955
cholesky51=-0.001397
cholesky52=0.498101
cholesky53=-0.021631
cholesky54=-0.0799561
cholesky55=0.8631189


# The survival probabilities (lambdas) of the five companies are 
# hard-coded below; the vectors provide lambdas for year 1 to 5 
lambda1=[-0.003274633, -0.010656053, -0.024385733, -0.049168606, -0.079780398]
lambda2=[-0.001337439, -0.003504656, -0.006932553, -0.01616506, -0.026466319]
lambda3=[-0.00416133, -0.010513795, -0.0255279, -0.049398464, -0.093690813]
lambda4=[-0.003507176, -0.007661234, -0.011970452, -0.020699409, -0.030802453]
lambda5=[-0.004378733, -0.015649062, -0.031267938, -0.076530438, -0.154997771]


# Discount factors to present-value future values for years 1 to 5 are 
# hard-coded below
discount_factors = [0.61, 0.35, 0.19, 0.10, 0.05]


#  The steps of the calculations are as follows:

#  (1) we produce independent normal random values and populate 
#  random_number vectors (below)
random_numbers1=[]
random_numbers2=[]
random_numbers3=[]
random_numbers4=[]
random_numbers5=[]

# (2) the independent random values are then correlated using the
# Cholesky decomposition matrix, the values are populated into the 
# correlated_random_numbers vectors (below)
correlated_random_numbers1=[]
correlated_random_numbers2=[]
correlated_random_numbers3=[]
correlated_random_numbers4=[]
correlated_random_numbers5=[]

# (3) Uniform random variables are populated in the uniform_vectors (below)
uniform_vector1=[]
uniform_vector2=[]
uniform_vector3=[]
uniform_vector4=[]
uniform_vector5=[]



for i in range(I):
	
	z1=gauss(0.0, 1.0)
	x1=z1
	u1=scipy.stats.norm.cdf(x1,0,1)
	
	z2=gauss(0.0, 1.0)
	x2=cholesky21*z1+cholesky22*z2
	u2=scipy.stats.norm.cdf(x2,0,1)
	
	z3=gauss(0.0, 1.0)
	x3=cholesky31*z1+cholesky32*z2+cholesky33*z3
	u3=scipy.stats.norm.cdf(x3,0,1)
		
	z4=gauss(0.0, 1.0)
	x4=cholesky41*z1 + cholesky42*z2 +cholesky43* z3 + cholesky44 * z4
	u4=scipy.stats.norm.cdf(x4,0,1)
	
	z5=gauss(0.0, 1.0)
	x5= cholesky51* z1 + cholesky52 * z2 + cholesky53 * z3 + cholesky54 * z4 + cholesky55 *z5
	u5=scipy.stats.norm.cdf(x5,0,1)
	
	random_numbers1.append(z1)
	random_numbers2.append(z2)
	random_numbers3.append(z3)
	random_numbers4.append(z4)
	random_numbers5.append(z5)
	
	correlated_random_numbers1.append(x2)
	correlated_random_numbers2.append(x2)
	correlated_random_numbers3.append(x3)
	correlated_random_numbers4.append(x4)
	correlated_random_numbers5.append(x5)
	
	uniform_vector1.append(u1)
	uniform_vector2.append(u2)
	uniform_vector3.append(u3)
	uniform_vector4.append(u4)
	uniform_vector5.append(u5)
	

# Default times are calculated with an iterative procedure and time_to_
# default vectors are populated accordingly (see below)  
# If ln⁡  (1-Ui )< Σ(-λi) the default does not happen in the corresponding 
# period. 
time_to_default_1=[] #default times for company 1
time_to_default_2=[] #default times for company 2
time_to_default_3=[] #default times for company 3
time_to_default_4=[] #default times for company 4
time_to_default_5=[] #default times for company 5

for i in range(0,I):
	if log(1-uniform_vector1[i])>lambda1[0]:
		time_to_default_1.append(0)
	elif log(1-uniform_vector1[i])>lambda1[1]:
		time_to_default_1.append(1)
	elif log(1-uniform_vector1[i])>lambda1[2]:
		time_to_default_1.append(2)
	elif log(1-uniform_vector1[i])>lambda1[3]:
		time_to_default_1.append(3)
	elif log(1-uniform_vector1[i])>lambda1[4]:
		time_to_default_1.append(4)
	else:
		time_to_default_1.append(5) 
		
		
	if log(1-uniform_vector2[i])>lambda2[0]:
		time_to_default_2.append(0)
	elif log(1-uniform_vector2[i])>lambda2[1]:
			time_to_default_2.append(1)
	elif log(1-uniform_vector2[i])>lambda2[2]:
		time_to_default_2.append(2)
	elif log(1-uniform_vector1[i])>lambda2[3]:
		time_to_default_2.append(3)
	elif log(1-uniform_vector1[i])>lambda2[4]:
		time_to_default_2.append(4)
	else:
		time_to_default_2.append(5)
		
		
	if log(1-uniform_vector3[i])>lambda3[0]:
		time_to_default_3.append(0)
	elif log(1-uniform_vector1[i])>lambda3[1]:
		time_to_default_3.append(1)
	elif log(1-uniform_vector3[i])>lambda3[2]:
		time_to_default_3.append(2)
	elif log(1-uniform_vector3[i])>lambda3[3]:
		time_to_default_3.append(3)
	elif log(1-uniform_vector1[i])>lambda3[4]:
		time_to_default_3.append(4)
	else:
		time_to_default_3.append(5)	
		
				
	if log(1-uniform_vector4[i])>lambda4[0]:
		time_to_default_4.append(0)
	elif log(1-uniform_vector4[i])>lambda4[1]:
		time_to_default_4.append(1)
	elif log(1-uniform_vector4[i])>lambda4[2]:
		time_to_default_4.append(2)
	elif log(1-uniform_vector4[i])>lambda4[3]:
		time_to_default_4.append(3)
	elif log(1-uniform_vector4[i])>lambda4[4]:
		time_to_default_4.append(4)
	else:
		time_to_default_4.append(5)
		
		
	if log(1-uniform_vector5[i])>lambda5[0]:
		time_to_default_5.append(0)
	elif log(1-uniform_vector5[i])>lambda5[1]:
		time_to_default_5.append(1)
	elif log(1-uniform_vector5[i])>lambda5[2]:
		time_to_default_5.append(2)
	elif log(1-uniform_vector5[i])>lambda5[3]:
		time_to_default_5.append(3)
	elif log(1-uniform_vector5[i])>lambda5[4]:
		time_to_default_5.append(4)
	else:
		time_to_default_5.append(5)
		

# Vectors for first, second, third, fourth and fifth defaults times are 
# populated by using the times to default of each company.
# For example if for the ith iteration  we have time_to_default_1[i] = 0
# time_to_default_2[i] = 2, time_to_default_3[i] = 4, time_to_default_5[i] = 5
# the first_default_time will be 0 (so at time 0), the second_default_time 
# will be 2 (so at time 2), etc.
# The function heapq.nsmallest(n, [list]) returns the nth smallest number
# in the list. For example heapq.nsmallest(2, [3,4, 5, 1, 22]) returns 3.

first_default_time=[]
second_default_time=[]
third_default_time=[]
fourth_default_time=[]
fifth_default_time=[]

for i in range(0, I):
	a=heapq.nsmallest(1, [time_to_default_1[i], time_to_default_2[i], 
	time_to_default_3[i], time_to_default_4[i], time_to_default_5[i]])[-1]
	
	first_default_time.append(a)
	
	b=heapq.nsmallest(2, [time_to_default_1[i], time_to_default_2[i], 
	time_to_default_3[i], time_to_default_4[i], time_to_default_5[i]])[-1]
	
	second_default_time.append(b)
	
	c=heapq.nsmallest(3, [time_to_default_1[i], time_to_default_2[i], 
	time_to_default_3[i], time_to_default_4[i], time_to_default_5[i]])[-1]
	
	third_default_time.append(c)
	
	d=heapq.nsmallest(4, [time_to_default_1[i], time_to_default_2[i], 
	time_to_default_3[i], time_to_default_4[i], time_to_default_5[i]])[-1]
	
	fourth_default_time.append(d)
	
	e=heapq.nsmallest(5, [time_to_default_1[i], time_to_default_2[i], 
	time_to_default_3[i], time_to_default_4[i], time_to_default_5[i]])[-1]
	
	fifth_default_time.append(e)
	


# For pricing purposes there are two legs: (1) the default leg (which pays 
# in case one of the five companies defaults; (2) the premium leg 
# (which will be decreased by 1/5 every time there is a default.)

default_leg_first_to_default = []  

for i in range(0,I):
	if first_default_time[i] > 4:
		default_leg_first_to_default.append(0)
	else:
		default_leg_first_to_default.append((1/5) * (1 - recovery_rate) * discount_factors[first_default_time[i]])

default_leg_second_to_default = []

for i in range(0,I):
	if second_default_time[i] > 4:
		default_leg_second_to_default.append(0)
	else:
		default_leg_second_to_default.append((1/5) * (1 - recovery_rate) * discount_factors[second_default_time[i]])
	
default_leg_third_to_default = []

for i in range(0,I):
	if third_default_time[i] > 4:
		default_leg_third_to_default.append(0)
	else:
		default_leg_third_to_default.append((1/5) * (1 - recovery_rate) * discount_factors[third_default_time[i]])
		
default_leg_fourth_to_default = []

for i in range(0,I):
	if fourth_default_time[i] > 4:
		default_leg_fourth_to_default.append(0)
	else:
		default_leg_fourth_to_default.append((1/5) * (1 - recovery_rate) * discount_factors[fourth_default_time[i]])
		
default_leg_fifth_to_default = []

for i in range(0,I):
	if fifth_default_time[i] > 4:
		default_leg_fifth_to_default.append(0)
	else:
		default_leg_fifth_to_default.append((1/5) * (1 - recovery_rate)* 
		discount_factors[fifth_default_time[i]])	
		
#print(default_leg_second_to_default)


premium_leg_first_to_default = []

for i in range(0, I):
	premium_leg_first_to_default.append(1 + min (4, first_default_time[i]) * 
	discount_factors[min (4, first_default_time[i])])
		# the first payment happens regardless (hence the 1), the subsequent
		# payments happen up to when the first default takes place, for 
		# example if the first_default_time is equal to 3, three premium 
		# payments will have taken place, (hence the first_default_time[i] 
		# in the formula; if the first_default_time[i] happens after the
		# fourth year a maximum of four payments, plus the first, take
		# place

premium_leg_second_to_default = []

for i in range(0, I):
	premium_leg_second_to_default.append(1 + min (4, first_default_time[i]) + 
	min (4, (second_default_time[i] - first_default_time[i])) * (4/5) *
	discount_factors[min (4, second_default_time[i])])
		# the first payment happens regardless (hence the 1), the subsequent
		# full payments happen up to when the first default takes place, if 
		# the first_default_time is equal to 3, three premium payments
		# will have taken place, for example, (hence the first_default_time[i] 
		# after the first default payment are decreased from 1 to (4/5), 
		# we have (4 - first_default_time[i]) * (4/5) payments

premium_leg_third_to_default = []

for i in range(0, I):
	premium_leg_third_to_default.append(1 + min(4, first_default_time[i]) * 
	discount_factors[min (4, first_default_time[i])] + 
	min(3,(second_default_time[i] - first_default_time[i])) * (4/5) * 
	discount_factors[min (4, second_default_time[i])] + 
	min (2, third_default_time[i] - second_default_time[i]) * (3/5)* 
	discount_factors[min (4, third_default_time[i])])

premium_leg_fourth_to_default = []

for i in range(0, I):
	premium_leg_fourth_to_default.append(1 + min (4, first_default_time[i])*
	discount_factors[min (4, first_default_time[i])] + 
	min (4, (second_default_time[i] - first_default_time[i])) * (4/5) * 
	discount_factors[min (4, second_default_time[i])] + 
	min ( 4, (third_default_time[i] - second_default_time[i])) * (3/5) * 
	discount_factors[min (4, third_default_time[i])] +
	min (4, (fourth_default_time[i] - third_default_time[i])) * (2/5) *
	discount_factors[min (4, fourth_default_time[i])])
	
premium_leg_fifth_to_default = []

for i in range(0, I):
	premium_leg_fifth_to_default.append(1 + min(4, first_default_time[i]) *
	discount_factors[min (4, first_default_time[i])] + 
	min (4, (second_default_time[i] - first_default_time[i])) * (4/5) *
	discount_factors[min (4, second_default_time[i])] + 
	min (4, (third_default_time[i] - second_default_time[i])) * (3/5) *
	discount_factors[min (4, third_default_time[i])] +
	min (4, (fourth_default_time[i] - third_default_time[i])) * (2/5) *
	discount_factors[min (4, fourth_default_time[i])] +
	min (4, (fifth_default_time[i] - fourth_default_time[i])) * (1/5) *
	discount_factors[min (4, fifth_default_time[i])])
	

average_premium_first_to_default = sum(premium_leg_first_to_default)/len(premium_leg_first_to_default)

average_default_leg_first_to_default = sum(default_leg_first_to_default)/ len(default_leg_first_to_default)

spread_first_to_default = average_default_leg_first_to_default / average_premium_first_to_default


average_premium_second_to_default = sum(premium_leg_second_to_default)/len(premium_leg_second_to_default)

average_default_leg_second_to_default = sum(default_leg_second_to_default)/ len(default_leg_second_to_default)

spread_second_to_default = average_default_leg_second_to_default / average_premium_second_to_default


average_premium_third_to_default = sum(premium_leg_third_to_default)/len(premium_leg_third_to_default)

average_default_leg_third_to_default = sum(default_leg_third_to_default)/ len(default_leg_third_to_default)

spread_third_to_default = average_default_leg_third_to_default / average_premium_third_to_default


average_premium_fourth_to_default = sum(premium_leg_fourth_to_default)/len(premium_leg_fourth_to_default)

average_default_leg_fourth_to_default = sum(default_leg_fourth_to_default)/ len(default_leg_fourth_to_default)

spread_fourth_to_default = average_default_leg_fourth_to_default / average_premium_fourth_to_default


average_premium_fifth_to_default = sum(premium_leg_fifth_to_default)/len(premium_leg_fifth_to_default)

average_default_leg_fifth_to_default = sum(default_leg_fifth_to_default)/ len(default_leg_fifth_to_default)

spread_fifth_to_default = average_default_leg_fifth_to_default / average_premium_fifth_to_default


print('spread for first to default', spread_first_to_default * 10000, 'bps \n')
print('spread for second to default', spread_second_to_default * 10000, 'bps \n')
print('spread for third to default', spread_third_to_default * 10000, 'bps \n')
print('spread for fourth to default', spread_fourth_to_default * 10000, 'bps \n')
print('spread for fifth to default', spread_fifth_to_default * 10000,  'bps \n')
		
