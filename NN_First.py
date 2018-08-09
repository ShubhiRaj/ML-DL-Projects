import numpy as np 

def nonlin(x, deriv=False):
	if(deriv==True):
		#Return derivative of sigmoid
		return x*(1-x)
	#Return sigmoid function
	return 1/(1+np.exp(-x))

#Each row is a different training example, every column represents a different neuron

#Input data
x=np.array([[0,0,1],
	[0,1,1],
	[1,0,1],
	[1,1,1]
	])

#Output data
y=np.array([[0],[1],[1],[0]
	])


#Check what is seeding: give randomly generated numbers a common strating point    #???????????
np.random.seed(1)

#Creating the synapse matrix
#Since we have three layers, we need two synapse matrices
syn0 = 2*np.random.random((3,4))-1													#??????????
syn1 = 2*np.random.random((4,1))-1

#Assigning random weights to each synapse

#Beginning with training examples
#Performing matrix multiplication between each layer and the synapse
for j in xrange(60000):																#??????????
	l0=x
	l1=nonlin(np.dot(l0,syn0))
	l2=nonlin(np.dot(l1,syn1))

	#Getting the error rate
	l2_error=y-l2

	#Printng average error rate to keep a track that it goes down everytime
	if(j%10000)==0:
		print "Error"+str(np.mean(np.abs(l2_error)))

	#Multiplying the error rate by the result of the sigmoid function
	#Used to get the derivative of our output prediction 
	#This gives the delta which will be used reduce the error rate of our predictions when we update our synapse after every iteration
	l2_delta = l2_error * nonlin(l2, deriv = True)
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1, deriv=True)
	#This was BACK PROPOGATION 


	#Using the deltas to update the weights to reduce the error
	#called GARDIENT DESCENT
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

print("Output after training")
print l2 







