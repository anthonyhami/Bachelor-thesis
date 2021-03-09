# example of loading the generator model and generating images
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot
import pandas as pd
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_class):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = asarray([n_class for _ in range(n_samples)])
	return [z_input, labels]

for n_class in range(91):


	# load model
	model = load_model('model_21500.h5')
	latent_dim = 100
	n_examples = 10 # must be a square
	#n_class = 78 #
	# generate images
	latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
	# generate images
	X  = model.predict([latent_points, labels])#[0]
	print("X.shape====",X.shape)
	print("type(X)==",type(X))
	print(X)
	X=X.reshape(10,-1)
	final=pd.DataFrame(X)
	print("type(final)====",type(final))
	print("final.shape==",final.shape)
	#print("final.min()===",final.min())
	#print("final.max()==",final.max())
	#final.plot.kde()
	#pyplot.savefig("")
	#final=final.T
	final.to_csv("fake_orginal_"+str(n_class)+".csv",sep= ";",header=None,index=False)
	print(final.head)
	print(type(final))
	#print("X.shape====",X.shape)
