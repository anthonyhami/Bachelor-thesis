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
""""
# create and save a plot of generated images
def save_plot(examples, n_examples):
	# plot images
	for i in range(n_examples):
		# define subplot
		pyplot.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		change=X[i, :, 0]#examples.reshape((100,28,28,1))
		change=change.reshape((28,28))
		print("change.shape====",change.shape)
		pyplot.imshow(change, cmap='gray_r')
	pyplot.savefig("plot_1D_new.png")
"""
# load model
model = load_model('model_21500.h5')
latent_dim = 100
n_examples = 3 # Anzahl an Zeilen hier ändern
n_class = 81 #
# generate images
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
# generate images
X  = model.predict([latent_points, labels])#[0]
print("X.shape====",X.shape)
print("type(X)==",type(X))
print(X)
X=X.reshape(3,-1)   #erste stelle ändern für anzahl an  Zeilen
final=pd.DataFrame(X)
print("type(final)====",type(final))
print("final.shape==",final.shape)
print("final.min()===",final.min())
print("final.max()==",final.max())
#final.plot.kde()
pyplot.savefig("81_500_PTPR,A_last_sig_new.png")
#final=final.T
final.to_csv("fake81_500_PTPR,A_last_sig_new.csv",sep= ";",header=None,index=False)
print(final.head)
print(type(final))
#print("X.shape====",X.shape)
