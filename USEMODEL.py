# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# create and save a plot of generated images (reversed grayscale)
#def save_plot(examples, n):
	# plot images
	#for i in range(n * n):
		# define subplot
		#pyplot.subplot(n, n, 1 + i)
		# turn off axis
		#pyplot.axis('off')
		# plot raw pixel data
		#pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	#pyplot.show()

# load model
model = load_model('generator_modelfinal_4970.h5')
# generate images
latent_points = generate_latent_points(91, 1) # zweite Stelle  zeilen
# generate images
X = model.predict(latent_points)[0]
print("X.shape==" ,X.shape)
print("type(X)==",type(X))
print(X)
final = pd.DataFrame(X)
#print(type(final))
#print(final.shape)
final.plot.kde()
plt.savefig("figureone.png")
final=final.T
final.to_csv("fake.csv",sep= ";",header=None,index=False)
#FINAL= pd.DataFrame("fake.csv")
print(final.head)
print(final.head)
print(type(final))
#FINAL=final.to_csv("FAKE.csv",sep=";",header= None)
#FAKE= pd.DataFrame("FAKE.csv")
#print(FAKE.head)
#df = pd.DataFrame(X)
#print(df.head())
#print(type(df))
#print(df.shape)

#df.to_csv("one.csv",sep=";",header=None)
#df.CompTotal.plot.density(figsize=(8,6),frontsize=14,linewidth=4)
#xlabel("beta values")
#savefig("Density of beta values")

"""
a=numpy.asarray(X)
numpy.savetxt("one.csv", a, delimiter=",")
b=pd.read_csv("final.csv",sep= ";",header=None)
print(b.head())
print(type(b))
print(b.shape)
# plot the result
#save_plot(X, 5)
"""
