# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
"""
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
"""
from matplotlib import pyplot
import pandas as pd
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

#2801, 5000
#2801,428799
# define the standalone discriminator model
def define_discriminator(in_shape=(5000,1)):
	model = Sequential()
	model.add(Conv1D(64, 3*1, strides=2*1, padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv1D(64, 3*1, activation="relu", strides=2*1, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()

	return model
print("xxxxold")
# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = int(128 *(5000/4) * 1)
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((math.ceil(5000/4)* 1, 128)))
	# upsample to 14x14
	model.add(Conv1DTranspose(128, 4*1, strides=2*1, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv1DTranspose(128, 4*1, strides=2*1, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv1D(1,int(5000/4)*1, activation='sigmoid', padding='same'))
	model.summary()

	return model
print("XXXXXX")
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load and prepare mnist training images
def load_real_samples():
	print ("----------------------------------------------------------------")
	print ("----------------------------------------------------------------")
	print ("----------------------------------------------------------------")
	print ("----------------------------------------------------------------")
	print ("----------------------------------------------------------------")
	print ("----------------------------------------------------------------")
	print ("----------------------------------------------------------------")
	print ("----------------------------------------------------------------")
	print ("----------------------------------------------------------------")
	# load mnist dataset
	#(trainX, _), (_, _) = load_data()
	#trainX = trainX[:1000, :, :] # take only 1000
	trainX = pd.read_csv("betaValues.csv", sep=";",header=None).to_numpy()
	trainX=trainX[:,:5000]
	print ("trainX.shape ============================ ", trainX.shape)


	# reshape
	shapes = trainX.shape
	#trainX = trainX.flatten().reshape(shapes[0], shapes[1]*shapes[2]) # von (60000, 28, 28) auf (60000, 784)
	print ("trainX.shape ============================ ", trainX.shape)

	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1) # von (60000, 784) auf (60000, 784, 1)
	print ("EXPAND :: X.shape ============================ ", X.shape)
	#X = expand_dims(X, axis=-1) # von (60000, 784, 1) auf (60000, 784, 1, 1)
	#print ("EXPAND :: X.shape ============================ ", X.shape)
	# convert from unsigned ints to floats

	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	#X = X / 255.0

	print ("X.shape ============================ ", X.shape)
	print ("X.shape ============================ ", X.shape)
	print ("X.shape ============================ ", X.shape)
	print ("X.shape ============================ ", X.shape)
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	print("ix====",ix)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	print("x_input.shape====",x_input.shape)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	print ("---------------------- save_plot")
	print ("---------------------- save_plot")
	print ("---------------------- save_plot")
	print ("---------------------- save_plot")
	print ("---------------------- save_plot")
	print ("---------------------- save_plot")
	shapes = examples.shape
	print ("shapes == ", shapes)
	#examples = examples.flatten().reshape(shapes[0], 28,28, 1)
	examples = examples.flatten().reshape(shapes[0], 28,28)
	print ("examples.shapes == ", examples.shape)

	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		#pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
		pyplot.imshow(examples[i, :, :], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	#save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'Modelfasthalfnew/generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		print("i======",i)
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			print("j=============",j)
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			print("X_real.shape==",X_real.shape)
			print("y_real.shape==",y_real.shape)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			print("X_fake.shape==",X_fake.shape)
			print("y_fake.shape==",y_fake.shape)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 2 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
