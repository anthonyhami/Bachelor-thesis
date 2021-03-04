# example of fitting an auxiliary classifier gan (ac-gan) on fashion mnsit
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
"""
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
"""


from matplotlib import pyplot
import pandas as pd
import math
import h5py
import numpy as np
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

#from  keras.utils.vis_utils import plot_model

from keras.datasets.fashion_mnist import load_data
from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Activation


from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# define the standalone discriminator model
def define_discriminator(in_shape=(5000,1), n_classes=91):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    # downsample to 14x14
    fe = Conv1D(32, 3*1, strides=2*1, padding='same', kernel_initializer=init)(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # normal
    fe = Conv1D(64, 3*1, padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # downsample to 7x7
    fe = Conv1D(128, 3*1, strides=2*1, padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # normal
    fe = Conv1D(256, 3*1, padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt, metrics=['acc',f1_m,precision_m, recall_m])
    return model
model=define_discriminator()
model.summary()
#plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)


# define the standalone generator model
def define_generator(latent_dim, n_classes=91):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # label input
    in_label = Input(shape=(1,))
    #print("in_label.shape===",in_label.shape)
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    #print("li.shape====",li.shape)
    # linear multiplication
    n_nodes = 1250*128  #196*128
    #print("n_nodes.shape==",n_nodes.shape)
    li = Dense(n_nodes, kernel_initializer=init)(li)
    # reshape to additional channel
    li = Reshape((1250, 128))(li) #196,128
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = int(128 *(5000/4) * 1)
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((math.ceil(5000/4)* 1, 128))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 14x14
    gen = Conv1DTranspose(128, 4*1, strides=2*1, padding='same', kernel_initializer=init)(merge)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    # upsample to 28x28
    gen = Conv1DTranspose(1, 4*1, strides=2*1, padding='same', kernel_initializer=init)(gen)
    out_layer = Activation('tanh')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model
latent_dim = 100
model = define_generator(latent_dim)
model.summary()
#plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

# load images
"""
def load_real_samples():
    # load dataset
    (trainX, trainy), (_, _) = load_data()
    print("trainX.shape===="trainX.shape)
    print("trainy.shape===="trainy.shape)
    print(trainy.head())

    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32').reshape((60000,784,-1))
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print(X.shape, trainy.shape)
    exit()
    return [X, trainy]
"""
#load methylationdata
def load_real_samples():
    #trainX = pd.read_csv("betaValues.csv", sep=";",header=None).to_numpy()
    #trainX=trainX[:,:5000]
    #print ("trainX.shape ============================ ", trainX.shape)

    trainy = pd.read_csv("Dx_with_num.csv", sep=";")["Dx_num"].to_numpy()
    print ("trainy.shape ============================ ", trainy.shape)

    h5f = h5py.File('betaValues.h5','r')
    trainX= h5f['dataset'][:]
    trainX= np.array((h5f["dataset"][:]),dtype=np.float32)
    print("trainX.shape---------=",trainX.shape)
    print("type(trainX)==",type(trainX))
    trainX=trainX[:,:5000]
    print("trainX.shape---------=",trainX.shape)
    print("type(trainX)==",type(trainX))
    # von (2800, 5000) auf (2800, 5000, 1)
    X = expand_dims(trainX, axis=-1)
    print ("EXPAND :: X.shape ============================ ", X.shape)
    h5f.close()

    return [X, trainy]





#new load images
"""
def load_real_samples():
    #load dataset
    (trainX,trainy),(_, _)= load_data()
    #reshape
    shapes= trainX.shape
    trainX= trainX.flatten().reshape(shapes[0],shapes[1]*shapes[2])
    #expand to §d,e.g. add channels
    X=expand_dims(trainX,axis=-1)
    X=expand_dims(X,axis=-1)
    #convert from ints to floats
   # X=X.astype("float32")
    #scale from [0,255] to [-1,1]
    X= (X - 127,5) / 127.5
    print(X.shape,trainy.shape)
    return [X,trainy]
"""

# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=91):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    print("images.shape===",images.shape)
    return [images, labels_input], y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    [X_real, _], y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    #_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    #loss_real, accuracy_real, f1_score_real, precision_real, recall_real = d_model.evaluate(X_real, y_real, verbose=0)
    _,d_r,d_r2, acc_r, f1_r, prec_r, recall_r, _,_,_,_ = d_model.evaluate(X_real, y_real, verbose=0)

    # prepare fake examples
    [x_fake, _], y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    #_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    #loss_fake, accuracy_fake, f1_score_fake, precision_fake, recall_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    _,d_f,d_f2, acc_f, f1_f, prec_f, recall_f, _,_,_,_ = d_model.evaluate(x_fake, y_fake, verbose=0)

    print ("\n\n\n")
    print ("step ===================", step)

    print ("d_r ====================", d_r)
    print ("d_r2 ===================", d_r2)
    print ("acc_r ==================", acc_r)
    print ("f1_r ===================", f1_r)
    print ("prec_r =================", prec_r)
    print ("recall_r ===============", recall_r)

    print ("d_f ====================", d_f)
    print ("d_f2 ===================", d_f2)
    print ("acc_f ==================", acc_f)
    print ("f1_f ===================", f1_f)
    print ("prec_f =================", prec_f)
    print ("recall_f ===============", recall_f)

    # summarize discriminator performance
    #print('>Accuracy real: %.0f%%, fake: %.0f%%' % (accuracy_real*100, accuracy_fake*100))
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_r*100, acc_f*100))

    """
    # prepare fake examples
    [X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    print("X.shape==",X.shape)
    """
    # save the generator model
    filename2 = 'AC_Model_Tumor/model_%04d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s' % (filename2))#(filename1,filename2)


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        #_,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
        D_r = d_model.train_on_batch(X_real, [y_real, labels_real])
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        #_,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        D_f = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        _,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])

        print ("D_r ==== ", D_r)
        print ("D_f ==== ", D_f)

        # summarize loss on this batch
        #print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))
        # evaluate the model performance every 'epoch'
        if (i+1) % (bat_per_epo * 10) == 0:
        #   summarize_performance(i, g_model, d_model, dataset, latent_dim)
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
