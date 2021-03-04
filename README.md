# Bachelor-thesis
1.Normal GAN with methylation data:

1.1Code:MNIST_1D_modifiziert_Conv1D_tumor2021.py


-> GAN implementation to generate random methylation data


-> Link source: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/

1.2.Code: USEMODEL.py


-> Use Model from MNIST_1D_modifiziert_Conv1D_tumor2021.py


-> generated random methylation data
____________________________________________________________________________________
2. AC-GAN with 2D MNIST Data
AC-GAN (2D MNIST)
2.1.Code:AC_GAN.py (2D)
-> AC GAN for class aware  MNIST Fashion data(2D)
-> Link source: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/

2.2 Code:USEMODELAC.py (2D)
-> Use final model to generate class aware MNIST Fashion data(2D)
-> Link source: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/
_____________________________________________________________________________________
3.AC-GAN-1D with 1D MNIST Data

3.1 Code:AC_GAN_1D_MNIST.py
-> AC GAN for class aware MNIST Fashion Data(1D)  
->1D MNIST Data on Conv1D
->Link source: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/

3.2 Code: USEMODELAC_1D.py (1D)
-> Use final model to generate class aware MNIST Fashion data(1D)
->Link source: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/
_____________________________________________________________________________________

4.AC-GAN 1D with Methylation data (version 1)

4.1.Code: AC_GAN_Tumor_1D.py
-> AC-GAN implementation to generated class aware methylation data
-> Version 1
-> Link source: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/

4.2 Code USEMODELAC_Tumor_1D.py
-> Use final model to generated class aware methylation data (just one sample)
-> Link source: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/

4.3 Code USEMODELAC_Tumor_1D_more_sample.py
-> Use final Mode to generated class aware methylation data( more then one sample)

_____________________________________________________________________________________

5. Evaluation AC-GAN 

5.1 Code AC_GAN_Tumor_1D_evaluation.py
-> Evaluating  Discriminator from AC-GAN 

5.2 Code : PCA_plot_corrected.py
-> Evaluating Generator with centroids for one sample

5.3 Code :PCA_plot_multiple_data_cleaned.py 
-> Evaluating Generator with centroids for more samples





