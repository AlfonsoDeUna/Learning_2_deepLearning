import numpy as np
from keras.datasets import cifar10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Download the data from dataset 
(x_train, y_train) , (x_test, y_test) = cifar10.load_data()

#pasamos los datos de una 2D a vectores para trabajar con ellos KNN TRABAJA CON VECTORES
y_train = y_train.ravel()
y_test = y_test.ravel()

# Redimensionamos las imagenes a vectores para trabajar con el algoritmo
# CIFAR-10 es un conjunto de datos de imágenes, donde cada imagen tiene una dimensión de 
# cada imagen es de 32 por 32 con 3 canales RGB 3072 valores por imagaenr

x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Usar k-NN con k=3 (puedes ajustar el valor de k según lo desees)
knn = KNeighborsClassifier(n_neighbors=3)

#Aqui entrena el modelo con los datos de entrenamiento de entrada y salida
knn.fit(x_train, y_train)

# we use the data test set and the return will be the prediction.
y_pred = knn.predict(x_test)

# We compare the prediction with the data that we had from x_test. We can see the precision and other interesting parameters
print(classification_report(y_test, y_pred))