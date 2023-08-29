# Learning space: Anatomy of neural network

* Layers o capas que son los elementos que conforman el modelo

   In this moment we can talk about weights. It's the result form input data store in this layer.
   The layer transform the data by this weights. And the network learn what weight is the most significant to this layer to transform the data


* Input data y target (datos de entrada y objetivos)
* Loss Function¡: Define la señal de feedback usada para aprender
*  Optimizer: the job of this piece is to adjust the value of weights. Implements backpropagation algorithm: the central piece of deep learning

How propagation works>>>>>

Initailly, the weights of the network are assigned random values., for every example the network process the value of weights is adjust a little in the correct direction.

This correct direcction is done by training loop. A network with minimal loos is the best option to acuracy predictions.

Lets talk now more about loss function: also called objective function takes the predictions of the network and the true target and computes a distance score. 
How well the network has done on this specific example. 

How much is learning or prediction acuracy to the reality.



* Optimizer: Determina la manera de aprender
* Loss function compara las predicciones con los objetivos. Es una medida de cómo las predicciones de la red concuerdan con lo esperado

## How it works
El optimizador utiliza el valor del loss function para actualizar los pesos de la red neuronal 

Las capas de un módulo de procesado de datos que toma uno o más tensores y devuelve uno o más tensores

Capas stateless (sin estado) pero lo normal es que tengan un estado
¿Qué puede ser el estado?

el peso de la capa, o o varios tensores que aprenden mediente el método de descenso del gradiente estocástico

fully conected o dense layer (keras lo llaman Dense) --> 2D

recurrent layers LSTM layer --> 3D

ImageData 4D tensor --> 2D convolution layers (Conv2D)


Softmax it's a function. It works converting a vector to real numbers. It's use for neural networks last layer (multiclass clasification)




