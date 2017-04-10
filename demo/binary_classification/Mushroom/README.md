# Mushroom classification

---

Here we demonstrate how to use factorization machines for binary classification task. Here we use *[mushroom dataset][1]* from UCI machine learning repository. The data format is the same as in LIBSVM: Each row contains a training case (x, y) for real-value feature vector x with target y. The row states first the value y and then the non-zero values of x. For binary classification, cases y > 0 are regarded as the positive class and with y <= 0 as the negative class.

## Install

First you should compile and install the f2m package on your own computer. You can read the *[install.md][2]* for the installation detail.


## File

We have three files in this demo:

 1. train.txt: Traning data set
 2. test.txt: Test data set
 3. mushroom.conf: Configuration for this demo

The train.txt and test.txt have been transformed into the libsvm format. Note that this demo the validation set and test set are the same file (test.txt)

## Hyper Parameter

[HyperParameter]
    
    # Training model
    is_train = true
        
    # Using ffm model
    model_type = fm
    
    # For sparse data
    is_sparse = false
    
    # Set learning rate
    learning_rate = 0.1
    
    # The size of the latent factor in ffm model
    num_factor = 20
    
    # Using libffm file format
    file_format = libsvm
    
    # Using l2 regularizer
    regu_type = l2
    
    # Set lambda for regularizer
    regu_lambda = 0.0001
    
    # The Training file
    train_set_file = "./train.txt"
    
    # The validation file
    test_set_file = "./test.txt"
    
    # Number of iteration
    num_iteration = 100
    
    # Set mini-batch size
    batch_size = 10
    
is_train indicate that we will train the model. Note that the Mushroom data is dense, so we set is_sparse = false, which can speed up our training.

## Train

Then we can run the training process. On default, f2m uses sgd to train model:

    chmod +x ./run.sh
    ./run.sh
    
After 100 iterations we get the loss: ***0.000175262***.

Using other updater such adam, momentum, adadelta and rmsprop, we can get the loss: ***0***, which is the same to the result training on xgboost.

We can run libfm to get the same result:

    ./libFM -task c -train /tmp/train.txt -test /tmp/test.txt -dim ’1,1,16’ -iter 100 -method sgd -learn_rate 0.1 -regular ’0,0,0.01’ -init_stdev 0.1

    #Iter= 77   Train=0 Test=0
    #Iter= 78   Train=0 Test=0
    #Iter= 79   Train=0 Test=0
    #Iter= 80   Train=0 Test=0
    #Iter= 81   Train=0 Test=0
    #Iter= 82   Train=0 Test=0
    #Iter= 83   Train=0 Test=0
    #Iter= 84   Train=0 Test=0
    #Iter= 85   Train=0 Test=0
    #Iter= 86   Train=0 Test=0
    #Iter= 87   Train=0 Test=0
    #Iter= 88   Train=0 Test=0
    #Iter= 89   Train=0 Test=0
    #Iter= 90   Train=0 Test=0
    #Iter= 91   Train=0 Test=0
    #Iter= 92   Train=0 Test=0
    #Iter= 93   Train=0 Test=0
    #Iter= 94   Train=0 Test=0
    #Iter= 95   Train=0 Test=0
    #Iter= 96   Train=0 Test=0
    #Iter= 97   Train=0 Test=0
    #Iter= 98   Train=0 Test=0
    #Iter= 99   Train=0 Test=0
    Final   Train=0 Test=0

 
 
  [1]: https://archive.ics.uci.edu/ml/datasets/Mushroom
  [2]: https://github.com/aksnzhy/f2m/blob/master/install.md