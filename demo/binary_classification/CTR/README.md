# Cretio CTR prediction

---

Display advertising is a billion dollar effort and one of the central uses of machine learning on the Internet. In this demo, we use a very small part of the data in the Cretio CTR challenge on *[kaggle][1]* to demonstrate how to use the ffm algorithm to tackle this challenge. You can get the details of this algoritm in this *[paper][2]*.

## Install

First you should compile and install the f2m package on your own computer. You can read the *[install.md][3]* for the installation detail.

## File

We have three files in this demo: 

 1. train.txt: Training data set
 2. test.txt: Test data set
 3. CTR.conf: Configuration for this demo
 
The *train.txt* and *test.txt* have been transformed into the *[libffm][4]* format. Note that in this demo the validation set and test set are the same file (*test.txt*).

## Hyper Parameters

The *CTR.conf* is the configuration for this demo. Each line containing the [attibute]=[value], which is shown in the following:

    [HyperParameter]
        
    # Training model
    is_train = true
        
    # Using ffm model
    model_type = ffm
    
    # For sparse data
    is_sparse = true
    
    # Set learning rate
    learning_rate = 0.1
    
    # The size of the latent factor in ffm model
    num_factor = 20
    
    # Using libffm file format
    file_format = libffm
    
    # Using l2 regularizer
    regu_type = l2
    
    # Set lambda for regularizer
    regu_lambda = 0.0001
    
    # The Training file
    train_set_file = "./train.txt"
    
    # The validation file
    test_set_file = "./test.txt"
    
    # Number of iteration
    num_iteration = 10
    
    # Set mini-batch size
    batch_size = 200

*is_train* indicate that we will train the model. Note that the CTR data is very sparse, so we set *is_sparse = true*, which can speed up our training.

## Train

Then we can run the training process. On default, f2m uses sgd to train model:

    chmod +x ./run.sh
    ./run.sh

After 10 iterations we get the loss: ***0.545506*** .

    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.675654
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.654361
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.620385
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.585947
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.564106
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.553456
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.548329
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.545506
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:361 (Train) Current loss value: 0.545506
    Mon Mar 20 10:16:01 2017
     /Users/alex/f2m/src/train/train.cc:225 (Finalize) Finalize successfully.

## Early stop

Early stop is a very useful feature as sometimes we do not how to set the iteration number to get a better score.
In this demo we can modify the configuration:

    # Number of iteration
    num_iteration = 100
    
    # Using early stop
    early_stop = true
    
then run the training process:

    ./run.sh

This time we can get a better loss : ***0.527836***

    Mon Mar 20 10:52:08 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.528831
    Mon Mar 20 10:52:08 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.52836
    Mon Mar 20 10:52:08 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.528018
    Mon Mar 20 10:52:09 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.527836
    Mon Mar 20 10:52:09 2017
     /Users/alex/f2m/src/train/train.cc:339 (Train) Current loss value: 0.527843
    Mon Mar 20 10:52:09 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Early stop at iteration 34 / 100
    Mon Mar 20 10:52:09 2017
     /Users/alex/f2m/src/train/train.cc:362 (Train) Current loss value: 0.527836
    Mon Mar 20 10:52:09 2017
     /Users/alex/f2m/src/train/train.cc:225 (Finalize) Finalize successfully.

We can see that the training process stop at 34th iteration.

## Cross-Validation

Cross-validation is used for choosing hyper parameters, f2m also provide this feature to users. We can add two lines code in the configuration:

    cross_validation = true
    
    num_folds = 5
    
We also need to set the iteration number:

    num_iteration = 100
    
where we use 5-folds cross-validation on the training set and we can get a average loss: ***0.581186***

    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:374 (CVTrain) Start to cross validation.
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:379 (CVTrain) K folds: 0/5
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:413 (CVTrain) Loss value for the 0th test set : 0.422609
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:379 (CVTrain) K folds: 1/5
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:413 (CVTrain) Loss value for the 1th test set : 0.456357
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:379 (CVTrain) K folds: 2/5
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:413 (CVTrain) Loss value for the 2th test set : 0.840365
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:379 (CVTrain) K folds: 3/5
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:413 (CVTrain) Loss value for the 3th test set : 0.65117
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:379 (CVTrain) K folds: 4/5
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:413 (CVTrain) Loss value for the 4th test set : 0.535428
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:418 (CVTrain) The average loss is : 0.581186
    Tue Mar 21 07:50:33 2017
     /Users/alex/f2m/src/train/train.cc:225 (Finalize) Finalize successfully.

## Try More Updaters

The updater provided by f2m is not limited in sgd. Our system can support six kinds of updaters, including sgd,
adadelta, adagrad, adam, momentum, and rmrsprop, you can learn the detail about these updaters in this *[blog][5]*.

Using f2m, we can set the *updater* parameter to indentify which updater we want to use. As we discussed before, f2m uses the sgd updater by default. Now Let's try more.

    updater = adagrad
    
    corss_validation = false
    
    learning_rate = 0.01
    
We can get loss value at the 6th iteration using early-stopping: ***0.557695***

    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:327 (Train) Start to train model.
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.595252
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.573463
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.564585
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.560257
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.558249
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.557695
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.55816
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:345 (Train) Early stop at iteration 6 / 100
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:365 (Train) Current loss value: 0.557695
    Tue Mar 21 08:22:22 2017
     /Users/alex/f2m/src/train/train.cc:225 (Finalize) Finalize successfully.
    



  [1]: https://www.kaggle.com/c/criteo-display-ad-challenge
  [2]: http://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
  [3]: https://github.com/aksnzhy/f2m/blob/master/install.md
  [4]: https://github.com/guestwalk/libffm/blob/master/README
  [5]: http://sebastianruder.com/optimizing-gradient-descent/
  
 Using momentum, we can a better loss at the 36th iteration: ***0.529538***
 
    updater = momentum

Loss print:
    
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.530977
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.530489
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.530071
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.529754
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.529566
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.529538
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:342 (Train) Current loss value: 0.5297
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:345 (Train) Early stop at iteration 36 / 100
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:365 (Train) Current loss value: 0.529538
    Tue Mar 21 08:24:52 2017
     /Users/alex/f2m/src/train/train.cc:225 (Finalize) Finalize successfully.
     
The other result:

    adam: 0.538944
    rmsprop: 0.543123
    adadelta: 0.567508
    
## Prediction

After training, f2m will save model parameter to disk file, the path is /tmp/model_ck. User can modify this path by using the *model_checkpoint_file* parameter.

    model_checkpoint_file = /home/alex/blablabla..

Using this model, we can make a prediction on test set. Here we just set:

    is_train = false

Then the result has been written into ./result.txt

    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:102 (Initialize) Start to initialize the context of f2m.
    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:112 (Initialize) Initialize hyper parameters successfully.
    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:121 (Initialize) Initialize Parser successfully.
    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:152 (Initialize) Read problem successfully.
    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:168 (Initialize) Initialize model parameters successfully.
    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:178 (Initialize) Initialize Loss successfully.
    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:437 (StartPredictWork) Start predication work.
    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:451 (StartPredictWork) ========= Write result in ./result.txt ===========
    Tue Mar 21 09:39:53 2017
     /Users/alex/f2m/src/train/train.cc:231 (Finalize) Finalize successfully.
     
result.txt:

    head -20 ./result.txt
    
    -1.469020
    -0.587282
    -0.719019
    -0.425391
    -0.833108
    -0.676215
    -0.454732
    -1.668074
    -0.671267
    -1.183283
    -0.240106
    -0.763266
    -0.800452
    -1.280936
    -0.799617
    -0.442692
    -0.970113
    -0.454732
    -1.314491
    -0.589459
    
We can transform this result using sigmoid function by seting the *sigmoid* parameter:

    sigmoid = true

then run:

    ./run.sh
    
result.txt:

    head -n 20
    
    0.187098
    0.357269
    0.327614
    0.395229
    0.302998
    0.337108
    0.388240
    0.158681
    0.338215
    0.234466
    0.440264
    0.317948
    0.309939
    0.217398
    0.310118
    0.391102
    0.274860
    0.388240
    0.211743
    0.356768
