# VBTA
## Semi-supervised experiment reproduction

1. Run python train.py script. This script will train VBTA model in semi-supervised regime. See the command line help for the details.
2. Run train_mnist_evaluator.sh scipt. This scripts downloads MNIST classifier and trains the classification model.
3. Run eval.py scipt. The scipt will calculate the accuracy of the trained VBTA model. 


## CelebA model training

1. Download celebA dataset
2. Run get_data_celeba.py
3. Run train_celeba.py script. This script runs VBTA training and saved model. See the command line help for the details.
