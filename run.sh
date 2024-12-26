#!/bin/bash

test_flag=$1 # either --test or nothing
if [[ "$test_flag" != "" && "$test_flag" != "--test" ]]; then
  echo "Error: Invalid test flag. Use '--test' or leave empty."
  exit 1
fi

set -ex

# python train.py -m AE -d MNIST -c config/ae_mnist.yaml $test_flag
# python train.py -m VAE -d MNIST -c config/vae_mnist.yaml $test_flag
# python train.py -m VADE -d MNIST -c config/vade_mnist.yaml $test_flag
python train.py -m VADE -d MNIST -c config/vade_mnist_pretrain.yaml $test_flag
# python train.py -m AE -d HAR -c config/ae_har.yaml $test_flag
# python train.py -m VAE -d HAR -c config/vae_har.yaml $test_flag
# python train.py -m VADE -d HAR -c config/vade_har.yaml $test_flag
# python train.py -m VADE -d HAR -c config/vade_har_pretrain.yaml $test_flag

echo "All trainings passed"