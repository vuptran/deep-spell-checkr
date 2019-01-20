# Sequence to Sequence Learning for Spelling Correction

This repository contains a Keras implementation of an encoder-decoder LSTM architecture for sequence-to-sequence spelling correction. The character-level spell checker is trained on unigram tokens derived from a vocabulary of more than 466k unique English words. After about 12 hours of training, the speller achieves an accuracy performance of 96.7% on a validation set comprised of more than 26k tokens.

```
Input sentence:
> The rabit holEe ewnt stiraight on like a tunenl ofr some wya an then zipped suddenl odwn so suddenly tht lice ad noHt a moment to think about stopping hertelf before she fovund herself faljling odwn a very deep weNl

Decoded sentence:
> The rabit hole went straight on like a tunnel ofr some way an then zipped suddenly down so suddenly thet lice ad not a moment to think about stopping heretly before she found herself falling down a very deep well

Target sentence:
> The rabbit hole went straight on like a tunnel for some way and then dipped suddenly down so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well
```

Clearly, the speller still has room for improvement. Here are some ideas:

* Implement an attention mechanism
* Incorporate beam search as part of the loss evaluation
* Extend to high-order ngrams to capture intra-word contexts

## Requirements
The code is tested on Ubuntu 16.04 with the following components:

### Software

* Anaconda Python 3.6
* Keras 2.2.4 using TensorFlow GPU 1.12.0 backend
* CUDA 9.1 with CuDNN 7.1

### Hardware

* Intel Xeon CPU with 32 cores
* 64GB of system RAM
* NVIDIA GeForce GTX TITAN X GPU with 12GB of VRAM

### Acknowledgment

The idea is inspired by this [blog post](https://machinelearnings.co/deep-spelling-9ffef96a24f6), with several enhancements in this implementation, such as teacher forcing, that result in faster convergence and better performance. The words dataset comes from [https://github.com/dwyl/english-words](https://github.com/dwyl/english-words).