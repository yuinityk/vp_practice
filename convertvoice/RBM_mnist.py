# -*- coding: utf-8 -*-
import time
import sys
import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

class RBM(object):

    def __init__(self, input, n_visible, n_hidden, np_rng):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        a = 1. / n_visible
        self.W = np.array(np_rng.uniform(low=-a, high=a, size=(n_visible, n_hidden)))

        self.np_rng = np_rng
        self.input = input
        self.hbias = np.zeros(n_hidden)
        self.vbias = np.zeros(n_visible)

    def cd(self, eta=0.1, k=1):
        ph_mean, ph_sample = self.sample_h(self.input)
        chain_start = ph_sample

        for step in range(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_samp(chain_start) 
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_samp(nh_samples)

        self.W     += eta * (np.dot(self.input.T, ph_mean) - np.dot(nv_samples.T, nh_means))
        self.vbias += eta * np.mean(self.input - nv_samples, axis=0)
        self.hbias += eta * np.mean(ph_mean - nh_means, axis=0)

    def sample_h(self, v0_sample):
        pre_sigmoid_activation = np.dot(v0_sample, self.W) + self.hbias
        h1_mean = sigmoid(pre_sigmoid_activation)
        h1_sample = self.np_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return [h1_mean, h1_sample]


    def sample_v(self, h0_sample):
        pre_sigmoid_activation = np.dot(h0_sample, self.W.T) + self.vbias
        v1_mean = sigmoid(pre_sigmoid_activation)
        v1_sample = self.np_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        return [v1_mean, v1_sample]

    def gibbs_samp(self, h0_sample):
        v1_mean, v1_sample = self.sample_v(h0_sample)
        h1_mean, h1_sample = self.sample_h(v1_sample)
        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def get_cross_entropy(self):
        pre_sigmoid_activation_h = np.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)
        cross_entropy =  - np.mean(
            np.sum(self.input * np.log(sigmoid_activation_v) + (1 - self.input) * np.log(1 - sigmoid_activation_v), axis=1))
        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        mRecon = sigmoid(np.dot(h, self.W.T) + self.vbias)
        sRecon = self.np_rng.binomial(size=mRecon.shape, n=1, p=mRecon)
        return mRecon, sRecon

class GBRBM(RBM):
    def __init__(self, input, n_visible, n_hidden, np_rng):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        a = 1. / n_visible
        self.W = np.array(np_rng.uniform(low=-a, high=a, size=(n_visible, n_hidden)))

        self.np_rng = np_rng
        self.input = input
        self.hbias = np.zeros(n_hidden)
        self.vbias = np.zeros(n_visible)
        self.var = np.ones(n_visible) #

    def cd(self, eta=0.1, k=1):
        ph_mean, ph_sample = self.sample_h(self.input)
        chain_start = ph_sample

        for step in range(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_samp(chain_start)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_samp(nh_samples)

                #############
                self.W += eta * ()

    def sample_v(self,v0_sample):
        pass

mnist = fetch_mldata('MNIST original', data_home=".")
thre = 200
data = mnist.data > thre
np.random.seed(1)
d_learn = data[:60000].astype(int)
np.random.shuffle(d_learn)
d_learn = d_learn[:50].astype(int)
d_test = data[60000:70000].astype(int)
np.random.shuffle(d_test)


rbm = RBM(input=d_learn,n_visible=len(d_learn[0]), n_hidden=500, np_rng=np.random.RandomState(123))
# train
training_epochs=1000
learning_rate=0.02441
k=1
st = time.time()

for epoch in range(training_epochs):
    rbm.cd(eta=learning_rate, k=k)
    cost = rbm.get_cross_entropy()
    print 'Training epoch: %d, cost: ' % epoch, cost
print time.time()-st
mRecon, sRecon = rbm.reconstruct(d_test)

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(rbm.W.T[i].reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

for i in range(50):
    plt.subplot(10,5,i+1)
    if i<25:
        plt.imshow(d_test[i].reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")
    else:
        plt.imshow(sRecon[i-25].reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

mRecon, sRecon = rbm.reconstruct(d_learn)
for i in range(50):
    plt.subplot(10,5,i+1)
    if i<25:
        plt.imshow(d_learn[i].reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")
    else:
        plt.imshow(sRecon[i-25].reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
