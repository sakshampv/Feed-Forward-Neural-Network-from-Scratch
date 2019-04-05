import time
import numpy as np
import matplotlib.pyplot as plt
from builtins import range
import numpy as np
import pandas as pd
l_r = 1e-3

def step_rel_for(x, w, b):
   
    a, fc_misc = step_forw(x, w, b)
    out, relu_misc = rel_for(a)
    misc = (fc_misc, relu_misc)
    return out, misc


def step_rel_bac(outward_der, misc):
    
    fc_misc, relu_misc = misc
    da = rel_bac(outward_der, relu_misc)
    delta_x, delta_w, delta_b = step_bac(da, fc_misc)
    return delta_x, delta_w, delta_b


def calc_relative_error(x, y):
  abs_val = abs(x-y)
  mx = np.max(abs_val)
  if y==0:
      return mx / (1e-9)
  else:
      alph= np.abs(x)
      beta = np.abs(y)
      return mx / (alph+beta)


def step_forw(x, w, b):
   
    out = x.reshape(x.shape[0], -1).dot(w) + b
    misc = (x, w, b)
    return out, misc


def step_bac(outward_der, misc):
   
    x, w, b = misc

    delta_x = outward_der.dot(w.T).reshape(x.shape)
    delta_w = x.reshape(x.shape[0], -1).T.dot(outward_der)
    delta_b = np.sum(outward_der, axis=0)
    return delta_x, delta_w, delta_b


def rel_for(x):
    out = np.maximum(0, x)
    misc = x
    return out, misc

def rel_bac(outward_der, misc):
    delta_x, x = None, misc
    delta_x = (x > 0) * outward_der
    return delta_x

def softmax_loss(x, y):
   
    shift = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shift), axis=1, keepdims=True)
    log_probs = shift - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]

    loss = -np.sum(log_probs[np.arange(N), y]) / N
    delta_x = probs.copy()
    delta_x[np.arange(N), y] -= 1
    delta_x /= N
    return loss, delta_x



class My_model(object):
   

    def __init__(self, hid_lyr_dim, x_dim=784, out_nums=10,
                lamda=0.0,
                 w_scale=1e-2, seed=None):
       
       
        self.lamda = lamda
        self.num_layers = 1 + len(hid_lyr_dim)
        self.parameters = {}
        
        dims = np.hstack((x_dim, hid_lyr_dim, out_nums))

        for i in range(self.num_layers):
            self.parameters['wts%d' % (i + 1)] = w_scale * np.random.randn(dims[i], dims[i+1])
            self.parameters['b%d' % (i + 1)] = np.zeros(dims[i+1])
            
    def compute_acc(self, X, y, n_samples=None, b_s=100):
       
        N = X.shape[0]
        if n_samples is not None and N > n_samples:
            bits = np.random.choice(N, n_samples)
            N = n_samples
            X = X[bits]
            y = y[bits]
        num_batches = N // b_s
        if N % b_s != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * b_s
            end = (i + 1) * b_s
            res = self.loss(X[start:end])
            y_pred.append(np.argmax(res, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self,X_tr, Y_tr, Xvl, Yvl,n_epch,b_s,learning_rate_decay, n_tr_samples, num_val_samples ):
        epoch = 0
        global l_r
        n_tr = X_tr.shape[0]
        it_per_epoch = max(n_tr //b_s, 1)
        n_it = n_epch * it_per_epoch

        for t in range(n_it):
            n = X_tr.shape[0]
            bits = np.random.choice(n, b_s)
            X_batch = X_tr[bits]
            y_batch = Y_tr[bits]
            loss, gradients = self.loss(X_batch, y_batch)
            for p, w in self.parameters.items():          
              self.parameters[p] =w - l_r*gradients[p]
              epoch_end = (t + 1) % it_per_epoch == 0
              if epoch_end:
                epoch += 1
                l_r *= learning_rate_decay
                
            first_it = (t == 0)
            last_it = (t == n_it - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.compute_acc(X_tr, Y_tr,
                    n_samples=n_tr_samples)
                val_acc = self.compute_acc(Xvl,Yvl,
                    n_samples=num_val_samples)
                print('Train_accuracy',train_acc)
                print('Val_accuracy',val_acc)

    def loss(self, X, y=None):
       
      
        
        hidden_num = self.num_layers - 1
        res = X
        misc_history = []
        l2_loss = 0
        for i in range(hidden_num):
            res, misc = step_rel_for(res, self.parameters['wts%d' % (i + 1)],
                                                            self.parameters['b%d' % (i + 1)])
            misc_history.append(misc)
            
            l2_loss += np.sum(self.parameters['wts%d' % (i + 1)]*self.parameters['wts%d' % (i + 1)])
        i += 1
        res, misc = step_forw(res, self.parameters['wts%d' % (i + 1)],
                                               self.parameters['b%d' % (i + 1)])
        misc_history.append(misc)
        l2_loss += np.sum(self.parameters['wts%d' % (i + 1)]*self.parameters['wts%d' % (i + 1)])
        l2_loss *= 0.5 * self.lamda

        
        if y is None:
            return res

        loss, gradients = 0.0, {}
       

        loss, outward_der = softmax_loss(res, y)
        loss += l2_loss

        outward_der, gradients['wts%d' % (i + 1)], gradients['b%d' % (i + 1)] = step_bac(outward_der, misc_history.pop())
        gradients['wts%d' % (i + 1)] += self.lamda * self.parameters['wts%d' % (i + 1)]
        i -= 1
        while i >= 0:
            
            outward_der, gradients['wts%d' % (i + 1)], gradients['b%d' % (i + 1)] = step_rel_bac(outward_der, misc_history.pop())
            gradients['wts%d' % (i + 1)] += self.lamda * self.parameters['wts%d' % (i + 1)]
            i -= 1

       
        return loss, gradients



data  = pd.read_csv("mnist_train.csv")

ww= data.as_matrix(columns = None)


X_tr= np.asarray(ww[:6000,1:])
Y_tr= np.asarray(data['6'][:6000])
Xvl= np.asarray(ww[6000:6999,1:])
Yvl= np.asarray(data['6'][6000:6999])

w_scale = 0.01
l_r = 1e-3
model = My_model([400, 400],w_scale=w_scale)
solver = model.train(X_tr,Y_tr,Xvl, Yvl, 150 , 100 , 1.0 ,X_tr.shape[0], Xvl.shape[0])

# X_tr, Y_tr, Xvl, Yvl, epch_num,batch_size, lr_decay_rate, num_train, num_val

data = pd.read_csv("mnist_test.csv", header = None)


ww= data.as_matrix(columns = None)




X_test= np.asarray(ww)



y = np.argmax(model.loss(X_test), axis=1)

xx = np.arange(3000) + 1

df = pd.DataFrame()
df['id'] = xx
df['label']=y

df.to_csv("submission.csv", index = False)



