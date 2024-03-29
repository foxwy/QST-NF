import itertools as it
import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))


class LatentAttention():
    def __init__(self, data_train, n_sample=1000, K=4, n_qubits=6, latent_rep_size=100, gru_hidden=100, N_epoch=100, decoder='TimeDistributed_mol', Nsamples=0):
        """ decoder='TimeDistributed'   is the simplest time distributed decoder feeding z z z as input to the decoder
        at every time step
        decoder='TimeDistributed_mol' during training, input of the decoder is concatenating [z,symbol_t]; during
        generation input should be [z,predicted_symbol_t]
        # where predicted molecule is a sample of the probability vector (or a greedy sample where we convert
        predicted_molecule_t to one hot vector of the most likely symbol )
        vae=0 autoencoder vae=1 variational autoencoder
        """
        if Nsamples > 0:
            if not os.path.exists("samples"):
                os.makedirs("samples")

        tf.reset_default_graph()

        if n_sample < 1000:
            self.batchsize = n_sample
        else:
            self.batchsize = 1000
        self.latent_rep_size = latent_rep_size
        self.gru_hidden = gru_hidden
        self.Nsamples = Nsamples
        self.N_epoch = N_epoch

        # # loading data sets
        self.data_train = data_train

        self.charset_length = K  # len(self.charset)
        self.max_length = n_qubits  # size of the molecule representation
        self.n_samples = self.data_train.shape[0]
        self.POVM_meas = tf.placeholder(tf.float32, [None, self.max_length * self.charset_length])
        self.molecules = tf.reshape(self.POVM_meas, [tf.shape(self.POVM_meas)[0], self.max_length, self.charset_length])

        self.generated_molecules, _ = self.generation(decoder, self.molecules)  # molecules from sampled z

        self.sample_onehot, self.logP = self.generation(decoder)
        self.sample_RNN = tf.argmax(self.sample_onehot, axis=2)

        self.generation_loss = -tf.reduce_sum(self.molecules * tf.log(1e-15 + self.generated_molecules), [1, 2])  # cross entropy

        self.cost = tf.reduce_mean(self.generation_loss)  # overall cost function

        tf.summary.scalar('Cost_function', self.cost)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.opt = tf.train.AdamOptimizer(0.001)
            self.gradients, self.variables = zip(*self.opt.compute_gradients(self.cost))
            for grad, var in zip(self.gradients, self.variables):
                tf.summary.histogram(var.name + '/gradient', grad)

            self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
            self.optimizer = self.opt.apply_gradients(zip(self.gradients, self.variables))

    def generation(self, decoder, molecules=None):

        if decoder == 'TimeDistributed':
            logP = tf.zeros([self.batchsize], dtype=tf.float32)
            with tf.variable_scope("generation"):
                zt = tf.get_variable('z', shape=[1, self.latent_rep_size])
                z = tf.tile(zt, [self.batchsize, 1])
                z_matrix = tf.layers.dense(inputs=z, units=self.latent_rep_size, activation=tf.nn.relu, name="fc_GEN")
                # fully connected layer applied to latent vector z
                h1 = tf.stack([z_matrix for _ in range(self.max_length)], axis=1)  # equivalent of repeatvector in keras.
                # z is repeated and meant to be the input to the unfolded recurrent nn
                # GRU Recurrent NN
                # cell=tf.contrib.rnn.GRUCell(self.gru_hidden) # GRU cell
                # cell=tf.contrib.rnn.MultiRNNCell([cell for _ in range(3)], state_is_tuple=False) # stacking 3 GRUs
                cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(self.gru_hidden) for _ in range(3)],
                                                   state_is_tuple=False)  # stacking 3 GRUs
                outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=h1, dtype=tf.float32)  # apply the cells to input

                # applying a fully connected layer with a softmax non linearity on top to each time output separately
                h2 = tf.reshape(tf.layers.dense(tf.reshape(outputs, [self.batchsize * self.max_length, self.gru_hidden]),
                                                units=self.charset_length, activation=tf.nn.softmax, name="output"),
                                [self.batchsize, self.max_length, self.charset_length])

        elif decoder == 'TimeDistributed_mol':
            if molecules == None:  # sample
                with tf.variable_scope("generation", reuse=True):
                    zt = tf.get_variable('z', shape=[1, self.latent_rep_size])
                    z = tf.tile(zt, [self.batchsize, 1])
                    # fully connected layer applied to latent vector z
                    z_matrix = tf.layers.dense(inputs=z, units=self.latent_rep_size, activation=tf.nn.relu, name="fc_GEN")
#
                # An atom of zeros. This is part of the (t=0) input of the RNN: input is concat([zmol, z_matrix]);
                y0 = tf.zeros([self.batchsize, self.charset_length], dtype=tf.float32)

                logP = tf.zeros([self.batchsize], dtype=tf.float32)

                # concatenates z_matrix to the atom of zeros as the first input of the RNN
                y_z = tf.concat([y0, z_matrix], axis=1)

                # initial state of the RNN during the unrolling
                s0 = tf.zeros([self.batchsize, 3 * self.gru_hidden], dtype=tf.float32)

                # output tensor unrolling
                h2 = tf.zeros([self.batchsize, 1, self.charset_length], dtype=tf.float32)

                i0 = tf.constant(0)  # counter

                # preparing the unrolling loop
                # condition
                time_steps = self.max_length

                def c(i, s0, h2, y_z, logP):
                    return i < time_steps
#
                # body

                def b(i, s0, h2, y_z, logP):
                    with tf.variable_scope("generation", reuse=True):
                        # GRU Recurrent NN
                        with tf.variable_scope("rnn"):
                            # cell=tf.contrib.rnn.GRUCell(self.gru_hidden) # GRU cell
                            # cell=tf.contrib.rnn.MultiRNNCell([cell for _ in range(3)], state_is_tuple=False) # stacking 3 GRUs
                            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(self.gru_hidden) for _ in range(3)], state_is_tuple=False)  # stacking 3 GRUs
                            outputs, state = cell(y_z, s0)

                        # apply a softmax layer to the output of the RNN; dlayer is probability accross the different symbols in the dictionary
                        dlayer = tf.layers.dense(outputs, units=self.charset_length, activation=tf.nn.softmax, name="output")  # returns probability

                        # Two options. 1 take the symbol with the largest probability using argmax or sample the symbols P(symbol| [sampled symbol i, z],[sampled symbol i-1, z].... [sampled symbol 0, z])

                        # pick the most likely symbol as input for the next loop step
                        # amax=tf.argmax( dlayer,axis=1 )
                        # dlayer=tf.one_hot(amax,depth=self.charset_length,axis=1, dtype=tf.float32)

                        # samples dlayer probabilities
                        logits = tf.log(dlayer)

                        samples = tf.reshape(tf.multinomial(logits, 1), [-1, ])  # reshape to a vector of shape=(self.batchsize,)l  multinomial returns one integer sample per element in the batch
                        dlayer = tf.one_hot(samples, depth=self.charset_length, axis=1, dtype=tf.float32)  # onehot symbol

                        logP = logP + tf.reduce_sum(dlayer * logits, [1])

                    return [i + 1, state, tf.concat([h2, tf.reshape(dlayer, [self.batchsize, 1, self.charset_length])], axis=1), tf.concat([dlayer, z_matrix], axis=1), logP]

                # unrolling using tf.while_loop
                ii, s0, h2, y_z, logP = tf.while_loop(
                    c, b, loop_vars=[i0, s0, h2, y_z, logP],
                    shape_invariants=[i0.get_shape(), s0.get_shape(), tf.TensorShape(
                        [self.batchsize, None, self.charset_length]), y_z.get_shape(), logP.get_shape()])  # shape invariants required
                # since h2 increases size as rnn unrolls

                h2 = tf.slice(h2, [0, 1, 0], [-1, -1, -1])  # cuts the initial zeros that inserted to start the while_loop

            else:  # train

                with tf.variable_scope("generation", reuse=None):

                    # #fully connected layer applied to latent vector z
                    zt = tf.get_variable('z', shape=[1, self.latent_rep_size])
                    z = tf.tile(zt, [self.batchsize, 1])

                    z_matrix = tf.layers.dense(inputs=z, units=self.latent_rep_size, activation=tf.nn.relu, name="fc_GEN")

                    # An atom of zeros. This is part of the (t=0) input of the RNN: input is concat([zmol, z_matrix]); inputs at time t=1,2,.. maxlength-1  are concat([molecule[:,t,:], z_matrix])
                    zmol = tf.zeros([self.batchsize, 1, self.charset_length], dtype=tf.float32)

                    # creates the new "molecule" input, i.e., zmol,mol_1,mol_2,...,mol_{maxlength-1}
                    mol = tf.slice(tf.concat([zmol, molecules], axis=1), [0, 0, 0], [self.batchsize, self.max_length,
                                                                                     self.charset_length])  # creates
                    # new "molecule" input, i.e., zmol,mol_1,mol_2,...,mol_{maxlength-1}

                    # concatenates z_matrix to all the different parts of the molecule to create the extended input of
                    # the RNN
                    h1 = tf.stack([tf.concat([t, z_matrix], axis=1) for t in tf.unstack(mol, axis=1)], axis=1)

                    # GRU Recurrent NN
                    # cell = tf.contrib.rnn.GRUCell(self.gru_hidden)  # GRU cell
                    # cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(3)], state_is_tuple=False)  # stacking 3 GRUs
                    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(self.gru_hidden) for _ in range(3)], state_is_tuple=False)  # stacking 3 GRUs

                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=h1, dtype=tf.float32)  # cells to the input
                    # applying a fully connected layer with a softmax non linearity on top to each time output separately
                    in_dense = tf.reshape(outputs, [self.batchsize * self.max_length, self.gru_hidden])
                    dlayer = tf.layers.dense(in_dense, units=self.charset_length, activation=tf.nn.softmax, name="output")
                    h2 = tf.reshape(dlayer, [self.batchsize, self.max_length, self.charset_length])
                    logP = tf.zeros([self.batchsize], dtype=tf.float32)

        return h2, logP

    def sample(self, sess):
        sa, llpp = sess.run([self.sample_RNN, self.logP])
        llpp = np.reshape(llpp, [-1, 1])
        P = np.exp(llpp)

        return [sa, P]

    def train(self):

        # train
        bcount = 0
        ept = np.random.permutation(self.data_train)

        # for small systems load the exact probability P
        if self.max_length < 1:
            P = np.loadtxt('./data/exact_P.txt', dtype=np.float32)
            Prob = np.reshape(P, [1, -1])

            # builds the entire hilbert space, so use only for small systems
            xm = np.reshape(np.eye(self.charset_length)[(np.array(list(it.product(range(self.charset_length),
                                                                                  repeat=self.max_length))))], [self.charset_length**self.max_length,
                                                                                                                self.max_length * self.charset_length])

            eps = 10e-160

            basis_s = self.charset_length**self.max_length  # size of the POVM state space
            n_calls = basis_s / self.batchsize
            rem = basis_s - n_calls * self.batchsize
            ext = self.batchsize - rem

        saver = tf.train.Saver(max_to_keep=2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #summary_write = tf.summary.FileWriter('logs/train_log', sess.graph)

            nstepspe = int(self.n_samples / self.batchsize)  # number of gradient updates per epoch

            rlist = range(int(self.n_samples / self.batchsize))
            lastNumber = rlist[-1]

            counter = 0
            for epoch in range(self.N_epoch):
                for idx in range(nstepspe):
                    # batch = self.mnist.train.next_batch(self.batchsize)[0]
                    if bcount * self.batchsize + self.batchsize >= self.n_samples:
                        bcount = 0
                        ept = np.random.permutation(self.data_train)

                    batch = ept[bcount * self.batchsize: bcount * self.batchsize + self.batchsize, :]
                    bcount = bcount + 1

                    # print "step and alpha valule ", epoch, counter, alpha
                    _, gen_loss = sess.run((self.optimizer, self.generation_loss),
                                           feed_dict={self.POVM_meas: batch})
                    # print np.mean(gen_loss)
                    counter = counter + 1

                    # print cost every at the end of each epoch
                    if idx == lastNumber:

                        # evaluation of the model
                        if self.max_length < 1:
                            # compute the probability of the entire basis set
                            k = 0
                            batch = xm[k * self.batchsize: k * self.batchsize + self.batchsize, :]
                            lnPr = sess.run(self.generation_loss, feed_dict={self.POVM_meas: batch})
                            lnPr = np.reshape(lnPr, [-1, 1])

                            for k in range(1, n_calls):
                                batch = xm[k * self.batchsize: k * self.batchsize + self.batchsize, :]
                                lnPrn = sess.run(self.generation_loss, feed_dict={self.POVM_meas: batch})
                                lnPrn = np.reshape(lnPrn, [-1, 1])
                                lnPr = np.vstack((lnPr, lnPrn))
                                print(lnPr.shape)
                            last_batch = np.zeros((self.batchsize, self.charset_length * self.max_length))
                            last_batch[0:rem, :] = xm[n_calls * self.batchsize:, :]
                            lnPrn = sess.run(self.generation_loss, feed_dict={self.POVM_meas: last_batch})
                            lnPrn = np.reshape(lnPrn, [-1, 1])
                            lnPr = np.vstack((lnPr, lnPrn[0:rem]))

                            # exact KL divergence
                            KL = -np.sum(-np.reshape(P, [-1, 1]) * lnPr - np.reshape(P, [-1, 1]) * np.log(np.reshape(P, [-1, 1]) + eps))

                            Prob = np.vstack((Prob, np.reshape(np.exp(-lnPr), [1, -1])))
                            np.savetxt('Prob.txt', Prob)
                            print("epoch KL ", epoch, KL)

                        if (epoch + 1) % 20 == 0:
                            print("epoch  loss ", epoch, np.mean(gen_loss))

                        # Obtaining samples from the model at epoch
                        if self.Nsamples != 0 and (epoch + 1) % self.N_epoch == 0:  # if epoch == self.N_epoch - 1 or epoch == self.N_epoch // 2 - 1 or epoch == self.N_epoch // 4 - 1:
                            Ncalls = int(self.Nsamples / self.batchsize)
                            params = [sess] * Ncalls
                            cpu_counts = mp.cpu_count()
                            if Ncalls < cpu_counts:
                                cpu_counts = Ncalls
                            with Pool(cpu_counts) as pool:
                                results = pool.map(self.sample, params)

                            pbar = tqdm(range(Ncalls - 1))
                            samples, P = results[0][0], results[0][1]
                            for k in pbar:
                                sa, lP = results[k+1][0], results[k+1][1]
                                samples = np.vstack((samples, sa))
                                P = np.vstack((P, lP))

                                pbar.set_description("sampling data from RNN")
                            pbar.close()

                            #np.savetxt('./samples/samples_' + str(epoch + 1) + '.txt', samples)
                            np.savetxt('./samples/P_' + str(epoch + 1) + '.txt', P)

                            # sample time
                            self.batchsize = 1
                            times = []
                            for k in range(10):
                                t0 = time.time()
                                sa, llpp = sess.run([self.sample_RNN, self.logP])
                                t1 = time.time()
                                times.append(t1 - t0)
                            print('Times mean: {:.6f}  std: {:.6f}'.format(np.mean(times), np.std(times)))
                            np.savetxt('./samples/times_' + str(epoch + 1) + '.txt', times)


if __name__ == '__main__':
    K = 4
    N_epoch = 50
    povm = 'Pauli_4'

    with open('results.txt', 'w') as f:
        for N_p in [0, 0.5]:
            f.write("---%s\n" % N_p)
            f.flush()
            for N_q in [10, 20, 50, 80]:
                f.write("%s\n" % N_q)
                f.flush()

                N_s = N_q * 10**3
                if N_p == 0:
                    trainFileName = filepath + '/datasets/data/GHZ_MPS_' + povm + '_train_N' + str(N_q) + '.txt'
                else:
                    trainFileName = filepath + '/datasets/data/GHZ_MPS_' + povm + '_train_N_05_' + str(N_q) + '.txt'

                data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
                data_test = np.loadtxt(trainFileName)[N_s:].astype(int)
                N_sample = len(data_test)

                model = LatentAttention(data_train=data_train, n_sample=N_s, K=K, n_qubits=N_q, N_epoch=N_epoch, decoder='TimeDistributed_mol', Nsamples=N_sample)
                model.train()

                P = np.loadtxt('samples/P_' + str(N_epoch) + '.txt')
                times = np.loadtxt('samples/times_' + str(N_epoch) + '.txt')
                nll_test = -np.mean(np.log(P.reshape(-1)))
                print('nll_test:', nll_test)

                f.write('Times mean: {:.6f} std: {:.6f}\n'.format(np.mean(times), np.std(times)))
                f.write('nll test: {:.6f}\n\n'.format(nll_test))
                f.flush()