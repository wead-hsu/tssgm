import tensorflow as tf

class GatedGRU(tf.contrib.rnn.RNNCell):
    def __init__(self, inp_size, num_units, scope_name='gated_gru'):
        """
        GRU implemetation. 
        Args:
            inp_size: input_size
            hidden_size: number of units of hidden variable
        """
        # intializer
        self.inp_size = inp_size
        self.num_units = num_units
        xav_init = tf.contrib.layers.xavier_initializer

        # parameters
        with tf.variable_scope(scope_name):
            self.W_0 = tf.get_variable('W0', shape=[num_units, 2 * num_units], )#initializer=xav_init())
            self.U_0 = tf.get_variable('U0', shape=[inp_size, 2 * num_units], )#initializer=xav_init())
            self.b_0 = tf.get_variable('b0', shape=[2 * num_units], initializer=tf.constant_initializer(0.))
    
            self.W_1 = tf.get_variable('W1', shape=[num_units, num_units], )
            self.U_1 = tf.get_variable('U1', shape=[inp_size, num_units], )
            self.b_1 = tf.get_variable('b1', shape=[num_units], initializer=tf.constant_initializer(0.))
    
            self.W_g0 = tf.get_variable('W_g0', shape=[inp_size + num_units, num_units], )#initializer=xav_init())
            self.b_g0 = tf.get_variable('b_g0', shape=[num_units], initializer=tf.constant_initializer(0.))
            self.W_g1 = tf.get_variable('W_g1', shape=[num_units, 1])
            self.b_g1 = tf.get_variable('b_g1', shape=[1])
    
    def forward(self, inp, msk, initial_state=None, time_major=False, return_final=False, scope='GatedGRU'):
        """ to build the graph, run forward """
        if not time_major:
            inp = tf.transpose(inp, [1, 0, 2])
            msk = tf.transpose(msk, [1, 0])
        
        # after transposition, the shape is [seqlen, batch_size, inp_size]
        batch_size = tf.shape(inp)[1]

        if initial_state is None:
            initial_state = self.zero_state(batch_size, dtype=tf.float32)

        states, gates = tf.scan(self._mask_step,
                elems=[inp, msk],
                initializer=initial_state,
                )
        
        if return_final:
            states = states[-1]
            if not time_major:
                gates = tf.transpose(gates, [1, 0, 2])
        else:
            if not time_major:
                states = tf.transpose(states, [1, 0, 2])
                gates = tf.transpose(gates, [1, 0, 2])

        return states, gates

    def zero_state(self, batch_size, dtype):
        return (tf.zeros([batch_size, self.num_units], dtype=dtype),
                tf.zeros([batch_size, 1], dtype=dtype),)
    
    @property
    def state_size(self):
        return (self.num_units, 1)

    @property
    def output_size(self):
        return (self.num_units, 1)

    def _mask_step(self, states, inputs):
        x_t, m_t = inputs
        s, g = self._gate_step(states, x_t)
        s = tf.where(tf.equal(m_t, 1), s, states[0])
        return s, g

    def _gate_step(self, states, x_t):
        prev_s, prev_g = states
        
        g = tf.tanh(tf.matmul(tf.concat([prev_s, x_t], axis=1), self.W_g0) + self.b_g0)
        g = tf.matmul(g, self.W_g1)+self.b_g1
        g = tf.sigmoid(g+1)

        s = self._gru_step(prev_s, x_t)
        s = g * s + (1 - g) * prev_s
        #s = g * s + prev_s # similar performance, difficult to train
        
        return s, g

    def _gru_step(self, prev_s, x_t):
        z, r = tf.split(tf.matmul(x_t, self.U_0) + tf.matmul(prev_s, self.W_0) + self.b_0,
                num_or_size_splits=2,
                axis=1)
        z, r = tf.sigmoid(z), tf.sigmoid(r)
        # slow
        #z = tf.sigmoid(tf.matmul(x_t, self.U[0]) + tf.matmul(prev_s, self.W[0]) + self.b[0]) 
        #r = tf.sigmoid(tf.matmul(x_t, self.U[1]) + tf.matmul(prev_s, self.W[1]) + self.b[1]) 
        h = tf.tanh(tf.matmul(x_t, self.U_1) + tf.matmul(r * prev_s, self.W_1) + self.b_1)
        s = (1 - z) * h + z * prev_s
        return s

    def __call__(self, inputs, states):
        """ change order """
        new_states = self._gate_step(states, inputs)
        return new_states, new_states

if __name__ == '__main__':
    net = GatedGRU(3,4)
    inp = tf.zeros([10, 11, 3])
    msk = tf.ones([10, 11])
    y = net.forward(inp, msk, return_final=False)
    print(type(y[0]))

    sess = tf.Session()
    #sess.run(tf.global_variable_initializer())
    sess.run(tf.global_variables_initializer())
    #print(sess.run(y))
    res = sess.run(y)
    import numpy as np
    print(res)
    print(np.shape(res[1]))


    a, b = tf.nn.dynamic_rnn(cell=net,
                        inputs=inp,
                        dtype=tf.float32,
                        #initial_state=cell.zero_state(tf.shape(emb_inp)[0], dtype=tf.float32),
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
    r0, r1 = sess.run([a, b])
    for x in r0:
        print(x.shape)
    for x in r1:
        print(x.shape)
