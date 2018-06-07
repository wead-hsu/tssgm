import tensorflow as tf

class GatedGRU(object):
    def __init__(self, inp_size, context_size, num_units):
        # intializer
        self.inp_size = inp_size
        self.num_units = num_units
        self.context_size = context_size
        xav_init = tf.contrib.layers.xavier_initializer

        # parameters
        self.W_0 = tf.get_variable('W0', shape=[num_units, 2 * num_units], )#initializer=xav_init())
        self.U_0 = tf.get_variable('U0', shape=[inp_size, 2 * num_units], )#initializer=xav_init())
        self.b_0 = tf.get_variable('b0', shape=[2 * num_units], initializer=tf.constant_initializer(0.))

        self.W_1 = tf.get_variable('W1', shape=[num_units, num_units], )
        self.U_1 = tf.get_variable('U1', shape=[inp_size, num_units], )
        self.b_1 = tf.get_variable('b1', shape=[num_units], initializer=tf.constant_initializer(0.))

        self.W_g = tf.get_variable('W_g', shape=[inp_size + num_units + context_size, num_units], )#initializer=xav_init())
        self.b_g = tf.get_variable('b_g', shape=[num_units], initializer=tf.constant_initializer(0.))
        self.u_g = tf.get_variable('u_g', shape=[num_units, 1])

    def forward(self, inp, ctx, msk, initial_state=None, time_major=False, return_final=False, scope='GatedLSTM'):
        """ to build the graph, run forward """
        if not time_major:
            inp = tf.transpose(inp, [1, 0, 2])
            ctx = tf.transpose(ctx, [1, 0, 2])
            msk = tf.transpose(msk, [1, 0])
        
        # after transposition, the shape is [seqlen, batch_size, inp_size]
        batch_size = tf.shape(inp)[1]

        if initial_state is None:
            initial_state = self.zero_state(batch_size)

        states, gates = tf.scan(self._gate_step,
                elems=[inp, ctx, msk],
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

    def zero_state(self, batch_size):
        return (tf.zeros([batch_size, self.num_units]),
                tf.zeros([batch_size, 1]),)

    def _gate_step(self, states, inputs):
        prev_s, prev_g = states
        x_t, b_t, m_t = inputs
        
        g = tf.matmul(tf.tanh(tf.matmul(tf.concat([prev_s, x_t, b_t], axis=1), self.W_g) + self.b_g), self.u_g)
        g = tf.sigmoid(g)

        s = self._gru_step(prev_s, inputs)
        s = g * s + (1 - g) * prev_s
        
        s = tf.where(tf.equal(m_t, 1), s, prev_s)
        return s, g

    def _gru_step(self, prev_s, inputs):
        x_t, b_t, m_t = inputs
        #x_t = tf.concat([x_t, b_t], axis=1)
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

if __name__ == '__main__':
    net = GatedGRU(3,4,5)
    inp = tf.zeros([10, 11, 3])
    ctx = tf.zeros([10, 11, 4])
    msk = tf.ones([10, 11])
    y = net.forward(inp, ctx, msk, return_final=False)
    print(type(y[0]))

    sess = tf.Session()
    #sess.run(tf.global_variable_initializer())
    sess.run(tf.global_variables_initializer())
    #print(sess.run(y))
    res = sess.run(y)
    import numpy as np
    print(res)
    print(np.shape(res[1]))
