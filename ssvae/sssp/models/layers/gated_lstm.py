import tensorflow as tf

class GatedLSTM(object):
    def __init__(self, inp_size, num_units):
        """
        Gated LSTM is LSTM equipped with gate mechanism to ignore the
        irrelevant part of input.
        Args:
            inp_size: input_size
            hidden_size: number of units of hidden variable
        """
        # intializer
        self.inp_size = inp_size
        self.num_units = num_units
        xav_init = tf.contrib.layers.xavier_initializer

        # parameters
        self.W = tf.get_variable('W', shape=[num_units, 4 * num_units], initializer=xav_init())
        self.U = tf.get_variable('U', shape=[inp_size, 4 * num_units], initializer=xav_init())
        self.b = tf.get_variable('b', shape=[4 * num_units], initializer=tf.constant_initializer(0.))

        self.W_g = tf.get_variable('W_g', shape=[inp_size + num_units, num_units], initializer=xav_init())
        self.b_g = tf.get_variable('b_g', shape=[num_units], initializer=tf.constant_initializer(0.))
        self.u_g = tf.get_variable('u_g', shape=[num_units, 1])

    def forward(self, inp, msk, initial_state=None, time_major=False, return_final=False, scope='GatedLSTM'):
        """ to build the graph, run forward """
        if not time_major:
            inp = tf.transpose(inp, [1, 0, 2])
            msk = tf.transpose(msk, [1, 0])
        
        # after transposition, the shape is [seqlen, batch_size, inp_size]
        batch_size = tf.shape(inp)[1]

        if initial_state is None:
            initial_state = self.zero_state(batch_size)

        c, h, g = tf.scan(self._gate_step,
                elems=[inp, msk],
                initializer=initial_state,
                )

        if return_final:
            c = c[-1]
            h = c[-1]
            if not time_major:
                g = tf.transpose(g, [1, 0, 2])
        else:
            if not time_major:
                c = tf.transpose(c, [1, 0, 2])
                h = tf.transpose(h, [1, 0, 2])
                g = tf.transpose(g, [1, 0, 2])

        return c, h, g

    def zero_state(self, batch_size):
        # (c, h, g)
        return (tf.zeros([batch_size, self.num_units]),
                tf.zeros([batch_size, self.num_units]),
                tf.zeros([batch_size, 1]),)

    def _gate_step(self, states, inputs):
        prev_c, prev_h, prev_g = states
        x_t, m_t = inputs
        
        g = tf.matmul(tf.matmul(tf.concat([prev_h, x_t], axis=1), self.W_g) + self.b_g, self.u_g)
        g = tf.sigmoid(g)

        c, h = self._lstm_step((prev_c, prev_h), inputs)
        c = g * c + (1 - g) * prev_c
        h = g * h + (1 - g) * prev_h

        c = tf.where(tf.equal(m_t, 1), c, prev_c)
        h = tf.where(tf.equal(m_t, 1), h, prev_h)

        return c, h, g

    def _lstm_step(self, states, inputs):
        prev_c, prev_h = states
        x_t, m_t = inputs
        i, f, o, g = tf.split(tf.matmul(x_t, self.U) + tf.matmul(prev_h, self.W) + self.b,
                num_or_size_splits=4,
                axis=1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        c = prev_c*f + g*i
        h = tf.tanh(c) * o
        # slow
        #c = m_t[:, None] * c + (1 - m_t[:, None]) * prev_c
        #h = m_t[:, None] * h + (1 - m_t[:, None]) * prev_h
        #c = tf.where(tf.equal(m_t, 1), c, prev_c)
        #h = tf.where(tf.equal(m_t, 1), h, prev_h)
        return c, h

if __name__ == '__main__':
    net = GatedLSTM(3,4)
    inp = tf.zeros([10, 11, 3])
    msk = tf.ones([10, 11])
    y = net.forward(inp, msk, return_final=True)
    print(type(y[0]))

    sess = tf.Session()
    #sess.run(tf.global_variable_initializer())
    sess.run(tf.global_variables_initializer())
    #print(sess.run(y))
    res = sess.run(y)
    import numpy as np
    print(np.shape(res[0]))
