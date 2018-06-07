import tensorflow as tf

class MultiGatedGRUCell(object):
    def __init__(self, inp_size, context_size, num_units, keys, scope_name='MultiGatedGRUCell'):
        self.inp_size = inp_size
        self.num_units = num_units
        self.context_size = context_size
        self.keys = keys
        self.num_tasks = num_tasks = int(self.keys.shape[0])
        self.key_size = key_size = int(self.keys.shape[1])
        xav_init = tf.contrib.layers.xavier_initializer

        with tf.variable_scope(scope_name):
            self.W_0 = tf.get_variable('W0', shape=[num_units, 2 * num_units], )#initializer=xav_init())
            self.U_0 = tf.get_variable('U0', shape=[inp_size, 2 * num_units], )#initializer=xav_init())
            self.V_0 = tf.get_variable('V0', shape=[key_size, 2 * num_units], )#initializer=xav_init())
            self.b_0 = tf.get_variable('b0', shape=[2 * num_units], initializer=tf.constant_initializer(0.))
    
            self.W_1 = tf.get_variable('W1', shape=[num_units, num_units], )#initializer=xav_init())
            self.U_1 = tf.get_variable('U1', shape=[inp_size, num_units], )#initializer=xav_init())
            self.V_1 = tf.get_variable('V1', shape=[key_size, num_units], )#initializer=xav_init())
            self.b_1 = tf.get_variable('b1', shape=[num_units], initializer=tf.constant_initializer(0.))
    
            self.W_g = tf.get_variable('Wg', shape=[num_units + context_size, num_units], )#initializer=xav_init())
            self.U_g = tf.get_variable('Ug', shape=[inp_size, num_units], )#initializer=xav_init())
            self.V_g = tf.get_variable('Vg', shape=[key_size, num_units], )#initializer=xav_init())
            self.b_g = tf.get_variable('bg', shape=[num_units], initializer=tf.constant_initializer(0.))
            self.u_g = tf.get_variable('ug', shape=[num_units], )#initializer=xav_init())
    
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

        res = tf.scan(self._step,
                elems=[inp, ctx, msk],
                initializer=initial_state,
                )
        
        return res

    def zero_state(self, batch_size):
        init_states = [(tf.zeros([batch_size, self.num_units]),
                tf.zeros([batch_size, 1]),) for _ in range(self.num_tasks)]
        return init_states

    def _gate_step(self, states, inputs):
        h_tm1, g_tm1 = states
        x_t, c_t, m_t, k_t = inputs
        
        g = tf.matmul(tf.concat([h_tm1, c_t], axis=1), self.W_g)
        g += tf.matmul(x_t, self.U_g) + tf.matmul(k_t[None,:], self.V_g) + self.b_g
        g = tf.reduce_sum(g * self.u_g, axis=1, keep_dims=True)
        g = tf.sigmoid(g)

        h = self._gru_step(h_tm1, [x_t, c_t, m_t])
        h = g * h + (1 - g) * h_tm1
        
        h = tf.where(tf.equal(m_t, 1), h, h_tm1)
        return h, g

    def _gru_step(self, h_tm1, inputs):
        x_t, c_t, m_t = inputs
        z, r = tf.split(tf.matmul(x_t, self.U_0) + tf.matmul(h_tm1, self.W_0) + self.b_0,
                num_or_size_splits=2,
                axis=1)
        z, r = tf.sigmoid(z), tf.sigmoid(r)
        h = tf.tanh(tf.matmul(x_t, self.U_1) + tf.matmul(r * h_tm1, self.W_1) + self.b_1)
        h = (1 - z) * h + z * h_tm1
        return h

    def _step(self, states_tm1, inputs):
        new_states = []
        for i in range(self.num_tasks):
            states_i = self._gate_step(states_tm1[i], inputs+[self.keys[i]])
            new_states.append(states_i)
        return new_states

    def _call__(self, inputs, states_tm1):
        states = self._step(states_tm1, inputs)
        return states, states

if __name__ == '__main__':
    keys = tf.ones([7,8])
    net = MultiGatedGRUCell(3,4,5, keys)
    inp = tf.zeros([10, 11, 3])
    ctx = tf.zeros([10, 11, 4])
    msk = tf.ones([10, 11])
    y = net.forward(inp, ctx, msk, return_final=False)
    print(type(y[0]))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run(y)
    import numpy as np
    print(res)
    for i in res:
        for j in i:
            print(j.shape)
