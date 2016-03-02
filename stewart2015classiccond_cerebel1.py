import nengo

D = 3
N = D*50

model = nengo.Network(label="My Network")
with model:

    def us_stim(t):
        t = t % 3
        if 0.9 < t< 1: return [1, 0, 0]
        if 1.9 < t< 2: return [0, 1, 0]
        if 2.9 < t< 3: return [0, 0, 1]
        return [0, 0, 0]
    us_stim = nengo.Node(us_stim)

    def cs_stim(t):
        t = t % 3
        if 0.7 < t< 1: return [0.7, 0, 0.7]
        if 1.7 < t< 2: return [0.6, 0.6, 0.6]
        if 2.7 < t< 3: return [0, 1, 0]
        return [0, 0, 0]
    cs_stim = nengo.Node(cs_stim)

    us = nengo.Ensemble(N, D)
    cs = nengo.Ensemble(N*2, D*2)

    nengo.Connection(us_stim, us[:D])
    nengo.Connection(cs_stim, cs[:D])
    nengo.Connection(cs[:D], cs[D:], synapse=0.2)


    act = nengo.Ensemble(N, D)

    nengo.Connection(us, act)

    predict = nengo.Ensemble(N, D)

    learn_conn = nengo.Connection(cs, predict, function=lambda x: [0]*D,
                        learning_rule_type=nengo.PES(learning_rate=3e-5))

    error = nengo.Ensemble(N, D)
    nengo.Connection(act, error)
    nengo.Connection(predict, error, transform=-1, synapse=0.1)

    error_conn = nengo.Connection(error, learn_conn.learning_rule)


    motor = nengo.Ensemble(N, D)
    nengo.Connection(act, motor)
    nengo.Connection(predict, motor)
    
    import time
    slowdown = nengo.Node(lambda t: time.sleep(0.005))
