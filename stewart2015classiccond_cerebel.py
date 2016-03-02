import nengo

D = 1
N = D*50

model = nengo.Network(label="My Network")
with model:

    def us_stim(t):
        if t < 4.0: return 1
        if t < 8.0: return -0.5
        if t < 12.0: return 1
        if t < 16.0: return -0.5
        #t = t % 3
        #if 0.95 < t< 1: return [1, 0, 0]
        #if 1.9 < t< 2: return [0, 1, 0]
        #if 2.9 < t< 3: return [0, 0, 1]
        return 0
    us_stim = nengo.Node(us_stim)

    def cs_stim(t):
        if t < 4.0: return [1, 0]
        if t < 8.0: return [-0.5, 1]
        if t < 12.0: return [1, 0]
        if t < 16.0: return [-0.5, 1]
        #t = t % 3
        #if 0.7 < t< 1: return [0.7, 0, 0.7]
        #if 1.7 < t< 2: return [0.6, 0.6, 0.6]
        #if 2.7 < t< 3: return [0, 1, 0]
        return [0, 0]
    cs_stim = nengo.Node(cs_stim)

    us = nengo.Ensemble(N, D)
    cs = nengo.Ensemble(N*2, 2*2)

    nengo.Connection(us_stim, us[:D])
    nengo.Connection(cs_stim, cs[:2])
    nengo.Connection(cs[:2], cs[2:], synapse=0.2)


    act = nengo.Ensemble(N, D)

    nengo.Connection(us, act)

    predict = nengo.Ensemble(N, D)

    learn_conn1 = nengo.Connection(cs, predict, function=lambda x: 0)
    #learn_conn2 = nengo.Connection(cs, predict, function=lambda x: [0]*D)


    learn_conn1.learning_rule_type = nengo.PES(learning_rate=3e-4)
    #learn_conn2.learning_rule_type = nengo.PES(learning_rate=3e-4)

    e1_conn = nengo.Connection(act, learn_conn1.learning_rule, transform=-1)
    e2_conn = nengo.Connection(predict, learn_conn1.learning_rule, 
                                transform=1, synapse=0.1)

    error = nengo.Ensemble(N, D)
    nengo.Connection(act, error, transform=-1)
    nengo.Connection(predict, error)
    
    import time
    def slowdown(t): 
        time.sleep(0.001)
    slow = nengo.Node(slowdown)