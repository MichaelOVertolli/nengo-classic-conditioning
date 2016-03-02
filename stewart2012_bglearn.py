
import nengo as n
from numpy import ones, array, zeros

D = 2
nn = 100

model = n.Network()
with model:
    basis = ones(D)
    def R(x):
        return array([max([i, 0]) for i in x])
    def RSum(x):
        return 0.9*sum(R(x))*basis
    
    state = n.Node(zeros(D))
    #reward = n.Node(zeros(D))
    #this is a hack to remove multiple bias nodes
    bias = n.Node(basis)
    
    Q = n.Ensemble(n_neurons=nn, dimensions=D)
    D1 = n.Ensemble(n_neurons=nn, dimensions=D)
    D2 = n.Ensemble(n_neurons=nn, dimensions=D)
    vSTN = n.Ensemble(n_neurons=nn, dimensions=D)
    #SNc = n.Ensemble(n_neurons=nn, dimensions=D)
    STN = n.Ensemble(n_neurons=nn, dimensions=D)
    GPi = n.Ensemble(n_neurons=nn, dimensions=D)
    GPe = n.Ensemble(n_neurons=nn, dimensions=D)

    #l_rule = n.PES(learning_rate=1e-7)
    #l_rule2 = n.PES(learning_rate=1e-7)
    #l_rule3 = n.PES(learning_rate=1e-7)

    n.Connection(state, Q)


    D1_c = n.Connection(Q, D1, transform=1.2*basis)#, learning_rule_type=l_rule)
    D2_c = n.Connection(Q, D2, transform=0.8*basis)#, learning_rule_type=l_rule2)
    #vSTN_c = n.Connection(Q, vSTN, transform=-1*basis)#, learning_rule_type=l_rule3)
    n.Connection(Q, STN)

    #n.Connection(reward, vSTN)
    #n.Connection(vSTN, SNc)

    #n.Connection(SNc, vSTN_c.learning_rule)
    #n.Connection(SNc, D1_c.learning_rule)
    #n.Connection(SNc, D2_c.learning_rule)

    #the rest of bias hack
    n.Connection(bias, D1, transform=-0.2*basis)
    n.Connection(bias, D2, transform=-0.2*basis)
    n.Connection(bias, STN, transform=0.25*basis)
    n.Connection(bias, GPe, transform=0.2*basis)
    n.Connection(bias, GPi, transform=0.2*basis)

    n.Connection(D1, GPi, function=R, transform=-1*basis, synapse=0.008)
    n.Connection(D2, GPe, function=R, transform=-1*basis, synapse=0.008)

    n.Connection(STN, GPi, function=RSum, synapse=0.002)
    n.Connection(STN, GPe, function=RSum, synapse=0.002)

    n.Connection(GPe, STN, function=R, transform=-1*basis, synapse=0.008)
    n.Connection(GPe, GPi, function=R, transform=-0.3*basis, synapse=0.008)
