import numpy as np
import matplotlib.pyplot as plt

from genzyme.models import modelFactory

P = np.ones((20,20))
pi_0 = np.zeros(20)
nonz_idx = [0, 5, 9, 15] #Serine Leucine Glycine Alanine
pi_0[nonz_idx] = 1/len(nonz_idx)

# construct transition matrix
p = 0.999
P *= (1-p)/(len(P)-1)
for i in range(20):
    P[i,i] = p

assert np.all(np.round(np.sum(P, 1), 3)==1)

# Perform the random walk
n_steps = 300
pi_curr = pi_0
pi_evo = []
entropy = []
for i in range(n_steps):
    pi_evo.append(pi_curr)
    entropy.append(-sum(pi_curr[pi_curr != 0] * np.log(pi_curr[pi_curr != 0])))
    pi_curr = pi_curr @ P

pi_evo.reverse()
entropy.reverse()

plt.scatter(range(len(entropy)), entropy)
plt.xlabel("step")
plt.ylabel("entropy")
plt.savefig("mc_test.png")



model = modelFactory("random")
model.frequency = np.array(pi_evo)
model.generate(10000, "./gen_data/scheduled_entropy/", keep_in_memory=False)





