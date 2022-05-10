import cma
import numpy as np
import matplotlib.pyplot as plt

def fitness(x):
    x = np.array(x)
    return ((x-np.pi)**4).sum()

es = cma.CMAEvolutionStrategy(x0=[0.]*100,
                                sigma0=0.1,
                                inopts={'popsize': 100
                                })

log = []
for _ in range(200):
    solutions = es.ask()
    fit = []
    for solution in solutions:
        fit.append(fitness(solution))
    log.append([min(fit),max(fit)])
    es.tell(solutions, fit)
log = np.array(log)
plt.plot(log[:,0])
plt.plot(log[:,1])
plt.grid(); plt.xlabel('iteration'); plt.ylabel('fitness value')
plt.legend(['min fit in pop', 'max fit in pop'])
plt.show()
print('best solution',solutions[np.argmin(fit)])