import cma, ray, time
import numpy as np
import matplotlib.pyplot as plt

ray.init(num_cpus=10)

@ray.remote
def fitness(x):
    x = np.array(x)
    time.sleep(0.001)
    return ((x - np.pi)**2).sum()

es = cma.CMAEvolutionStrategy(x0=[0.]*30,
                                sigma0=0.1,
                                inopts={'popsize': 1000
                                })
tick = time.time()
log = []
for _ in range(200):
    solutions = es.ask()
    fit = []
    results = [fitness.remote(solution) for solution in solutions]
    for result in results:
        fit.append(ray.get(result))
    log.append([min(fit),max(fit)])
    es.tell(solutions, fit)
print('time:',time.time()-tick)
print('best solution',solutions[np.argmin(fit)])
ray.shutdown()
'''
log = np.array(log)
plt.plot(log[:,0])
plt.plot(log[:,1])
plt.grid(); plt.xlabel('iteration'); plt.ylabel('fitness value')
plt.legend(['min fit in pop', 'max fit in pop'])
plt.show()
'''
