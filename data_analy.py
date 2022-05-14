import pickle
import numpy as np

with open('./results/ckpt_deepsymbol-v3_LunarLander-v2/CMA_ES-390.pkl', 'rb') as f:
    best_solution = pickle.load(f)
print(best_solution.fc.weight, best_solution.fc.bias)
print(len(best_solution.model.get_params())/3)

print(best_solution.model())


