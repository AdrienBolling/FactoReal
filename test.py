from src.envs.FactoReal import FactoReal
from src.config.config import DEFAULT_ARGS
env = FactoReal(ARGS=DEFAULT_ARGS)
import time 
env.reset()

start = time.time()
for i in range(10000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:        
        break
print(f"Episode finished after {i+1} timesteps and took {time.time()-start} seconds")

print(env)