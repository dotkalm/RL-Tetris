from pufferlib.ocean.tetris import tetris
import time

env = tetris.Tetris(render_mode='human')  # or 'ansi'
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()  # Just call this!
    time.sleep(0.1)
    
    done = terminated or truncated

env.close()
