# RL Self-Driving Simulation

![env image](Images/car.gif)

## What is this?

This is a game environment designed for Reinforcement Learning/Imitation Learning tasks. It is implemented using the 3D game engine Unity. We also implemented several algorithms based on this environment.

![env image](Images/hero_img.png)

## How to Run it?

For people who wants to test their RL algorithms, you can download the _build_ folder, and use without any further configurations. If you want to have more freedom in customizing the scene, for example, different routes, rewards settings, sensor numbers, please clone this repository and open it in Unity(2020.1.14f1). The _Library_ folder is not included in this repository.

You can test the game by executing the **RLCar.exe** file in the _build_ folder.

## Which Platform?

This game is built using Windows 10. Please use Windows(32/64 bit) if you want to directly run the executable. You can also build for other platforms using Unity.

## Connect to Python?

We offer gym-like APIs in python 3. We recommend using Python 3.6 or above to interact with this environment.

Please install these dependencies before connecting.

```
pip install gym-unity==0.22.0 mlagents-envs==0.22.0
```

Include these libraries in Python.

```
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
```

You can connect the Unity environment using the binary executable file:

```
unity_env = UnityEnvironment("build\RLCar.exe")
env = UnityToGymWrapper(unity_env,0)
```

Or run it in the Unity Engine play mode:

```
unity_env = UnityEnvironment(base_port=5004)
env = UnityToGymWrapper(unity_env,0)
```

The APIs are similar to the gym.

```
state, reward, done, _ = env.step(env.action_space.sample())
```

## Related Projects

Please check for the projects which uses this game environment.

<hr>

### Deep Deterministic Policy Gradient (DDPG)

We used DDPG to train the agent so it learns to drive using the horizontal and the vertical axes. The horizontal axis controls the steering while the vertical axis controls the acceleration.

The implementation can be found [here](https://github.com/songlinhou/RL-Selfdriving-Env/blob/master/Experiments/RL-Project4-Car.ipynb).

<hr>

### Deep Q Learning (DQN)

We used DQN to train the agent after discretizing the action space.

The implementation can be found [here](https://github.com/songlinhou/RL-Selfdriving-Env/blob/master/Experiments/RLCar_Project4_DQN.ipynb).

<hr>

### Imitation Learning (IL)

We play the game and used the user-generated data to train the agent.

The implementation can be found [here](https://github.com/songlinhou/RL-Selfdriving-Env/tree/master/Experiments/Imitation%20Learning) and [here](https://github.com/songlinhou/RL-Selfdriving-Env/blob/master/Experiments/IL_CAR.ipynb).

## Contributors

[Songlin Hou](https://github.com/songlinhou), [Ziyuan Chen](https://github.com/zchen8-hub), [Akshata Pore](https://github.com/Akshata1801) and [Soumya Balijepally](https://github.com/SoumyaSrilekha)
