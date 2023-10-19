# DK

```bash
pip install sb3-contrib
pip install 'stable-baselines3[extra]'
pip install torch
pip install gymnasium
pip install scipy
pip install pyglet
```

# FQF, IQN, QR-DQN and DQN in PyTorch

This is a PyTorch implementation of Fully parameterized Quantile Function(FQF)[1], Implicit Quantile Networks(IQN)[2], Quantile Regression DQN(QR-DQN)[3] and Deep Q Network(DQN)[4].

python version 3.10

```bash
pip install scipy==1.11.1
pip install numpy==1.25.1
pip install matplotlib==3.7.2
pip install gym==0.26.2
pip install pyglet==2.0.8
pip install mcts
pip install wandb
```

Task : four in a row (9 x 4)


### Problem

간헐적으로 obs[3] 이 36이 아닌데 draw가 찍힘
이게 왜 그러는거지

### version

black 학습, white 랜덤

black win -> reward : 1
white win -> reward : abs(-1)
draw -> reward : 0
