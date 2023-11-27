# DK

```bash
pip install sb3-contrib
pip install 'stable-baselines3[extra]'
pip install torch
pip install gymnasium
pip install scipy
pip install pyglet
```

# SH

python version 3.10

```bash
pip install sb3-contrib
pip install 'stable-baselines3[extra]'
pip install torch
pip install gymnasium
pip install pyglet
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

배치에 아직 안들어갔고, 배치에 꺼내는것도 아직 없음.


코드가 엉망진창인데, 생각정리가 안되서 일단 올림


내가 alphazero가 아닌 mcts_pure 코드 따라가다가
_evaluate_rollout메서드 때문에 해메서 이렇게 되었음 
이 버전은 evaluate_rollout 있는 버전




### version  
이건 나중에 다시 정리함

black 학습, white 랜덤

black win -> reward : 1
white win -> reward : abs(-1)
draw -> reward : 0

player_myself = 0 , player_enemy = 1 일때 흑이 돌을 놓을 차례
player_myself = 1 , player_enemy = 0 일때 백이 돌을 놓을 차례

