import numpy as np
import wandb

from Project.fiar_env import Fiar, turn, action2d_ize
from collections import deque
from Project.mcts import MCTSPlayer
from Project.model.dqn import DQN


# self-play parameter
c_puct = 5
n_playout = 400  # previous 400
# num of simulations for each move

self_play_sizes = 1
temp = 1e-3
buffer_size = 10000
epochs = 5  # num of train_steps for each update
self_play_times = 1000   # previous 1500
pure_mcts_playout_num = 500     # previous 1000

# policy update parameter
batch_size = 64  # previous 512
learn_rate = 2e-3
lr_mul = 1.0
lr_multiplier = 1.0     # adaptively adjust the learning rate based on KL
check_freq = 50  # previous 50
best_win_ratio = 0.0

kl_targ = 0.02  # previous 0.02


# dqn parameter
total_timesteps = 100000
learning_starts = 1000
eps = 0.05


init_model = None


def collect_selfplay_data(n_games=1):
    for i in range(n_games):
        rewards, play_data = self_play(env, model, temp=temp)
        play_data = list(play_data)[:]
        data_buffer.extend(play_data)


def self_play(env, model, temp=1e-3):
    states, mcts_probs, current_player = [], [], []
    obs, _ = env.reset()

    player_0 = turn(obs)
    player_1 = 1 - player_0

    obs_post[0] = obs[player_0]
    obs_post[1] = obs[player_1]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[player_0] + obs[player_1]

    while True:
        while True:
            action = None
            move_probs = None
            if obs[3].sum() == 36:
                print('draw')
            else:
                move, move_probs = mcts_player.get_action(env, obs_post, temp=temp, return_prob=1)
                action = move
                # action = model.predict(obs_post.reshape(*[1, *obs_post.shape]))[0]
                # action = action[0]

            # action = env.action_space.sample()
            action2d = action2d_ize(action)

            if obs[3, action2d[0], action2d[1]] == 0:
                break

        player_0 = turn(obs)
        player_1 = 1 - player_0

        if player_0 == 1:
            while True:
                action = env.action_space.sample()
                action2d = action2d_ize(action)
                if obs[3, action2d[0], action2d[1]] == 0:
                    break
        obs, reward, terminated, info = env.step(action)

        obs_post[0] = obs[player_0]
        obs_post[1] = obs[player_1]
        obs_post[2] = np.zeros_like(obs[0])

        if terminated:
            if obs[3].sum() == 36:
                print('draw')
                env.render()
            obs, _ = env.reset()

            # print number of steps
            if player_0 == 1:
                reward *= -1

            rewards.append(reward)
            # mcts_probs()
            won_side.append(player_0)

            c += 1
            if c == 100:
                break

    return np.array(rewards), np.array(won_side)
# 지금 그냥 rewards랑 won_side 이렇게 반환하고 있는데
# 총 결국엔 총 4개를 반환해야함
# reward (무승부 판별) , end state , mcts probablity, winner (흑 or 백 or 무승부)












if __name__ == '__main__':

    wandb.init(mode="offline",
               entity="hails",
               project="4iar_DQN")

    env = Fiar()
    obs, _ = env.reset()
    data_buffer = deque(maxlen=buffer_size)

    model = DQN("CNNPolicy", env, verbose=1, learning_starts=learning_starts)

    total_timesteps, callback = model._setup_learn(
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="DQN_fiar",
        progress_bar=True,
    )

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[turn_A] + obs[turn_B]

    mcts_player = MCTSPlayer(c_puct, n_playout)

    try:
        for i in range(self_play_times):
            collect_selfplay_data(self_play_sizes, model)


        for i in range(len(rewards)):
            data_buffer.append((rewards, wons))

        env.reset()





        # 그 다음에 평가하는 부분
        while True:
            # sample an action until a valid action is sampled
            while True:
                if obs[3].sum() == 36:
                    print('draw')
                    break
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    if player_0 == 0:  # black train version
                        action = model.predict(obs_post.reshape(*[1, *obs_post.shape]))[0]
                        action = action[0]
                    else:
                        action = env.action_space.sample()

                # action = env.action_space.sample()
                action2d = action2d_ize(action)

                if obs[3, action2d[0], action2d[1]] == 0:
                    break

            player_0 = turn(obs)
            player_1 = 1 - player_0

            obs, reward, terminated, info = env.step(action)

            # reward = np.abs(reward)  # make them equalized for any player
            # -1 로 reward가 들어가게되면 문제가 생기긴하지만 일단 lock

            num_timesteps += 1

            obs_post[0] = obs[player_0]
            obs_post[1] = obs[player_1]
            obs_post[2] = np.zeros_like(obs[0])
            # obs_post = obs[player_myself] + obs[player_enemy] * (-1)

            c += 1

            model._store_transition(model.replay_buffer, np.array([action]), obs_post.reshape(*[1, *obs_post.shape]),
                                    np.array([reward]), np.array([terminated]), [info])

            if num_timesteps > 0 and num_timesteps > learning_starts:
                model.train(batch_size=model.batch_size, gradient_steps=1)

            if terminated:
                env.render()
                if obs[3].sum() == 36:
                    print('draw')
                obs, _ = env.reset()
                # print number of steps
                print('steps:', c)
                print('player:{}, reward:{}'.format(player_0, reward))

                if reward == 0:
                    pass
                else:
                    if player_0 == 0.0:
                        b_win += 1
                    elif player_0 == 1.0:
                        w_win += 1

                b_wins = b_win / ep
                w_wins = w_win / ep

                print({"episode ": ep, "black win (%)": round(b_wins, 5) * 100, "white win (%)": round(w_wins, 5) * 100,
                      "black wins time": b_win,"white wins time": w_win, "tie time": ep - b_win - w_win})
                print('\n\n')
                # 나중에 이부분 round로 나두지말고 format으로 처리해서 부동소수점 문제 처리

                c = 0
                ep += 1

