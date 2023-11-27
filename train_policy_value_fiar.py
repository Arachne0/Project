from fiar_env import Fiar, turn, action2d_ize
import numpy as np
import wandb
import random

from collections import deque
from mcts import MCTS, MCTSPlayer
from model.dqn import DQN


def collect_selfplay_data(n_games=1):

    for i in range(n_games):
        rewards, play_data = self_play(env)
        # play_data에 총 3가지 데이터가 들어가야하는데
        # states, mcts_probs, winners_z 이렇게 줌
        # 3번째는 만약 게임이 종료 되었다면 승자를 반환함

        play_data = list(play_data)[:]
        data_buffer.extend(play_data)


def self_play(env):

    states, mcts_probs, current_player = [], [], []
    obs, _ = env.reset()

    players = [0, 1]
    player_0 = turn(obs)
    player_1 = 1 - player_0

    obs_post[0] = obs[player_0]
    obs_post[1] = obs[player_1]
    obs_post[2] = np.zeros_like(obs[0])
    c = 0

    while True:
        env.reset()

        while True:
            if obs[3].sum() == 36:
                print('draw')
                break
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                move, move_probs = mcts_player.get_action(env, obs_post)
                # 잠만 여기 좀 만져야함
                # obs으로 안하고 env로 끝낼 수 있을거 같기도 한데 일단 돌아가게만 만들어봄
                action = move

            action2d = action2d_ize(action)

            if obs[3, action2d[0], action2d[1]] == 0:
                break

            print("sex")
            # store the data
            states.append(board.current_state())
            mcts_probs.append(move_probs)
            current_player.append(self.board.current_player)

            # 줘야하는건 1.states, 2.mcts_prob , 3.current player












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

#일단 끝났을때 states를 반환해야함


# 지금 그냥 rewards랑 won_side 이렇게 반환하고 있는데
# 총 결국엔 총 4개를 반환해야함
# reward (무승부 판별) , end state , mcts probablity, winner (흑 or 백 or 무승부)

def policy_update(self):
    """update the policy-value net"""
    mini_batch = random.sample(data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = self.policy_value_net.policy_value(state_batch)











total_timesteps = 100000
learning_starts = 1000
eps = 0.05

if __name__ == '__main__':

    wandb.init(project="4iar_DQN")

    env = Fiar()
    obs, _ = env.reset()
    buffer_size = 1000
    data_buffer = deque(maxlen=buffer_size)

    model = DQN("MlpPolicy", env, verbose=1, learning_starts=learning_starts)

    total_timesteps, callback = model._setup_learn(
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="DQN_fiar",
        progress_bar=True,
    )

    player_0 = turn(obs)
    player_1 = 1 - player_0

    obs_post = obs.copy()
    obs_post[0] = obs[player_0]
    obs_post[1] = obs[player_1]
    obs_post[2] = np.zeros_like(obs[0])
    # obs_post = obs[player_myself] + obs[player_enemy]*(-1)

    c = 0
    num_timesteps = 0
    ep = 1
    b_win = 0
    w_win = 0
    c_puct = 5
    n_playout = 2000
    self_play_times = 1500
    self_play_sizes = 1

    batch_size = 512

    mcts_player = MCTSPlayer(c_puct, n_playout)

    for i in range(self_play_times):
        collect_selfplay_data(self_play_sizes)

        if len(data_buffer) > batch_size:
            loss, entropy = policy_update()

             # 여기에서 데이터 버퍼에 집어넣어놓고 그 다음에 actor-critic 하는 부분 있어야 함
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

            # evaluation against random agent
            # if ep % 1000 == 0:
            #    rewards, wons = evaluation_against_random(env, model)
                # save model
            #    model.save("qrdqn_fiar")