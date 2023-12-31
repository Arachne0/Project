import numpy as np
import wandb
import random

from Project.fiar_env import Fiar, turn, action2d_ize
from collections import defaultdict, deque
from Project.mcts import MCTSPlayer
from Project.mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_network import PolicyValueNet
# from policy_value_network_mlp import PolicyValueNet


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


init_model = None


def policy_value_fn(board):  # board.shape = (9,4)
    # return uniform probabilities and 0 score for pure MCTS
    availables = [i for i in range(36) if not np.any(board[3][i // 4][i % 4] == 1)]
    action_probs = np.ones(len(availables)) / len(availables)
    return zip(availables, action_probs), 0


def get_equi_data(env, play_data):
    """augment the data set by flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_prob, winner in play_data:
        # flip horizontally
        equi_state = np.array([np.fliplr(s) for s in state])
        equi_mcts_prob = np.fliplr(mcts_prob.reshape(env.state().shape[1], env.state().shape[2]))
        extend_data.append((equi_state,
                            np.flipud(equi_mcts_prob).flatten(),
                            winner))
    return extend_data


def collect_selfplay_data(n_games=1):
    for i in range(n_games):
        rewards, play_data = self_play(env, temp=temp)
        play_data = list(play_data)[:]
        play_data = get_equi_data(env, play_data)
        data_buffer.extend(play_data)


def self_play(env, temp=1e-3):
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
            action2d = action2d_ize(action)

            if obs[3, action2d[0], action2d[1]] == 0.0:
                break

        # store the data
        states.append(obs)
        mcts_probs.append(move_probs)
        current_player.append(turn(obs))

        obs, reward, terminated, info = env.step(action)

        player_0 = turn(obs)
        player_1 = 1 - player_0

        obs_post[0] = obs[player_0]
        obs_post[1] = obs[player_1]
        obs_post[2] = np.zeros_like(obs[0])
        obs_post[3] = obs[player_0] + obs[player_1]

        end, winners = env.winner()
        # Return value from env.winner is identical to the return value
        # so, black win -> 1 , white win -> 0.1

        if end:
            if obs[3].sum() == 36:
                print('draw')

            print(env)
            obs, _ = env.reset()

            # reset MCTS root node
            mcts_player.reset_player()
            print("batch i:{}, episode_len:{}".format(
                i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))

            if winners != -1:
                if winners == -0.5:  # if win white return : 0.1
                    winners = 0
                winners_z[np.array(current_player) == 1 - winners] = 1.0
                winners_z[np.array(current_player) != 1 - winners] = -1.0
            return reward, zip(states, mcts_probs, winners_z)


def policy_update(lr_mul):
    kl, loss, entropy = 0, 0, 0
    lr_multiplier = lr_mul

    """update the policy-value net"""
    mini_batch = random.sample(data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)

    for i in range(epochs):
        loss, entropy = policy_value_net.train_step(
            state_batch,
            mcts_probs_batch,
            winner_batch,
            learn_rate * lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1)
                     )
        if kl > kl_targ * 4:  # early stopping if D_KL diverges badly
            break

    # adaptively adjust the learning rate
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    explained_var_old = (1 -
                         np.var(np.array(winner_batch) - old_v.flatten()) /
                         (np.var(np.array(winner_batch)) + 1e-10))
    explained_var_new = (1 -
                         np.var(np.array(winner_batch) - new_v.flatten()) /
                         (np.var(np.array(winner_batch)) + 1e-10))

    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{},"
           "explained_var_old:{:.3f},"
           "explained_var_new:{:.3f}"
           ).format(kl,
                    lr_multiplier,
                    loss,
                    entropy,
                    explained_var_old,
                    explained_var_new))
    return loss, entropy, lr_multiplier


def policy_evaluate(env, n_games=10):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    current_mcts_player = MCTSPlayer(policy_value_fn,
                                     c_puct=c_puct,
                                     n_playout=n_playout)
    pure_mcts_player = MCTS_Pure(c_puct=5,
                                 n_playout=pure_mcts_playout_num)
    win_cnt = defaultdict(int)

    for i in range(n_games):
        winner = start_play(env,
                            current_mcts_player,
                            pure_mcts_player)
        if winner == -0.5:
            winner = 0
        win_cnt[winner] += 1
        print('one time evaluate end')

    win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games

    print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
        pure_mcts_playout_num,
        win_cnt[1], win_cnt[0], win_cnt[-1]))
    return win_ratio


def start_play(env, player1, player2):
    """start a game between two players"""

    obs, _ = env.reset()

    players = [0, 1]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 0

    while True:
        player_in_turn = players[current_player]
        move = player_in_turn.get_action(env, obs)
        obs, reward, terminated, info = env.step(move)

        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player

        else:
            print(env)
            return winner


if __name__ == '__main__':

    wandb.init(mode="offline",
               entity="hails",
               project="policy_value_4iar")

    env = Fiar()
    obs, _ = env.reset()
    data_buffer = deque(maxlen=buffer_size)
    policy_value_net = PolicyValueNet(obs.shape[1], obs.shape[2])

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[turn_A] + obs[turn_B]

    if init_model:
        # start training from an initial policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1],
                                          env.state().shape[2],
                                          model_file=init_model)
    else:
        # start training from a new policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1],
                                          env.state().shape[2])

    mcts_player = MCTSPlayer(policy_value_fn, c_puct, n_playout, is_selfplay=1)

    try:
        for i in range(self_play_times):
            collect_selfplay_data(self_play_sizes)

            if len(data_buffer) > batch_size:
                loss, entropy, lr_multiplier = policy_update(lr_mul=lr_multiplier)
                wandb.log({"loss": loss, "entropy": entropy})

            if (i + 1) % check_freq == 0:
                print("current self-play batch: {}".format(i + 1))
                win_ratio = policy_evaluate(env)
                print("win rate : ", win_ratio * 100, "%")

                policy_value_net.save_model('./current_policy.model')

                if win_ratio > best_win_ratio:
                    print("New best policy!!!!!!!!")
                    best_win_ratio = win_ratio

                    # update the best_policy
                    policy_value_net.save_model('./best_policy.model')

                    if (best_win_ratio == 1.0 and
                            pure_mcts_playout_num < 5000):
                        pure_mcts_playout_num += 500    # previous 1000
                        print("Pure mcts level up!!!")
                        best_win_ratio = 0.0

    except KeyboardInterrupt:
        print('\n\rquit')
