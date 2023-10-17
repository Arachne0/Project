from fiar_env import Fiar, turn, action2d_ize
import numpy as np
from model.qrdqn import QRDQN
from gym_4iar.prev_codes.mcts import MCTSPlayer


def evaluation_against_random(env, model):
    won_side = []
    rewards = []
    obs, _ = env.reset()
    player_myself = turn(obs)
    player_enemy = 1 - player_myself
    obs_post[0] = obs[player_myself]
    obs_post[1] = obs[player_enemy]
    obs_post[2] = np.zeros_like(obs[0])
    c = 0

    while True:
        while True:
            if obs[3].sum() == 36:
                print('draw')
                break
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = model.predict(obs_post.reshape(*[1, *obs_post.shape]))[0]
                action = action[0]
            # action = env.action_space.sample()
            action2d = action2d_ize(action)

            if obs[3, action2d[0], action2d[1]] == 0:
                break

        player_myself = turn(obs)
        player_enemy = 1 - player_myself

        if player_myself == 1:
            while True:
                action = env.action_space.sample()
                action2d = action2d_ize(action)
                if obs[3, action2d[0], action2d[1]] == 0:
                    break
        obs, reward, terminated, info = env.step(action)

        obs_post[0] = obs[player_myself]
        obs_post[1] = obs[player_enemy]
        obs_post[2] = np.zeros_like(obs[0])

        if terminated:
            if obs[3].sum() == 36:
                print('draw')
            obs, _ = env.reset()
            # print number of steps
            # if player_myself == 1:
            # 	reward *= -1
            rewards.append(reward)
            won_side.append(player_myself)
            c += 1
            if c == 100:
                break

    return np.array(rewards), np.array(won_side)


total_timesteps = 100000
learning_starts = 1000
eps = 0.05

if __name__ == '__main__':
    env = Fiar()
    obs, _ = env.reset()

    policy_kwargs = dict(n_quantiles=50)
    model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_starts=learning_starts)

    total_timesteps, callback = model._setup_learn(
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="QRDQN_fiar",
        progress_bar=True,
    )

    player_myself = turn(obs)
    player_enemy = 1 - player_myself

    obs_post = obs.copy()
    obs_post[0] = obs[player_myself]
    obs_post[1] = obs[player_enemy]
    obs_post[2] = np.zeros_like(obs[0])
    c = 0
    num_timesteps = 0
    ep = 1

    mcts_player = MCTSPlayer(env, obs)

    while True:
        action = mcts_player.get_action(obs)
        action2d = action2d_ize(action)

        if obs[3, action2d[0], action2d[1]] == 0:
            break

        player_myself = turn(obs)
        player_enemy = 1 - player_myself

        obs, reward, terminated, info = env.step(action)
        reward = np.abs(reward)  # make them equalized for any player
        num_timesteps += 1

        obs_post[0] = obs[player_myself]
        obs_post[1] = obs[player_enemy]
        obs_post[2] = np.zeros_like(obs[0])

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
            # print player_myself, reward, ep in a line
            print('ep:{}, player:{}, reward:{}'.format(ep, player_myself, reward))

            c = 0
            ep += 1

            # evaluation against random agent
            if ep % 1000 == 0:
                rewards, wons = evaluation_against_random(env, model)
                # save model
                model.save("qrdqn_fiar")
