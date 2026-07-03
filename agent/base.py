from .utils.utils import Utils


class BaseAgent(object):
    '''
    BaseAgent Class is mainly used for creating a base agent and base methods.

    Inspired by LibSignal
    '''
    def __init__(self, gym_env):
        self.gym_env = gym_env

    # Get observation from gym env
    def get_info(self):
        raise NotImplementedError()

    # Get reward (makes sense mostly for RL agents)
    def get_reward(self):
        raise NotImplementedError()
    
    # Outputs available actions: during a yellow phase only one action is
    # available; otherwise all actions except yellow are. If the agent
    # outputs an unavailable action in next_action, the environment raises.
    def get_avail_action(self) -> dict[str, list[int]]:
        raise NotImplementedError()


    # Generate next action, phase id per intersection
    # ob -> observation
    def next_action(self, ob):
        raise NotImplementedError()
    
    # Shared training loop for RL agents; requires the subclass to
    # implement act(info, explore) and observe(next_info, reward,
    # terminated, truncated).
    def train(self, episodes, max_steps=None):
        episode_returns = []
        step_limit = int(max_steps) if max_steps is not None else int(
            getattr(self.gym_env, "max_steps", 3600)
        )

        for _ in range(int(episodes)):
            info = self.gym_env.reset()
            ep_return = 0.0

            for _ in range(step_limit):
                action = self.act(info, explore=True)
                reward, terminated, truncated, next_info = self.gym_env.step(action)

                self.observe(next_info, reward, bool(terminated), bool(truncated))
                ep_return += Utils.scalar_reward(reward)
                info = next_info

                if terminated or truncated:
                    break

            episode_returns.append(ep_return)

        return episode_returns

    def save(self, path):
        pass

    def load(self, path):
        pass
