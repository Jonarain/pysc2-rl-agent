from pysc2.lib.actions import FunctionCall, FUNCTIONS
from common.config import DEFAULT_ARGS, is_spatial


class EnvWrapper:
    def __init__(self, envs, config):
        self.envs, self.config = envs, config #传入的是pysc2的可识别env, pysc2.bin.agent line74的结果

    def step(self, acts):
        acts = self.wrap_actions(acts)#打包[actions, args]
        results = self.envs.step(acts)#pysc2 直接执行acts, 返回的results是由pysc2生成的environment.TimeStep(step_type, reward, discount, observation)
        a = self.wrap_results(results)
        return self.wrap_results(results)

    def reset(self):
        results = self.envs.reset()
        return self.wrap_results(results)

    def wrap_actions(self, actions):
        """
        根据action和action_args的矩阵，
        输出可以pysc2执行的Function call实例
        :param actions: action和action_args的矩阵
        :return: pysc2对应的action ID和pysc2可以执行的args
        """
        # 取出action和action的参数
        acts, args = actions[0], actions[1:]

        wrapped_actions = []
        for i, act in enumerate(acts):#当前顺序i，action的ID act
            act_args = []
            for arg_type in FUNCTIONS[act].args: #根据action的ID找到参数名称
                act_arg = [DEFAULT_ARGS[arg_type.name]]#用config定义的默认参数值来初始化
                if arg_type.name in self.config.act_args:
                    act_arg = [args[self.config.arg_idx[arg_type.name]][i]]
                if is_spatial(arg_type.name):  # spatial args, convert to coords
                    act_arg = [act_arg[0] % self.config.sz, act_arg[0] // self.config.sz]  # (y,x), fix for PySC2
                act_args.append(act_arg)
            wrapped_actions.append(FunctionCall(act, act_args)) #pysc2 可以执行的是这个，原本都是数据

        return wrapped_actions

    def wrap_results(self, results):
        """
        打包pysc2中出来的结果
        :param results:
        :return:states rewards dones
        """

        #从中取出了obs(pysc2风格的list[dict])
        #rewards([int64])
        obs = [res.observation for res in results]
        rewards = [res.reward for res in results]
        dones = [res.last() for res in results]

        states = self.config.preprocess(obs)

        return states, rewards, dones

    def save_replay(self, replay_dir='PySC2Replays'):
        self.envs.save_replay(replay_dir)

    def spec(self):
        return self.envs.spec()

    def close(self):
        return self.envs.close()

    @property
    def num_envs(self):
        return self.envs.num_envs