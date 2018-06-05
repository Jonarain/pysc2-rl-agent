import json


class EnvLogger:
    def __init__(self, envs, config):
        self.envs, self.config = envs, config

    def step(self, acts):
        results = self.envs.step(acts)
        self.log(results, acts)
        return results

    def reset(self):
        results = self.envs.reset()
        self.log(results, [])
        return results

    def log(self, results, acts):
        rewards = [int(res.reward) for res in results]
        dones = [int(res.last()) for res in results]

        obs = [{} for _ in range(len(results))]
        for i, res in enumerate(results):
            for _type in ['screen', 'minimap'] + self.config.feats['non_spatial']:
                obs[i][_type] = res.observation[_type].astype(int).tolist()

        raw_acts = []
        for a in acts:
            raw_act = [int(a.function)]
            for arg in a.arguments:
                raw_act.append(int(arg[0])) if len(arg) == 1 else raw_act.append(int(arg[0] * self.config.sz + arg[1]))
            raw_acts.append(raw_act)

        with open('logs/%s/env_logs.json' % self.config.full_id(), 'a+') as fl:
            json.dump({'states': [obs, rewards, dones], 'acts': raw_acts}, fl)

    def save_replay(self, replay_dir='PySC2Replays'):
        self.envs.save_replay(replay_dir)

    def spec(self):
        return self.envs.spec()

    def close(self):
        return self.envs.close()

    @property
    def num_envs(self):
        return self.envs.num_envs
