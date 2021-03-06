import json
import numpy as np
from s2clientprotocol import ui_pb2 as sc_ui
from s2clientprotocol import spatial_pb2 as sc_spatial
from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS, TYPES

CAT = features.FeatureType.CATEGORICAL

DEFAULT_ARGS = dict(
    screen=0, # converts to (0,0)
    minimap=0,
    screen2=0,
    queued=False,
    control_group_act=sc_ui.ActionControlGroup.Append,
    control_group_id=1,
    select_point_act=sc_spatial.ActionSpatialUnitSelectionPoint.Select,
    select_add=True,
    select_unit_act=sc_ui.ActionMultiPanel.SelectAllOfType,
    select_unit_id=0,
    select_worker=sc_ui.ActionSelectIdleWorker.AddAll,
    build_queue_id=0,
    unload_id=0
)

# name => dims
NON_SPATIAL_FEATURES = dict(
    player=(11,),
    game_loop=(1,),
    score_cumulative=(13,),
    available_actions=(len(FUNCTIONS),),
    single_select=(1, 7),
    # multi_select=(0, 7), # TODO
    # cargo=(0, 7), # TODO
    cargo_slots_available=(1,),
    # build_queue=(0, 7), # TODO
    control_groups=(10, 2),
)


class Config:
    # TODO extract embed_dim_fn to config
    def __init__(self, sz, map, run_id, embed_dim_fn=lambda x: max(1, round(np.log2(x)))):
        self.run_id = run_id
        self.sz, self.map = sz, map
        self.embed_dim_fn = embed_dim_fn  #?
        self.feats = self.acts = self.act_args = self.arg_idx = self.ns_idx = None

    def build(self, cfg_path):
        #读了config，生成了feats acts act_args,然后有的东西就不变，没有的就加上
        feats, acts, act_args = self._load(cfg_path)
        print(acts)

        if 'screen' not in feats:
            feats['screen'] = features.SCREEN_FEATURES._fields #取出元组的索引(名字)
        if 'minimap' not in feats:
            feats['minimap'] = features.MINIMAP_FEATURES._fields
        if 'non_spatial' not in feats:
            feats['non_spatial'] = NON_SPATIAL_FEATURES.keys() #取出字典的索引
        self.feats = feats

        # TODO not connected to anything atm
        if acts is None:
            acts = FUNCTIONS
        self.acts = acts

        if act_args is None:
            act_args = TYPES._fields
        self.act_args = act_args

        self.arg_idx = {arg: i for i, arg in enumerate(self.act_args)} #act_args在pysc2里面的定义顺序
        self.ns_idx = {f: i for i, f in enumerate(self.feats['non_spatial'])}

    def map_id(self):
        return self.map + str(self.sz)

    def full_id(self):
        if self.run_id == -1:
            return self.map_id()
        return self.map_id() + "/" + str(self.run_id)

    def policy_dims(self):# 这里的+是合并两个[()]
        return [(len(self.acts), 0)] + [(getattr(TYPES, arg).sizes[0], is_spatial(arg)) for arg in self.act_args]

    def screen_dims(self):
        return self._dims('screen')

    def minimap_dims(self):
        return self._dims('minimap')

    def non_spatial_dims(self):
        return [NON_SPATIAL_FEATURES[f] for f in self.feats['non_spatial']]

    # TODO maybe move preprocessing code into separate class?
    def preprocess(self, obs):
        #obs = pysc2风格的observation
        #_type = ['screen', 'minimap', self.feats['non_spatial']里面的内容]
        return [self._preprocess(obs, _type) for _type in ['screen', 'minimap'] + self.feats['non_spatial']]

    def _dims(self, _type):
        a = [f.scale ** (f.type == CAT) for f in self._feats(_type)]
        return [f.scale**(f.type == CAT) for f in self._feats(_type)]

    def _feats(self, _type):
        feats = getattr(features, _type.upper() + '_FEATURES')
        return [getattr(feats, f_name) for f_name in self.feats[_type]]

    def test(self):
        for _type in ['screen', 'minimap'] + self.feats['non_spatial']:
            print(_type)

    def _preprocess(self, obs, _type):
        #obs = pysc2风格的observation
        #_type = ['screen', 'minimap', self.feats['non_spatial']里面的内容]

        #处理non_spatial
        if _type in self.feats['non_spatial']:
            return np.array([self._preprocess_non_spatial(ob, _type) for ob in obs])

        #处理spatial
        spatial = [[ob[_type][f.index] for f in self._feats(_type)] for ob in obs]#ob是obs这个由dict组成的list里面的一个dict
        return np.array(spatial).transpose((0, 2, 3, 1))

    def _preprocess_non_spatial(self, ob, _type):
        if _type == 'available_actions':
            acts = np.zeros(len(self.acts))
            acts[ob['available_actions']] = 1
            return acts
        return ob[_type]

    def save(self, cfg_path):
        with open(cfg_path, 'w') as fl:
            json.dump({'feats': self.feats, 'act_args': self.act_args}, fl) #写的时候只写了 feats和act_args

    def _load(self, cfg_path):
        with open(cfg_path, 'r') as fl:
            data = json.load(fl)
        return data.get('feats'), data.get('acts'), data.get('act_args') #读的时候 检查了feats acts act_args


def is_spatial(arg):
    return arg in ['screen', 'screen2', 'minimap']
