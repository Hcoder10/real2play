"""Offline high-level command progression for game-trained placement policies.

Labels and actions remain learned. This state-only skeleton scheduler stands in
for Argon's high-level commands during a zero-request CPU evaluation.
"""
from r2p.policy.eval import CommandTracker
from r2p.sim.twin import extents
from r2p.tasks import check


class GameCommands(CommandTracker):
    def update(self, state, q, start_q):
        if self.done:
            return
        step = self.task['skeleton'][self.index]
        # A separate release command cannot require the placing policy to have
        # opened already. Require the piece at the destination AND lowered;
        # the final success checker still requires open jaws and a stable window.
        if step['skill'] != 'place_at' or self.index+1 >= len(self.task['skeleton']) or self.task['skeleton'][self.index+1]['skill'] != 'release':
            return super().update(state, q, start_q)
        self.age += 1
        obj = step['obj']
        pose = state['objects'].get(obj)
        definition = next(o for o in self.task['objects'] if o['id'] == obj)
        goals = [g for g in self.task['goal'] if g.get('a') == obj]
        geometry = {'objects': state['objects'], 'initial': state.get('initial', self.initial)}
        low = pose is not None and pose['xyz'][2] < extents({**definition, 'size_m': pose.get('size_m', definition['size_m'])})[2]/2+.02
        complete = bool(goals) and low and check(dict(self.task, goal=goals), geometry)['ok']
        self.stable = self.stable+1 if complete else 0
        if self.stable >= 4:
            self.index += 1
            self.age = self.stable = 0
