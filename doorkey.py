#To replace the doorkey.py file within the gym_minigrid/envs package folder


from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class DoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8, h_trim = 0,
            rand_goal=False,
            agent_start_pos=None,
            agent_start_dir=None,
            easier=False,
            max_steps=None,
            ):
        
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.rand_goal = rand_goal
        self.easier=easier
        if max_steps is None:
            max_steps=10*size*size
        
        if h_trim == 0:
            super().__init__(
                grid_size=size,
                max_steps=max_steps
            )
        else:
            super().__init__(
                width=size,
                height=size-h_trim,
                max_steps=max_steps
            )
            
        

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)
        
        if self.rand_goal:
            self.place_obj(Goal(),top=(max(splitIdx+1,width-3),0),size=(width-1-splitIdx,height))
        else:
            # Place a goal in the bottom-right corner
            self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent(size=(min(splitIdx,3), height))

    # Place a yellow key on the left side
        key_pos = self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        # Place a door in the wall
        if self.easier:
            doorIdx = self._rand_int(key_pos[1], height-1)
        else:
            doorIdx = self._rand_int(1, height-1)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        

        self.mission = "use the key to open the door and then get to the goal"

class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5,
            max_steps=100)
        
class DoorKeyRandGoalEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5,
            rand_goal=True,
            agent_start_pos=(1,1),
            agent_start_dir=1,
            max_steps=100)
        
class DoorKeyRandGoalEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6,
            rand_goal=True,
            agent_start_pos=(1,1),
            agent_start_dir=1,
            max_steps=100)
        
class DoorKeyRandEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6,
            rand_goal=True,
            max_steps=100
            )

class DoorKeyRandEnv8x8(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8,
            rand_goal=True,
            max_steps=250
            )
        
class DoorKeyRandEnv10x10(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=10,
            rand_goal=True,
            max_steps=250
            )

class DoorKeyEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)
        
class DoorKeyRandGoalEnv5x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6, h_trim=1,
            rand_goal=True,
            agent_start_pos=(1,1),
            agent_start_dir=1,
            max_steps=100)

class DoorKeyEnv5x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6, h_trim=1,
            agent_start_pos=(1,1),
            agent_start_dir=1,
            max_steps=100)

class DoorKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-DoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv5x5'
)

register(
    id='MiniGrid-DoorKey-RandGoal-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyRandGoalEnv5x5'
)

register(
    id='MiniGrid-DoorKey-RandGoal-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyRandGoalEnv6x6'
)

register(
    id='MiniGrid-DoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv6x6'
)

register(
    id='MiniGrid-DoorKey-RandGoal-5x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyRandGoalEnv5x6'
)

register(
    id='MiniGrid-DoorKey-Rand-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyRandEnv6x6'
)

register(
    id='MiniGrid-DoorKey-Rand-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyRandEnv8x8'
)

register(
    id='MiniGrid-DoorKey-Rand-10x10-v0',
    entry_point='gym_minigrid.envs:DoorKeyRandEnv10x10'
)

register(
    id='MiniGrid-DoorKey-5x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv5x6'
)

register(
    id='MiniGrid-DoorKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv'
)

register(
    id='MiniGrid-DoorKey-16x16-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv16x16'
)
