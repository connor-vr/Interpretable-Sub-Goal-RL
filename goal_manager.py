# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:28:58 2020

@author: Connor
"""


class Goal_Manager(object):

    def __init__(self, goal, goal_pos1, goal_pos2, flags) -> None:
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        super(Goal_Manager, self).__init__()
       
        self.vision = flags.vision
        
        self.steps_with_goal = 0
        self.step_limit = flags.goal_steps_max
        self.future_goals_limit = 1 #TODO add support for goal depth > 1
        
        self.goal = [goal, [goal_pos1, goal_pos2]]
        self.future_goals = []




    def reset(self):
        self.steps_with_goal = 0
        if len(self.future_goals)>0:
            self.goal = self.future_goals[0]
            self.future_goals = []
        return self.goal[0], self.goal[1][0], self.goal[1][1]



    def step(self):
        self.steps_with_goal += 1
        if self.steps_with_goal >= self.step_limit and len(self.future_goals)>0:
            return True
        else:
            return False



    def check_goal_reached(self, state, done, reward):
        current_hit = (state[:,:,self.goal[1][0][0][0],self.goal[1][1][0][0],0] == self.goal[0][0][0]
                       or (done and reward>0 and self.goal == self.future_goals[0]))
        if not current_hit:
            for i, goal in enumerate(self.future_goals):
                if (state[:,:,goal[1][0][0],goal[1][1][0],0] == goal[0]) or (done and reward>0):
                    return i, True
        return len(self.future_goals), current_hit



    def add_goal(self, goal, goal_pos1, goal_pos2):
        self.future_goals.append(self.goal)
        self.goal = [goal,[goal_pos1,goal_pos2]]
        self.steps_with_goal = 0
        




    def goal_reached(self, idx):
        #check if goal was a queued future goal
        if idx<len(self.future_goals):
            self.goal = self.future_goals[idx]
            self.future_goals = self.future_goals[:idx]
            
        #update the current goal
        if len(self.future_goals)>0:
            self.goal = self.future_goals.pop()

        self.steps_with_goal = 0
        

        return self.goal[0], self.goal[1][0], self.goal[1][1]



    def goal_not_reached(self):
        self.goal = self.future_goals.pop()
        self.steps_with_goal = 0
        
        return self.goal[0], self.goal[1][0], self.goal[1][1]
        
