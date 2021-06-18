# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:50:15 2020

@author: Connor
"""


import torch

from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, state_dim, output_dim, hidden_dim, 
                 vocab_size, embedding_dim, colour_size,
                 att_size, att_emb_dim, vision, use_lstm=True, goal_gen=False):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vision = vision
        self.goal_gen=goal_gen
        self.vocab_size = vocab_size
        
        
        self.obj_dim = embedding_dim
        self.att_dim = att_emb_dim
        self.goal_dim = embedding_dim*2
        self.goal_loc_dim = embedding_dim

        
        self.embed_object = nn.Embedding(vocab_size, self.obj_dim)
        self.embed_colour = nn.Embedding(colour_size, self.att_dim)
        self.embed_attribute = nn.Embedding(att_size, self.att_dim)
        self.embed_goal = nn.Embedding(vocab_size, self.goal_dim)
        self.embed_goal_pos1 = nn.Embedding(vision, self.goal_loc_dim)
        self.embed_goal_pos2 = nn.Embedding(vision, self.goal_loc_dim)

        self.fc = nn.Sequential(
            nn.Linear(state_dim*(self.obj_dim+self.att_dim*2) +  self.goal_dim + self.goal_loc_dim*2, self.hidden_dim),
            nn.ReLU(),
        )

        
        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(self.hidden_dim, self.hidden_dim, 2)

        self.policy = nn.Linear(self.hidden_dim, self.output_dim)
        self.baseline = nn.Linear(self.hidden_dim, 1)
        
        

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2)
        )
    
    def _select(self, embed, x):
        """Efficient function to get embedding from an index."""
        out = embed.weight.index_select(0, x.reshape(-1))
        # handle reshaping x to 1-d and output back to N-d
        return out.reshape(x.shape +(-1,))

    
    def create_embeddings(self, x, id):
        """Generates compositional embeddings."""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_colour, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_attribute, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings
    
    
    def goal_to_policy_index(self, goal, pos1, pos2):
        idx = goal*self.vision**2
        idx += pos1*self.vision + pos2
        return idx

    
    def create_goal_embedding(self, goal, pos1, pos2):
        """Generates compositional embeddings."""
        goal_emb = self._select(self.embed_goal, goal)
        pos1_emb = self._select(self.embed_goal_pos1, pos1)
        pos2_emb = self._select(self.embed_goal_pos2, pos2)
        embeddings = torch.cat((goal_emb, pos1_emb, pos2_emb),-1)
        return embeddings
    

    def forward(self, inputs, core_state=(), goal=(), goal_pos1=(), goal_pos2=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs["frame"]
        T, B, h, w, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        goal = torch.flatten(goal, 0, 1)
        goal_pos1 = torch.flatten(goal_pos1, 0, 1)
        goal_pos2 = torch.flatten(goal_pos2, 0, 1)


        x = x.long()
        goal = goal.long()
        goal_pos1 = goal_pos1.long()
        goal_pos2 = goal_pos2.long()

        # -- [B x H x W x K]
        goal_emb = self.create_goal_embedding(goal, goal_pos1, goal_pos2)
        state_emb = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)

        
        state_emb = state_emb.view(T * B, -1)
        fc_input = torch.cat((state_emb,goal_emb),-1)
        core_input = self.fc(fc_input)

        
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            # print(notdone.shape)
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        
        out_prob = F.softmax(policy_logits, dim=1)

        if self.training:
            out = torch.multinomial(out_prob, num_samples=1)
        else:
            # Don't sample when testing.
            out = torch.argmax(out_prob, dim=1)

        policy_logits = policy_logits.view(T, B, self.output_dim)
        baseline = baseline.view(T, B)
        out = out.view(T, B)

        if self.goal_gen:
            position = out%(self.vision**2)
            gp1 = position//self.vision
            gp2 = position%self.vision
            g = out//(self.vision**2)
            return (
                dict(goal_gen_policy_logits=policy_logits, goal_gen_baseline=baseline, goal_gen_out=out, goal=g, goal_pos1=gp1, goal_pos2=gp2),
                core_state,
                )
        else:
            return (
                dict(policy_logits=policy_logits, baseline=baseline, action=out),
                core_state,
                )

        
    
    
    