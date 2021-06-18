# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:28:43 2020

@author: Connor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, 
                 vocab_size, embedding_dim, colour_size,
                 att_size, sec_emb_dim, vision, use_lstm=False):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.vision = vision
        
        self.obj_dim = embedding_dim
        self.sec_emb_dim = sec_emb_dim

        self.embed_object = nn.Embedding(vocab_size, self.obj_dim)
        self.embed_colour = nn.Embedding(colour_size, self.sec_emb_dim)
        self.embed_attribute = nn.Embedding(att_size, self.sec_emb_dim)


        self.fc = nn.Sequential(
            nn.Linear(state_dim*(self.obj_dim+self.sec_emb_dim*2)*2 + action_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )

        
        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(self.hidden_dim, self.hidden_dim, 2)

        self.highway = nn.Linear(self.hidden_dim, self.hidden_dim)
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
    

    def forward(self, state, next_state, actions, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = state
        y = next_state
        T, B, h, w, *_ = x.shape
        
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        y = torch.flatten(y, 0, 1)



        x = x.long()
        y = y.long()
        
        # -- [B x H x W x K]
        x_emb = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
        y_emb = torch.cat([self.create_embeddings(y, 0), self.create_embeddings(y, 1), self.create_embeddings(y, 2)], dim = 3)

        
        x_emb = x_emb.view(T * B, -1)
        y_emb = y_emb.view(T * B, -1)
        actions = F.one_hot(torch.flatten(actions), self.action_dim).float()
        fc_input = torch.cat((x_emb,y_emb, actions),-1)
        core_input = self.fc(fc_input)

        
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            # print(notdone.shape)
            for input in core_input.unbind():
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()
            
        highway = self.highway(core_output)
        out = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * core_output  # highway

        baseline = torch.sigmoid(self.baseline(out))
        baseline = baseline.view(T, B)


        return (
                dict(baseline=baseline),
                core_state,
                )




class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, 
                 vocab_size, embedding_dim, colour_size,
                 att_size, sec_emb_dim, vision, latent_dim, 
                 use_lstm=True):
        super(Generator, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.vision = vision
        self.vocab_size = vocab_size
        self.att_size = att_size
        self.colour_size = colour_size
       
        self.obj_dim = embedding_dim
        self.sec_emb_dim = sec_emb_dim

        
        self.embed_object = nn.Embedding(vocab_size, self.obj_dim)
        self.embed_colour = nn.Embedding(colour_size, self.sec_emb_dim)
        self.embed_attribute = nn.Embedding(att_size, self.sec_emb_dim)

        
        self.use_lstm = use_lstm
        if use_lstm:
            self.fc = nn.Sequential(
            nn.Linear(state_dim*(self.obj_dim+self.sec_emb_dim*2) + action_dim + latent_dim, self.hidden_dim),
            nn.ReLU(),
            )
            self.core = nn.LSTM(self.hidden_dim, self.hidden_dim, 2)
        else:
            self.fc = nn.Sequential(
            nn.Linear(state_dim*(self.obj_dim+self.sec_emb_dim*2) + action_dim + latent_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            )

        self.out_obj = nn.Linear(self.hidden_dim, self.state_dim*vocab_size)
        self.out_col = nn.Linear(self.hidden_dim, self.state_dim*colour_size)
        self.out_att = nn.Linear(self.hidden_dim, self.state_dim*att_size)
                
    

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
    
    

    def forward(self, x, actions, latent, done, core_state=(), force_argmax=False):
        # -- [unroll_length x batch_size x height x width x channels]
        T, B, h, w, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.long()

        # -- [B x H x W x K]
        state_emb = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
        
        state_emb = state_emb.view(T * B, -1)
        actions = F.one_hot(torch.flatten(actions), self.action_dim).float()
        latent = latent.view(T * B, -1)

        fc_input = torch.cat((state_emb, actions, latent),-1)

        core_input = self.fc(fc_input)

        
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~done).float()
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

        obj_policy = self.out_obj(core_output)
        col_policy = self.out_col(core_output)
        att_policy = self.out_att(core_output)
        
        objs_out = obj_policy.view(-1, self.vocab_size)
        cols_out = col_policy.view(-1, self.colour_size)
        atts_out = att_policy.view(-1, self.att_size)

        if self.training and not force_argmax:
            objs_out = torch.multinomial(F.softmax(objs_out, dim=-1), num_samples=1)
            cols_out = torch.multinomial(F.softmax(cols_out, dim=-1), num_samples=1)
            atts_out = torch.multinomial(F.softmax(atts_out, dim=-1), num_samples=1)
        else:
            # Don't sample when testing.
            objs_out = torch.argmax(objs_out, dim=-1)
            cols_out = torch.argmax(cols_out, dim=-1)
            atts_out = torch.argmax(atts_out, dim=-1)

        obj_policy = obj_policy.view(T, B, self.vision**2, self.vocab_size)
        col_policy = col_policy.view(T, B, self.vision**2, self.colour_size)
        att_policy = att_policy.view(T, B, self.vision**2, self.att_size)

        objs_out = objs_out.view(T, B, self.vision, self.vision, 1)
        cols_out = cols_out.view(T, B, self.vision, self.vision, 1)
        atts_out = atts_out.view(T, B, self.vision, self.vision, 1)
        
        pred = torch.cat((objs_out, cols_out, atts_out),-1)

        return (
            dict(obj_policy_logits=obj_policy, col_policy_logits=col_policy, att_policy_logits=att_policy,
            prediction=pred),
            core_state,
            )


class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim,
                 vocab_size, embedding_dim, colour_size,
                 att_size, sec_emb_dim, vision, use_lstm=False):
        super(Encoder, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vision = vision
        
        self.obj_dim = embedding_dim
        self.sec_emb_dim = sec_emb_dim

        self.embed_object = nn.Embedding(vocab_size, self.obj_dim)
        self.embed_colour = nn.Embedding(colour_size, self.sec_emb_dim)
        self.embed_attribute = nn.Embedding(att_size, self.sec_emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(state_dim*(self.obj_dim+self.sec_emb_dim*2), self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        
        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(self.hidden_dim, self.hidden_dim, 2)

        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim)


    
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
    


    def forward(self, x, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        T, B, h, w, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.long()

        # -- [B x H x W x K]
        state_emb = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)

        state_emb = state_emb.view(T * B, -1)
        core_input = self.fc(state_emb)

        
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            for input in core_input.unbind():
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:

                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        mu = self.fc_mu(core_output)
        logvar = self.fc_logvar(core_output)


        mu = mu.view(T, B, self.latent_dim)
        logvar = logvar.view(T, B, self.latent_dim)


        return (
            dict(mu=mu, logvar=logvar),
            core_state,
            )

