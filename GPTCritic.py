
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from reader import MultiWozReader
from config import global_config as cfg
import os
import logging
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(256, 1, bias=True)

    def forward(self, input):
        return self.fc2(self.fc1(input))

class GPT2Policy(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class GPTCritic(object):
    def __init__(self, device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        self.reader = MultiWozReader(self.tokenizer)
        self.policy = GPT2Policy.from_pretrained(cfg.gpt_path)
        self.gamma = 0.99

        
        self.critic = Critic()
        self.target_critic = Critic()
    def dataset_evaluation(self, iteration=0, seed=0):
        pass

    def bc(self, iteration=0, seed=0):
        self.save_model()
        

    def save_model(self):
        save_path = os.path.join(cfg.exp_path, 'best_model')
        if not os.path.exists(cfg.exp_path):
            os.mkdir(cfg.exp_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('***** Saving model checkpoint to %s *****', save_path)

        self.policy.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        return save_path

    def remove_model(self, path):
        pass

    def self_generate(self, iteration=0, seed=0, sampling_num=5):
        pass

    def policy_evaluation(self, iteration=0, seed=0):
        pass

    def generate_rewards(self, iteration=0, seed=0):
        pass
