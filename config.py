import logging
import time
import os
import json


class _Config:
    def __init__(self):
        self._paramater_init()

    def _paramater_init(self):
        self.gpt_path = './distilgpt2'
        self.model_path = None
        self.result_path = None
        self.vocab_path_eval = None
        self.algorithm = 'GPT-Critic'
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # experiment settings
        self.mode = 'train'
        self.cuda = True
        self.cuda_device = [0]
        self.gpu_idx = 0
        self.multi_gpu = False
        self.exp_no = ''
        

        self.seed = 0
        self.save_log = True
        # training settings

        self.iteration = 60

        # evaluation settings
        self.eval_load_path = ''

    def _init_logging_handler(self, mode):
        # 初始化日志处理器
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and self.mode == 'train':
            file_handler = logging.FileHandler('./log/log_{}_{}_{}_sd{}.txt'.format(self.log_time, mode, self.exp_no, self.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        elif self.mode == 'test':
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            file_handler = logging.FileHandler(eval_log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()
