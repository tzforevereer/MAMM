import torch
import os
import random
import logging
import numpy as np
from config import global_config as cfg
from GPTCritic import GPTCritic
import warnings
from tqdm import tqdm


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG)


def seed_initialize(seed):
    # 设置程序的随机种子，以保证程序运行的可复现性

    torch.backends.cudnn.deterministic = True  # 确保在使用 cuDNN 加速时，每次计算结果都是确定的
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 中的自动优化，保证每次计算结果的一致性
    # 设置 PyTorch 中 CPU 的随机种子
    torch.manual_seed(seed)
    # 设置 PyTorch 中所有 GPU 的随机种子
    torch.cuda.manual_seed_all(seed)
    # 设置 Python 标准库中的随机种子
    random.seed(seed)
    # 设置 NumPy 库中的随机种子
    np.random.seed(seed)


def set_path(gpt_path, algorithm, iteration, seed):
    cfg.gpt_path = gpt_path
    cfg.exp_path = os.path.join(
        './experiments', '{}_iter_{}_seed_{}'.format(algorithm, iteration, seed))
    cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
    cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
    cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
    cfg.eval_load_path = cfg.exp_path


def main():
    logging.info("***** Starting Main.py *****")
    if not os.path.exists("./experiments"):
        os.mkdir("./experiments")
    if cfg.mode == 'test':
        pass
    else:  # train
        if cfg.exp_path in ['', 'to be generated']:
            experiments_path = './experiments'
            experiments_path = os.path.join(experiments_path, '{}_iter_{}_seed_{}'.format(
                cfg.algorithm, cfg.iteration, cfg.seed))
            logging.info('save path: %s', cfg.exp_path)

    cfg._init_logging_handler(cfg.mode)
    if cfg.cuda:
        if len(cfg.cuda_device) == 1:
            cfg.multi_gpu = False
            device = torch.device("cuda:{}".format(
                cfg.gpu_idx) if torch.cuda.is_available() else "cpu")
        else:
            pass  # multi-gpu
    else:
        device = torch.device('cpu')
    logging.info('Device: %s', str(device))
    seed_initialize(cfg.seed)

    # 初始化模型

    if cfg.algorithm == 'GPT-Critic':
        model = GPTCritic(device)
    else:
        raise ValueError("Incorrect algorithm name ", cfg.algorithm)
    logging.info("***** Initialize model: %s  *****", cfg.algorithm)
    if cfg.mode == 'train':
        if cfg.algorithm == 'GPT-Critic':
            # 初始化策略，微调GPT-2
            set_path('distilgpt2', cfg.algorithm, 0, cfg.seed)
            model.bc(iteration=0, seed=cfg.seed)
            # 进度条
            iteration_bar = tqdm(total=cfg.iteration)
            for iteration in range(cfg.iteration):
                cfg.exp_path = os.path.join(
                    './experiments', '{}_iter_{}_seed_{}'.format(cfg.algorithm, iteration, cfg.seed))
                policy_path = os.path.join(cfg.exp_path, 'best_model')
                set_path(policy_path, cfg.algorithm, iteration, cfg.seed)
                model = GPTCritic(device)
                

                # Generate rewards for policy evaluation
                model.generate_rewards(iteration=iteration, seed=cfg.seed)

                # Policy evaluation
                model.policy_evaluation(iteration=iteration, seed=cfg.seed)

                # Self-generation with trained gpt-2 and critic
                model.self_generate(iteration=iteration, seed=cfg.seed)

                # Check the dataset performance
                model.dataset_evaluation(iteration=iteration, seed=cfg.seed)


                seed_initialize(cfg.seed)
                # Policy update with fine-tuning the GPT-2 with updated dataset

                set_path('distilgpt2', cfg.algorithm, iteration+1, cfg.seed)
                iteration_bar.update(1)
                m = GPTCritic(device)
                m.bc(iteration=iteration+1, seed=cfg.seed)
            m.generate_rewards(iteration=cfg.iteration, seed=cfg.seed)
            m.dataset_evaluation(iteration=cfg.iteration, seed=cfg.seed)
            iteration_bar.close()

        else:
            raise ValueError("Incorrect algorithm name ", cfg.algorithm)
    else:  # test
        pass


if __name__ == "__main__":
    main()
