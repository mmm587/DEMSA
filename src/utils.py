import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import random

from src.processor import get_tokenizer, convert_to_features


def set_random_seed(seed: int):
    print("Seed: {}".format(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_appropriate_dataset(data):
    from configs import args
    tokenizer = get_tokenizer(args.model)
    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
    all_segment_ids = torch.tensor(np.array([f.segment_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )

    return dataset


def set_up_data_loader(args):
    with open(f"data/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step) * args.n_epochs)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

    return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps


def save_model(hyp_params, model):
    torch.save(model, f'models/{hyp_params.name}.pt')


def load_model(hyp_params):
    model = torch.load(f'models/{hyp_params.name}.pt')
    return model
