import pickle

import torch
from torch import nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from src import models, ProjModel
from src.metrics import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def initiate(hyp_params, num_train_optimization_steps):
    model = getattr(models, hyp_params.name+'Model')(hyp_params)
    proj_model = getattr(ProjModel, "ClassificationLayer")(hyp_params)
    model.to(hyp_params.device)
    proj_model.to(hyp_params.device)
    if hyp_params.use_cuda:
        model = model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # Prepare optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=hyp_params.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=hyp_params.warmup_proportion * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps, )
    criterion = getattr(nn, hyp_params.criterion)()

    settings = {'model': model, 'proj_model': proj_model, 'optimizer': optimizer, 'criterion': criterion, 'scheduler': scheduler}
    return settings


def train_model(settings, hyp_params, train_loader):
    model = settings['model']
    proj_model = settings['proj_model']
    optimizer = settings['optimizer']
    scheduler = settings['scheduler']
    criterion = settings['criterion']

    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(tqdm(train_loader, desc="Train Iteration")):
        batch = tuple(t.to(hyp_params.device) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        fusion_h, denoised_fusion, _ = model(input_ids, visual, acoustic, segment_ids, input_mask)
        loss_denoising = criterion(denoised_fusion, fusion_h)
        logits = proj_model(denoised_fusion, fusion_h)
        loss = criterion(logits.view(-1), label_ids.view(-1))  
        total_loss = loss_denoising*0.5 + loss*1.0
        if hyp_params.gradient_accumulation_step > 1:
            total_loss = total_loss / hyp_params.gradient_accumulation_step

        total_loss.backward()

        tr_loss += total_loss.item()
        nb_tr_steps += 1

        if (step + 1) % hyp_params.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def evaluate_model(settings, hyp_params, valid_loader):
    model = settings['model']
    proj_model = settings['proj_model']
    criterion = settings['criterion']
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_loader, desc="Valid Iteration")): 
            batch = tuple(t.to(hyp_params.device) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            fusion_h, denoised_fusion,_ = model(input_ids, visual, acoustic, segment_ids, input_mask)
            loss_denoising = criterion(denoised_fusion, fusion_h)
            logits = proj_model(denoised_fusion, fusion_h)
            loss = criterion(logits.view(-1), label_ids.view(-1))  
            total_loss = loss_denoising + loss

            if hyp_params.gradient_accumulation_step > 1:
                total_loss = total_loss / hyp_params.gradient_accumulation_step

            dev_loss += total_loss.item()
            nb_dev_steps += 1
    return dev_loss / nb_dev_steps


def test_model(model, ProjModel, hyp_params, test_loader):
    model.eval()
    preds = []
    labels = []
    features = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Iteration"):
            batch = tuple(t.to(hyp_params.device) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            fusion_h, denoised_fusion, f_squeeze = model(input_ids, visual, acoustic, segment_ids, input_mask)
            features.append([fusion_h, denoised_fusion])
            logits = ProjModel(denoised_fusion, fusion_h)
            if hyp_params.dataset == 'iemocap':
                logits = logits.view(-1, 2)
                label_ids = label_ids.view(-1)

            original_shape = (128, 8)
            target_shape = (8, 8)
            rows_ratio = original_shape[0] // target_shape[0]
            cols_ratio = original_shape[1] // target_shape[1]
            averaged_data = np.zeros(target_shape)
            for i in range(target_shape[0]):
                for j in range(target_shape[1]):
                    block = f_squeeze[i * rows_ratio:(i + 1) * rows_ratio, j * cols_ratio:(j + 1) * cols_ratio]
                    averaged_data[i, j] = block.mean()

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)
        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels, features


def train(hyp_params, train_loader, valid_loader, test_loader, num_train_optimization_steps):
    settings = initiate(hyp_params, num_train_optimization_steps)
    test_mae = []
    test_corr = []
    test_f_score = []
    test_accuracies = []
    test_features = []
    acc, mae, corr, f_score = 0, 0, 0, 0
    for epoch in range(1, hyp_params.n_epochs + 1):  
        train_loss = train_model(settings, hyp_params, train_loader)
        valid_loss = evaluate_model(settings, hyp_params, valid_loader)

        print("\n" + "-" * 50)
        print('Epoch {:2d}| Train Loss {:5.4f} | Valid Loss {:5.4f} '.format(epoch, train_loss, valid_loss))
        print("-" * 50)

        results, truths, features = test_model(settings['model'], settings['proj_model'], hyp_params, test_loader)

        if hyp_params.dataset == "mosei":
            acc, mae, corr, f_score = eval_mosei_senti(results, truths, True)
        elif hyp_params.dataset == 'mosi':
            acc, mae, corr, f_score = eval_mosi(results, truths, True)
        elif hyp_params.dataset == 'iemocap':
            acc, f_score = eval_iemocap(results, truths)
        test_mae.append(mae)
        test_corr.append(corr)
        test_f_score.append(f_score)
        test_accuracies.append(acc)
        test_features.append(features)

    with open('./test_features.pkl', 'wb') as f:
        pickle.dump(test_features, f)

    print('The test results:\n| MAE {:5.4f} | Corr {:5.4f} | F1 {:5.4f} | ACC {:5.4f}'.format(min(test_mae),
                                                                                                  max(test_corr),
                                                                                                  max(test_f_score),
                                                                                                  max(test_accuracies)))
