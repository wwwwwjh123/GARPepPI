# -*- coding: utf-8 -*-
"""Training and 5-fold cross-validation for GARPepPI."""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import xlwt

from data_loader import *
from GARPepPI import *
from args import *

device = torch.device('cuda')


def create_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    return Variable(tensor)


def validation(model, device, loader):
    model.eval()
    total_labels = torch.Tensor()
    total_preds  = torch.Tensor()
    total_scores = torch.Tensor()
    print(f"Validation on {len(loader.dataset)} samples...")
    with torch.no_grad():
        for p1, p2, G1, dmap1, G2, dmap2, y in loader:
            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)
            output_score = model(dgl.batch(G1), pad_dmap1, dgl.batch(G2), pad_dmap2)
            output = torch.round(output_score.squeeze(1))
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            total_scores = torch.cat((total_scores.cpu(), output_score.cpu()), 0)
            total_preds  = torch.cat((total_preds.cpu(), output.cpu()), 0)
    return (
        total_labels.cpu().numpy().flatten(),
        total_preds.cpu().numpy().flatten(),
        total_scores.cpu().numpy().flatten(),
    )


def test(model, device, loader, k):
    model.eval()
    total_labels = torch.Tensor()
    total_preds  = torch.Tensor()
    total_scores = torch.Tensor()
    print(f"Test on {len(loader.dataset)} samples...")
    workbook = xlwt.Workbook('encoding=utf-8')
    sheet = workbook.add_sheet('sheet1', cell_overwrite_ok=True)
    headers = ["index", "receptor", "peptide", "label", "predict_score", "predict_label"]
    for col, h in enumerate(headers):
        sheet.write(0, col, h)
    row = 1
    with torch.no_grad():
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, y) in enumerate(loader):
            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)
            predict_score = model(dgl.batch(G1), pad_dmap1, dgl.batch(G2), pad_dmap2)
            predict_label = torch.round(predict_score.squeeze(1))

            ps_np = predict_score.cpu().numpy().tolist()
            pl_np = predict_label.cpu().numpy().tolist()

            for i in range(len(ps_np)):
                sheet.write(row, 0, row - 1)
                sheet.write(row, 1, str(p1[i]))
                sheet.write(row, 2, str(p2[i]))
                sheet.write(row, 3, str(y[i].item()))
                sheet.write(row, 4, str(ps_np[i]))
                sheet.write(row, 5, str(pl_np[i]))
                row += 1

            score_np  = predict_score.cpu().numpy()
            label_np  = predict_label.cpu().numpy()
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            total_scores = torch.cat((total_scores.cpu(), predict_score.cpu()), 0)
            total_preds  = torch.cat((total_preds.cpu(), label_np), 0)

    os.makedirs(os.path.dirname(test_5fold_path.rstrip('/') + '/'), exist_ok=True)
    workbook.save(test_5fold_path + f'fold{k}/result.xls')

    return (
        total_labels.cpu().numpy().flatten(),
        total_preds.cpu().numpy().flatten(),
        total_scores.cpu().numpy().flatten(),
    )


def train(trainArgs):
    if not os.path.exists(test_5fold_path):
        os.makedirs(test_5fold_path, exist_ok=True)
    for fold_num in range(1, 6):
        os.makedirs(os.path.join(test_5fold_path, f'fold{fold_num}'), exist_ok=True)

    all_protein1, all_protein2, all_Y = [], [], []
    with open(actions_file, 'r') as f:
        for line in f:
            row = line.rstrip().split('\t')
            all_protein1.append(row[0])
            all_protein2.append(row[1])
            all_Y.append(float(row[2]))

    criterion = torch.nn.BCELoss()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_Y, all_Y), 1):
        print(f"\n===== Fold {fold} =====")
        with open(rst_file, 'a+') as fp:
            fp.write(f"===== Fold {fold} =====\n")

        train_valid_p1 = np.array(all_protein1)[train_idx]
        train_valid_p2 = np.array(all_protein2)[train_idx]
        train_valid_Y  = np.array(all_Y)[train_idx]

        test_p1 = np.array(all_protein1)[test_idx]
        test_p2 = np.array(all_protein2)[test_idx]
        test_Y  = np.array(all_Y)[test_idx]

        train_size = train_valid_p1.shape[0]
        valid_size = int(train_size * 0.05)

        half = int(valid_size // 2)
        train_p1 = np.concatenate((train_valid_p1[:half], train_valid_p1[-half:]))
        train_p2 = np.concatenate((train_valid_p2[:half], train_valid_p2[-half:]))
        train_Y  = np.concatenate((train_valid_Y[:half], train_valid_Y[-half:]))

        valid_p1 = train_valid_p1[half:-half]
        valid_p2 = train_valid_p2[half:-half]
        valid_Y  = train_valid_Y[half:-half]

        train_ds  = MyDataset(train_p1.tolist(), train_p2.tolist(), train_Y.tolist())
        valid_ds  = MyDataset(valid_p1.tolist(), valid_p2.tolist(), valid_Y.tolist())
        test_ds   = MyDataset(test_p1.tolist(),  test_p2.tolist(),  test_Y.tolist())

        train_loader  = DataLoader(train_ds,  batch_size=batchsize, shuffle=True, drop_last=False, collate_fn=collate)
        valid_loader  = DataLoader(valid_ds,  batch_size=batchsize, shuffle=True, drop_last=False, collate_fn=collate)
        test_loader   = DataLoader(test_ds,   batch_size=batchsize, shuffle=True, drop_last=False, collate_fn=collate)

        model = GATPPI(modelArgs).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=trainArgs['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        best_acc = 0
        for epoch in range(trainArgs['epochs']):
            model.train()
            total_loss = 0
            correct = 0
            n_batches = 0

            for p1, p2, G1, dmap1, G2, dmap2, y in train_loader:
                y_pred = model(dgl.batch(G1), pad_dmap(dmap1), dgl.batch(G2), pad_dmap(dmap2))
                correct += torch.eq(torch.round(y_pred.squeeze(1).type(torch.DoubleTensor)),
                                    y.type(torch.DoubleTensor)).data.sum()
                loss = criterion(y_pred.squeeze(1).type(torch.DoubleTensor),
                                 y.type(torch.DoubleTensor))
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                total_loss += loss.data
                optimizer.zero_grad()
                loss.backward()
                if trainArgs['grad_clip'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), trainArgs['grad_clip'])
                optimizer.step()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            acc = correct.numpy() / len(train_loader.dataset)

            print(f"Epoch {epoch + 1}/{trainArgs['epochs']}  loss={avg_loss:.4f}  acc={acc:.4f}")

            # Validation
            val_labels, val_preds, val_scores = validation(model, device, valid_loader)
            val_acc  = accuracy_score(val_labels, val_preds)
            val_prec = precision_score(val_labels, val_preds)
            val_rec  = recall_score(val_labels, val_preds)
            val_f1   = f1_score(val_labels, val_preds)
            val_auc  = roc_auc_score(val_labels, val_scores)
            val_auprc = average_precision_score(val_labels, val_scores)
            cm = confusion_matrix(val_labels, val_preds)
            val_sen = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            val_spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            val_mcc = ((cm[0, 0] * cm[1, 1] - cm[0, 1] * cm[1, 0]) /
                       np.sqrt((cm[1, 1] + cm[0, 1]) * (cm[1, 1] + cm[1, 0]) *
                                (cm[0, 0] + cm[0, 1]) * (cm[0, 0] + cm[1, 0]))) if (
                           (cm[1, 1] + cm[0, 1]) * (cm[1, 1] + cm[1, 0]) *
                           (cm[0, 0] + cm[0, 1]) * (cm[0, 0] + cm[1, 0]) > 0) else 0

            print(f"  val - acc:{val_acc:.4f}  prec:{val_prec:.4f}  rec:{val_rec:.4f}  "
                  f"f1:{val_f1:.4f}  auc:{val_auc:.4f}  spec:{val_spec:.4f}  "
                  f"mcc:{val_mcc:.4f}  auprc:{val_auprc:.4f}  sen:{val_sen:.4f}")

            with open(rst_file, 'a+') as fp:
                fp.write(f"epoch:{epoch + 1}\tacc={acc:.4f}\tloss={avg_loss.item():.4f}\t"
                          f"val_acc={val_acc:.4f}\tval_prec={val_prec:.4f}\tval_rec={val_rec:.4f}\t"
                          f"val_f1={val_f1:.4f}\tval_auc={val_auc:.4f}\tval_spec={val_spec:.4f}\t"
                          f"val_mcc={val_mcc:.4f}\tval_auprc={val_auprc:.4f}\tval_sen={val_sen:.4f}\n")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"{pkl_path}_of_fold{fold}.pkl")
                print(f"  -> saved best model (val_acc={best_acc:.4f})")

        # Test on best model
        model.load_state_dict({k.replace('module.', ''): v
                               for k, v in torch.load(f"{pkl_path}_of_fold{fold}.pkl").items()},
                              strict=False)
        test_labels, test_preds, test_scores = test(model, device, test_loader, fold)
        t_acc  = accuracy_score(test_labels, test_preds)
        t_prec = precision_score(test_labels, test_preds)
        t_rec  = recall_score(test_labels, test_preds)
        t_f1   = f1_score(test_labels, test_preds)
        t_auc  = roc_auc_score(test_labels, test_scores)
        t_auprc = average_precision_score(test_labels, test_scores)
        cm = confusion_matrix(test_labels, test_preds)
        t_sen = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        t_spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        t_mcc = ((cm[0, 0] * cm[1, 1] - cm[0, 1] * cm[1, 0]) /
                 np.sqrt((cm[1, 1] + cm[0, 1]) * (cm[1, 1] + cm[1, 0]) *
                          (cm[0, 0] + cm[0, 1]) * (cm[0, 0] + cm[1, 0]))) if (
                      (cm[1, 1] + cm[0, 1]) * (cm[1, 1] + cm[1, 0]) *
                      (cm[0, 0] + cm[0, 1]) * (cm[0, 0] + cm[1, 0]) > 0) else 0

        print(f"Test fold{fold} - acc:{t_acc:.4f}  prec:{t_prec:.4f}  rec:{t_rec:.4f}  "
              f"f1:{t_f1:.4f}  auc:{t_auc:.4f}  spec:{t_spec:.4f}  "
              f"mcc:{t_mcc:.4f}  auprc:{t_auprc:.4f}  sen:{t_sen:.4f}")

        with open(rst_file, 'a+') as fp:
            fp.write(f"TEST\tacc={t_acc:.4f}\tprec={t_prec:.4f}\trec={t_rec:.4f}\t"
                      f"f1={t_f1:.4f}\tauc={t_auc:.4f}\tspec={t_spec:.4f}\t"
                      f"mcc={t_mcc:.4f}\tauprc={t_auprc:.4f}\tsen={t_sen:.4f}\n")
