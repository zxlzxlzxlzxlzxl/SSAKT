import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from model import SSAKT, Storage, Embeddings
from model_sakt import SAKT
from dataloader import KTData
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
import json
import pickle


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

parser = argparse.ArgumentParser(description='SSAKT')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers')
parser.add_argument('--emb_dropout', type=float, default=0.2,
                    help='dropout applied to embeddings')
parser.add_argument('--skill_num', type=int, 
                    help='number of skills in the dataset')
parser.add_argument('--problem_num', type=int, default=0,
                    help='number of problems in the dataset')
parser.add_argument('--pmodel', action='store_true')
parser.add_argument('--train_set', type=int, default=1)

parser.add_argument('--embed_dim', type=int, default=256,
                    help='embedding dimension')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=5,
                    help='# of levels (default: 5)')
parser.add_argument('--seq_len', type=int, default=200,
                    help='initial history size (default: 200)')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--seed', type=int, default=284,
                    help='random seed (default: 284)')
parser.add_argument('--nhead', type=int, default=8)

args = parser.parse_args()
if args.dataset == 'assist2009':
    args.skill_num = 110
    args.problem_num = 16891

if args.dataset == 'assist2012':
    args.skill_num = 245
    args.problem_num = 50988

if args.dataset == 'Junyi':
    args.skill_num = 1326
    args.problem_num = 25785

if args.dataset == 'ednet':
    args.skill_num = 1792
    args.problem_num = 13169

elif args.dataset in ['STATICS','statics']:
    args.skill_num = 1223

elif args.dataset in ['assist2015']:
    args.skill_num = 100

elif args.dataset == 'assist2017':
    args.skill_num = 102
    args.problem_num = 3162

torch.manual_seed(args.seed)
#device = torch.device('cuda')
train_loader = DataLoader(KTData(args.dataset ,'train{}'.format(args.train_set),
                             args.skill_num, args.seq_len, args.problem_num>0),
                             batch_size=args.batch_size, shuffle=True)
vaild_loader = DataLoader(KTData(args.dataset, 'valid{}'.format(args.train_set), 
                         args.skill_num, args.seq_len, args.problem_num>0),
                         batch_size=args.batch_size)

model = SSAKT(skill_num=args.skill_num, embed_dim=args.embed_dim, num_channels=[args.nhid] * args.levels, nhead=args.nhead,
                kernel_size=args.ksize, d_k=args.embed_dim//args.nhead, d_v=args.embed_dim//args.nhead, problem_num = args.problem_num, is_pmodel=args.pmodel,
                dropout=args.dropout, emb_dropout=args.emb_dropout)
model = model.to(device)

optimizer = optim.Adam(model.parameters(),lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [14], 0.3)
criterion = nn.BCEWithLogitsLoss()
Q_mat = None
if args.problem_num > 0 and args.pmodel:
    Q_mat = torch.zeros((args.problem_num, args.skill_num)).to(device)
    with open('data/{}/problem_skill_mapping'.format(args.dataset),'r') as f:
        for line in f.readlines():
            line = [ int(i) for i in line.strip().split('\t')]
            Q_mat[line[0]-1][line[1]-1] = 1

def train_on_batch(q_pad, qa_pad, target_pad, p_pad,skill_num):
    target_list = []
    mask = target_pad > -0.9
    target = target_pad[mask].float()
    target_all = target.view(-1)
    output, reg_loss = model(q_pad, qa_pad, p_pad, Q_mat)
    pred_all = output[mask].view(-1)
    loss = criterion(pred_all, target_all) + reg_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, pred_all, target_all

def valid_on_batch(q_pad, qa_pad, target_pad, p_pad,skill_num, test=False):
    with torch.no_grad():
        mask = target_pad > -0.9
        target_all = target_pad[mask].float()
        output, reg_loss = model(q_pad, qa_pad, p_pad, Q_mat, test)
        pred_all = output[mask].view(-1)
    return pred_all, target_all

def train(ep):
    global f
    max_auc = 0
    training_dat = {'epoch':[], 'train':[], 'valid':[]}
    for epoch in range(ep):
        training_dat['epoch'].append(epoch)
        model.train()
        total_loss = 0
        train_epoch_pred = []
        train_epoch_target = []
        for q, qa, target, p in tqdm(train_loader, ncols=75):
            q, qa, target, p = q.to(device), qa.to(device), target.to(device), p.to(device)
            loss, pred_all, target_all = train_on_batch(q, qa, target, p, args.skill_num)
            total_loss += loss
            train_epoch_pred.append(pred_all.detach())
            train_epoch_target.append(target_all.detach())
        #scheduler.step()
        train_epoch_pred = torch.cat(train_epoch_pred).cpu()
        train_epoch_target = torch.cat(train_epoch_target).cpu()
        train_auc = metrics.roc_auc_score(train_epoch_target, train_epoch_pred)
        training_dat['train'].append(train_auc)
        print('epoch %d,total loss: %f' % (epoch, total_loss), flush=True)
        print('train auc: %f' % train_auc)
        model.eval()
        valid_epoch_pred = []
        valid_epoch_target = []
        for valid_q, valid_qa, valid_target, valid_p in tqdm(vaild_loader, ncols=75):
            valid_q, valid_qa, valid_target, valid_p = valid_q.to(device), valid_qa.to(device), valid_target.to(device), valid_p.to(device)
            valid_pred_all, valid_target_all = valid_on_batch(valid_q, valid_qa, valid_target, valid_p, args.skill_num)
            valid_epoch_pred.append(valid_pred_all)
            valid_epoch_target.append(valid_target_all)
        valid_epoch_pred = torch.cat(valid_epoch_pred).cpu()
        valid_epoch_target = torch.cat(valid_epoch_target).cpu()
        auc = metrics.roc_auc_score(valid_epoch_target, valid_epoch_pred)
        training_dat['valid'].append(auc)
        print('valid auc:%f' % auc, flush=True)
        if auc > max_auc:
            max_auc = auc
            torch.save(model, 'model.pkl')
    return training_dat
        

train_dat = train(args.epochs)
#with open('data/'+args.dataset + '/results.json', 'w') as out:
#    json.dump(train_dat, out)
model = torch.load('model.pkl')
test_epoch_pred = []
test_epoch_target = []
test_loader = DataLoader(KTData(args.dataset, 'test', args.skill_num, args.seq_len,args.problem_num>0), batch_size=1)
for test_q, test_qa, test_target, test_p in tqdm(test_loader):
    test_q, test_qa, test_target, test_p = test_q.to(device), test_qa.to(device), test_target.to(device), test_p.to(device)
    test_pred_all, test_target_all = valid_on_batch(test_q, test_qa, test_target, test_p, args.skill_num, True)
    test_epoch_pred.append(test_pred_all)
    test_epoch_target.append(test_target_all)
test_epoch_pred = torch.cat(test_epoch_pred).detach().cpu()
test_epoch_target = torch.cat(test_epoch_target).detach().cpu()
auc = metrics.roc_auc_score(test_epoch_target, test_epoch_pred)
print("test auc:%f" % auc)
#pickle.dump(Storage, open('storage.pkl','wb'))
pickle.dump(Embeddings, open('{}_embeddings.pkl'.format(args.dataset),'wb'))