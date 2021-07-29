import math
import argparse
import time, datetime
import numpy as np
import torch
import torch.nn as nn
from lib import utils
from models import model


parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type = int, default = 5,
                    help = 'a time step is 5 mins')
parser.add_argument('--H', type = int, default = 12,
                    help = 'history steps')
parser.add_argument('--P', type = int, default = 12,
                    help = 'present steps')
parser.add_argument('--F', type = int, default = 12,
                    help = 'future steps')
parser.add_argument('--L', type = int, default = 2,
                    help = 'number of encoder-decoder layers')
parser.add_argument('--K', type = int, default = 3,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 8,
                    help = 'dims of each head attention outputs')
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 8,
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 100,
                    help = 'epoch to run')
parser.add_argument('--patience', type = int, default = 10,
                    help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--weight_decay', type=float, default = 0.00001,
                    help='weight decay')
parser.add_argument('--decay_epoch', type=int, default = 5,
                    help = 'decay epoch')
parser.add_argument('--path', default = './',
                    help = 'traffic file')
parser.add_argument('--dataset', default = 'pems04',
                    help = 'Traffic dataset name')
parser.add_argument('--load_model', default = 'F',
                    help = 'Set T if pretrained model is to be loaded before training start')
parser.add_argument('--seed', type =int,  default=10, 
                    help = 'initialization')
parser.add_argument('--dhm', type=bool, default=True,
                    help='whether to use the DHM module')
parser.add_argument('--max_step', type=int, default=6,
                    help = 'max recurrent step')
parser.add_argument('--dhm_weight', type=float, default=0.001,
                    help = 'the penalty weight for the DHM')
parser.add_argument('--epsilon', type=float, default=0.0001,
                    help = 'threshold for the DHM')
parser.add_argument('--recollection_weight', type=float, default=0.01,
                    help = 'recollection decoder weight')

args = parser.parse_args()

LOG_FILE = args.path+'log('+args.dataset+')'
MODEL_FILE = args.path+'BISTAT('+args.dataset+')'


start = time.time()

log = open(LOG_FILE, 'w')
utils.log_string(log, str(args)[10 : -1])

# load data
utils.log_string(log, 'loading data...')
(trainZ, trainX, trainTE, trainY, valZ, valX, valTE, valY, testZ, testX, testTE, testY, SE,
 mean, std) = utils.loadData(args)

utils.log_string(log, 'trainZ: %s\t trainX: %s\t trainY: %s' % (trainZ.shape, trainX.shape, trainY.shape))
utils.log_string(log, 'valZ: %s\t valX: %s\t valY: %s' % (valZ.shape, valX.shape, valY.shape))
utils.log_string(log, 'testZ: %s\t testX: %s\t testY: %s' % (testZ.shape, testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transform data to tensors
trainZ = torch.FloatTensor(trainZ).to(device)
trainX = torch.FloatTensor(trainX).to(device)
trainTE = torch.LongTensor(trainTE).to(device)
trainY = torch.FloatTensor(trainY).to(device)
valZ = torch.FloatTensor(valZ).to(device)
valX = torch.FloatTensor(valX).to(device)
valTE = torch.LongTensor(valTE).to(device)
valY = torch.FloatTensor(valY).to(device)
testZ = torch.FloatTensor(testZ).to(device)
testX = torch.FloatTensor(testX).to(device)
testTE = torch.LongTensor(testTE).to(device)
testY = torch.FloatTensor(testY).to(device)
SE = torch.FloatTensor(SE).to(device)

utils.init_seed(args.seed)

TEmbsize = (24*60//args.time_slot)+7 #number of slots in a day + number of days in a week
hidden_size = args.K*args.d
num_sensor = trainX.shape[-1]

# define the Bi-STAT model
bi_stat = model.BISTAT(args.K, args.d, SE.shape[1], TEmbsize, args.P, args.F, args.H, num_sensor, args.L, args.epsilon, hidden_size, args.max_step, args.dhm, device).to(device)
optimizer = torch.optim.Adam(bi_stat.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


# init model
for p in bi_stat.parameters():
   if p.dim() > 1:
       nn.init.xavier_uniform_(p)
   else:
       nn.init.uniform_(p)
# print parameters  
# utils.print_model_parameters(bi_stat, only_num=False)


utils.log_string(log, '**** training model ****')
if args.load_model == 'T':
    utils.log_string(log, 'loading pretrained model from %s' % MODEL_FILE)
    bi_stat.load_state_dict(torch.load(MODEL_FILE))

num_train = trainX.shape[0]
num_val = valX.shape[0]
wait = 0
val_loss_min = np.inf

val_time = []
train_time = []
for epoch in range(args.max_epoch):
    if wait >= args.patience:
        utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
        break
    # shuffle
    permutation = np.random.permutation(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainY = trainY[permutation]
    trainZ = trainZ[permutation]
    
    # train loss
    start_train = time.time()
    train_loss = 0
    num_batch = math.ceil(num_train / args.batch_size)
    loss = []
    for batch_idx in range(num_batch):
        bi_stat.train()
        optimizer.zero_grad()
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        batchX = trainX[start_idx : end_idx]        
        batchTE = trainTE[start_idx : end_idx]
        batchlabel1 = trainY[start_idx : end_idx]
        batchlabel2 = trainZ[start_idx : end_idx]
        batchpred1, batchpred2, dhm0_S, dhm1_S, dhm0_T, dhm1_T = bi_stat(batchX, SE, batchTE, flag='train')
        P_t_S = dhm0_S + dhm1_S
        P_t_T = dhm0_T + dhm1_T
        batchloss_dec1 = model.mae_loss(batchpred1, batchlabel1, device)
        batchloss_dec2 = model.mae_loss(batchpred2, batchlabel2, device)
        batchloss_dhm = torch.mean(torch.mean(torch.mean(torch.sum(P_t_S,-1),-1),-1),-1)+\
        torch.mean(torch.mean(torch.mean(torch.sum(P_t_T,-1),-1),-1),-1)
        batchloss = batchloss_dec1 + args.recollection_weight*batchloss_dec2 + args.dhm_weight*batchloss_dhm
        loss.append(batchloss)
        if (batch_idx+1) % 100 == 0:
            print("Batch: ", batch_idx+1, "out of", num_batch, end=" | ")
            print("Loss: ", batchloss.item(), flush=True)
        batchloss.backward()
        optimizer.step()
        train_loss += batchloss.item() * (end_idx - start_idx)
    end_train = time.time()

    train_time.append(end_train-start_train)

    train_loss /= num_train
    

    # val loss
    start_val = time.time()
    val_loss = 0
    num_batch = math.ceil(num_val / args.batch_size)
    valPred_dec1 = []
    valPred_dec2 = []
    for batch_idx in range(num_batch):
        bi_stat.eval()
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        batchX = valX[start_idx : end_idx]
        batchTE = valTE[start_idx : end_idx]
        batchlabel1 = valY[start_idx : end_idx]
        batchlabel2 = valZ[start_idx : end_idx]
        batchpred_dec1, batchpred_dec2, _, _, _, _ = bi_stat(batchX, SE, batchTE, flag='val')
        valPred_dec1.append(batchpred_dec1.detach().cpu().numpy())
        valPred_dec2.append(batchpred_dec2.detach().cpu().numpy())
        batchloss = model.mae_loss(batchpred1, batchlabel1, device)
        val_loss += batchloss.item() * (end_idx - start_idx)

    end_val = time.time()
    val_time.append(end_val - start_val)

    valPred_dec1 = np.concatenate(valPred_dec1, axis=0)
    valPred_dec2 = np.concatenate(valPred_dec2, axis=0)

    val_loss /= num_val
    
    utils.log_string(
        log,
        '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
         args.max_epoch, end_train - start_train, end_val - start_val))
    utils.log_string(
        log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
    if val_loss <= val_loss_min:
        utils.log_string(
            log,
            'val loss decrease from %.4f to %.4f, saving model to %s' %
            (val_loss_min, val_loss, MODEL_FILE))
        wait = 0
        val_loss_min = val_loss
        torch.save(bi_stat.state_dict(), MODEL_FILE)
    else:
        wait += 1

        
    val_mae_dec1, val_rmse_dec1, val_mape_dec1 = utils.metric(valPred_dec1, valY.cpu().numpy())
    utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                 (val_mae_dec1, val_rmse_dec1, val_mape_dec1 * 100))

    val_mae_dec2, val_rmse_dec2, val_mape_dec2 = utils.metric(valPred_dec2, valZ.cpu().numpy())
    utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                 (val_mae_dec2, val_rmse_dec2, val_mape_dec2 * 100))

utils.log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
utils.log_string(log, "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % MODEL_FILE)
bi_stat.load_state_dict(torch.load(MODEL_FILE))
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')

num_test = testX.shape[0]

trainPred = []
num_batch = math.ceil(num_train / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
    batchX = trainX[start_idx : end_idx]
    batchTE = trainTE[start_idx : end_idx]
    batchlabel = trainY[start_idx : end_idx]
    batchpred, _, _, _, _ = bi_stat(batchX, SE, batchTE, flag='test')
    trainPred.append(batchpred.detach().cpu().numpy())
trainPred = np.concatenate(trainPred, axis = 0)

valPred = []
num_batch = math.ceil(num_val / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
    batchX = valX[start_idx : end_idx]
    batchTE = valTE[start_idx : end_idx]
    batchlabel = valY[start_idx : end_idx]
    batchpred, _, _,_, _ = bi_stat(batchX, SE, batchTE, flag='test')
    valPred.append(batchpred.detach().cpu().numpy())
valPred = np.concatenate(valPred, axis = 0)

testPred = []
num_batch = math.ceil(num_test / args.batch_size)
start_test = time.time()

for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    batchX = testX[start_idx : end_idx]
    batchTE = testTE[start_idx : end_idx]
    batchlabel = testY[start_idx : end_idx]
    batchpred, _, _,_, _ = bi_stat(batchX, SE, batchTE, flag='test')
    testPred.append(batchpred.detach().cpu().numpy())
end_test = time.time()
testPred = np.concatenate(testPred, axis = 0)

trainY = trainY.cpu().numpy()
valY = valY.cpu().numpy()
testY = testY.cpu().numpy()

train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
test_mae, test_rmse, test_mape = utils.metric(testPred, testY)
utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
                 (train_mae, train_rmse, train_mape * 100))
utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                 (val_mae, val_rmse, val_mape * 100))
utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
                 (test_mae, test_rmse, test_mape * 100))
utils.log_string(log, 'performance in each prediction step')

MAE, RMSE, MAPE = [], [], []
for q in range(args.Q):
    mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    utils.log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                     (q + 1, mae, rmse, mape * 100))
average_mae = np.mean(MAE)
average_rmse = np.mean(RMSE)
average_mape = np.mean(MAPE)

utils.log_string(
    log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
    (average_mae, average_rmse, average_mape * 100))
end = time.time()
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
log.close()
