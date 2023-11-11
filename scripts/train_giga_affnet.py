import argparse
from pathlib import Path
from datetime import datetime
import wandb
import numpy as np

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy, Precision, Recall

import torch
# from torch.utils import tensorboard
import torch.nn.functional as F

#from vgn.dataset_pc import DatasetPCOcc
from vgn.dataset_affnet_voxel import DatasetAffnetVoxelOccFile
from vgn.networks import get_network, load_network

LOSS_KEYS = ['loss_all', 'loss_qual', 'loss_affrdnce', 'loss_occ']

def main(args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": args.num_workers, "pin_memory": True, "persistent_workers": True} if use_cuda else {}

    # create log directory
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    if args.savedir == '':
        description = "{}_dataset={},augment={},net=6d_{},batch_size={},lr={:.0e},{}".format(
            time_stamp,
            args.dataset.name,
            args.augment,
            args.net,
            args.batch_size,
            args.lr,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
        # meshdir = logdir / "meshes"
        # meshdir.mkdir(parents=True, exist_ok=True)
    else:
        logdir = Path(args.savedir)
    
    # filename = 'summary_%s.txt' % (args.dataset_raw.name)
    # with open(filename, 'r') as f:
    #     note = (";").join(f.readlines()[1:5]).replace('\n', '')

    if args.log_wandb:
        wandb.init(config=args, project="6dgrasp", entity="irosa-ias", id=args.net+'_'+args.dataset.name+'_'+time_stamp)#, notes=note)

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.dataset_raw, args.batch_size, args.val_split, args.augment, kwargs)

    # build the network or load
    if args.load_path == '':
        net = get_network(args.net).to(device) # <- Changed to predict only grasp quality (check inside)
    else:
        net = load_network(args.load_path, device, args.net)
    
    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min')

    metrics = {
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])), # out[1][0] => y_pred -> quality
                                                                              # out[2][0] => y -> quality
                                                                              # ^ Refer def _update() returns
        "precision": Precision(lambda out: (torch.round(out[1][0]), out[2][0])),
        "recall": Recall(lambda out: (torch.round(out[1][0]), out[2][0])),
        "affrdnce_accuracy": Accuracy(lambda out: (torch.round(out[1][2]), out[2][2]), is_multilabel=True), # out[1][1] => y_pred -> affrdnces
        "affrdnce_precision": Precision(lambda out: (torch.round(out[1][2]), out[2][2]), is_multilabel=True, average=True),
        "affrdnce_recall": Recall(lambda out: (torch.round(out[1][2]), out[2][2]), is_multilabel=True, average=True),
        # "affrdnce_labelwise_accuracy": # TODO: from https://github.com/pytorch/ignite/issues/513#issuecomment-488983281
    }
    for k in LOSS_KEYS:
        metrics[k] = Average(lambda out, sk=k: out[3][sk])
    
    if args.log_wandb:
        wandb.watch(net, log_freq=100)

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, scheduler, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True, dynamic_ncols=True, disable=args.silence).attach(trainer)

    #train_writer, val_writer = create_summary_writers(net, device, logdir)
    

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics

        msg = 'Train'
        for k, v in metrics.items():
            if args.log_wandb:
                wandb.log({'train_'+k:v})                
            msg += f' {k}: {v:.4f}'
        print(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        # out = evaluator.state.output

        msg = 'Val'
        for k, v in metrics.items():
            if args.log_wandb:
                wandb.log({'val_'+k:v})
            msg += f' {k}: {v:.4f}'
        print(msg)

    @evaluator.on(Events.COMPLETED)
    def reduct_step(engine):
    # engine is evaluator
    # engine.metrics is a dict with metrics, e.g. {"loss": val_loss_value, "acc": val_acc_value}
        scheduler.step(evaluator.state.metrics['loss_all'])

    def default_score_fn(engine):
        score = engine.state.metrics['accuracy']
        return score

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        "vgn",
        n_saved=None, # Changed from 1. Save everythinggg
        require_empty=True,
    )
    best_checkpoint_handler = ModelCheckpoint(
        logdir,
        "best_vgn",
        n_saved=1,
        score_name="val_acc",
        score_function=default_score_fn,
        require_empty=True,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: net}
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, best_checkpoint_handler, {args.net: net}
    )

    # run the training loop
    epoch_length = int(args.epoch_length_frac*len(train_loader))
    trainer.run(train_loader, max_epochs=args.epochs, epoch_length=epoch_length)


def create_train_val_loaders(root, root_raw, batch_size, val_split, augment, kwargs):
    # load the dataset

    dataset = DatasetAffnetVoxelOccFile(root, root_raw)

    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )
    return train_loader, val_loader


def prepare_batch(batch, device):
    pc, (label, rotations, aff_labels), pos, pos_occ, occ_value = batch
    pc = pc.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    aff_labels = aff_labels.float().to(device)
    # width = width.float().to(device)
    pos.unsqueeze_(1) # B, 1, 3
    pos = pos.float().to(device)
    pos_occ = pos_occ.float().to(device)
    occ_value = occ_value.float().to(device)
    return pc, (label, rotations, aff_labels, occ_value), pos, pos_occ


def select(out):
    qual_out, rot_out, aff_out, occ = out
    rot_out = rot_out.squeeze(1)
    occ = torch.sigmoid(occ) # to probability
    return qual_out.squeeze(-1), rot_out, aff_out, occ


def loss_fn(y_pred, y):
    label_pred, rotation_pred, aff_preds, occ_pred = y_pred
    label, rotations, affs, occ = y
    loss_qual = _qual_loss_fn(label_pred, label).mean()
    loss_rot = _rot_loss_fn(rotation_pred, rotations)
    loss_affrdnce = _affrdnce_loss_fn(aff_preds, affs, label.bool()) # only consider loss for postitive grasp labels. No mean here since dice loss is over batch
    # loss_width = _width_loss_fn(width_pred, width)
    loss_occ = _occ_loss_fn(occ_pred, occ).mean()
    loss = loss_qual + (label * (loss_rot)).mean() + (2*loss_occ) + (3*loss_affrdnce)
    loss_dict = {'loss_qual': loss_qual,
                'loss_rot': loss_rot,
                'loss_affrdnce': loss_affrdnce,
                'loss_occ': loss_occ,
                'loss_all': loss
                }
    return loss, loss_dict


def _qual_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="none")


def _rot_loss_fn(pred, target):
    loss0 = _quat_loss_fn(pred, target[:, 0])
    loss1 = _quat_loss_fn(pred, target[:, 1])
    return torch.min(loss0, loss1)


def _quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

def _affrdnce_loss_fn(pred, target, labels_to_consider):
    # we only consider the loss for the true grasp labels
    # Adapted from EstimationLoss in 3DAffordanceNet. We use the DICE loss over the batch instead of over seg points!

    pred = pred[labels_to_consider]
    target = target[labels_to_consider]
    if pred.shape[0] == 0:
        return 0.0
    
    temp1 = -torch.mul(pred,
                    torch.mul(1-target, torch.log(1-pred+1e-6)))
    temp2 = -torch.mul((1-pred),
                    torch.mul(target, torch.log(pred+1e-6)))
    temp = temp1+temp2
    CELoss = torch.mean(torch.mean(temp, 1)) # mean over affrdnce types, mean over batch

    intersection_positive = torch.sum(pred*target, 0) # sum over batch for dice!
    cardinality_positive = torch.sum(pred+target, 0)
    dice_positive = (intersection_positive+1e-6) / \
        (cardinality_positive+1e-6)

    intersection_negative = torch.sum((1.-pred)*(1.-target), 0)
    cardinality_negative = torch.sum(2-pred-target, 0)
    dice_negative = (intersection_negative+1e-6) / \
        (cardinality_negative+1e-6)
    temp3 = 1.5-dice_positive-dice_negative # Change from 1.5 to 1.0?

    DICELoss = torch.mean(temp3) # mean over affrdnce types

    aff_loss = 0.5*CELoss + 0.5*DICELoss

    return aff_loss

def _width_loss_fn(pred, target):
    return F.mse_loss(40 * pred, 40 * target, reduction="none")

def _occ_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)


def create_trainer(net, optimizer, scheduler, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()
        # forward
        x, y, pos, pos_occ = prepare_batch(batch, device)
        y_pred = select(net(x, pos, p_tsdf=pos_occ))
        loss, loss_dict = loss_fn(y_pred, y)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss_dict

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, pos, pos_occ = prepare_batch(batch, device)
            y_pred = select(net(x, pos, p_tsdf=pos_occ))
            loss, loss_dict = loss_fn(y_pred, y)
        return x, y_pred, y, loss_dict

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="giga_classic_hr_affnet")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset_raw", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--log_wandb", type=bool, default='')
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--epoch_length_frac", type=float, default=1.0, help="fraction of training data that constitutes one epoch")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.05, help="fraction of data used for validation")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--load_path", type=str, default='', help="load checkpoint network and continue")
    args, _ = parser.parse_known_args()
    print(args)
    main(args)
