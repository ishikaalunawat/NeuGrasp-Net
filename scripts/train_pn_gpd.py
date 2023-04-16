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

from vgn.dataset_voxel_grasp_pc import DatasetVoxelGraspPCOcc
from vgn.networks import get_network, load_network

LOSS_KEYS = ['loss_all', 'loss_qual'] # Removed OCC

def main(args):

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = "cpu"
    kwargs = {"num_workers": args.num_workers, "pin_memory": True} if use_cuda else {}

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
        meshdir = logdir / "meshes"
        meshdir.mkdir(parents=True, exist_ok=True)
    else:
        logdir = Path(args.savedir)
    
    filename = 'summary_%s.txt' % (args.dataset_raw.name)
    with open(filename, 'r') as f:
        note = (";").join(f.readlines()[1:5]).replace('\n', '')

    if args.log_wandb:
        wandb.init(config=args, project="6dgrasp", entity="irosa-ias", id=args.net+'_'+args.dataset.name+'_'+time_stamp, notes=note)

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
        "accuracy": Accuracy(lambda out: (torch.round(out[1]), out[2][0])), # out[1] => y_pred -> quality PNGPD
                                                                              # out[2][0] => y -> quality PNGPD
                                                                              # ^ Refer def _update() returns
        "precision": Precision(lambda out: (torch.round(out[1]), out[2][0])),
        "recall": Recall(lambda out: (torch.round(out[1]), out[2][0])),
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
        "neural_grasp",
        n_saved=None, # Changed from 1. Save everything
        require_empty=True,
    )
    best_checkpoint_handler = ModelCheckpoint(
        logdir,
        "best_neural_grasp",
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

    dataset = DatasetVoxelGraspPCOcc(root, root_raw)

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
    # pc==tsdf
    pc, (label, width), (pos, rotations, grasps_pc_local, grasps_pc), pos_occ, occ_value = batch

    pc = pc.float().to(device)
    label = label.float().to(device)
    width = width.float().to(device)
    pos = pos.float().to(device)
    rotations = rotations.float().to(device)
    grasps_pc_local = grasps_pc_local.float().to(device)
    grasps_pc = grasps_pc.float().to(device)
    pos_occ = pos_occ.float().to(device)
    occ_value = occ_value.float().to(device)

    return pc, (label, width, occ_value), (pos, rotations, grasps_pc_local, grasps_pc), pos_occ


def select(out):
    #qual_out, rot_out, width_out, sdf = out
    qual_out = out     # <- Changed to predict only grasp quality (check inside)
    # rot_out = rot_out.squeeze(1)
    # occ = torch.sigmoid(sdf) # to probability
    #return qual_out.squeeze(-1), rot_out, width_out.squeeze(-1), occ
    return qual_out.squeeze()


def loss_fn(y_pred, y):
    label_pred = y_pred # PointNetGPD output
    label, width, occ_value = y
    loss_qual = _qual_loss_fn(label_pred, label)
    # loss_rot = _rot_loss_fn(rotation_pred, rotations)
    # loss_width = _width_loss_fn(width_pred, width)
    # loss_occ = _occ_loss_fn(occ_pred, occ)
    loss = loss_qual #+ label * (0.01 * loss_width) + loss_occ # <-label * (loss_rot + 0.01 * loss_width): new one, Changed
    loss_dict = {'loss_qual': loss_qual,
                #  'loss_rot': loss_rot.mean(),
                # 'loss_width': loss_width.mean(),
                # 'loss_occ': loss_occ.mean(),
                'loss_all': loss.mean()
                }
    return loss, loss_dict


def _qual_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="mean")


# def _rot_loss_fn(pred, target):
#     loss0 = _quat_loss_fn(pred, target[:, 0])
#     loss1 = _quat_loss_fn(pred, target[:, 1])
#     return torch.min(loss0, loss1)


# def _quat_loss_fn(pred, target):
#     return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


def _width_loss_fn(pred, target):
    return F.mse_loss(40 * pred, 40 * target, reduction="none")

def _occ_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)

# def get_mesh(voxels):
#     return mcubes.marching_cubes(voxels, 0)

def create_trainer(net, optimizer, scheduler, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()
        # forward
        x, y, grasp_query, _ = prepare_batch(batch, device)

        # forward() [PointNetGPD()] takes only grasp_query: (pos, rot, grasp_local_pc, <grasp_pc>s)
        y_pred = select(net(grasp_query)) 
        if y_pred.nelement ==0:
            print(y_pred)
        loss, loss_dict = loss_fn(y_pred, y)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss_dict # (32,40,40); pred [(label, width, occ)], true [(label, width, occ)], loss_dict (already defined above)

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):

        net.eval()
        with torch.no_grad():
            # prepare batch
            x, y, grasp_query, _ = prepare_batch(batch, device)
            # forward() [PointNetGPD()] takes only grasp_query: (pos, rot, grasp_local_pc, <grasp_pc>s)
            y_pred = select(net(grasp_query)) 
            # compute loss
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
    parser.add_argument("--net", default="pointnetgpd")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset_raw", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch_length_frac", type=float, default=1.0, help="fraction of training data that constitutes one epoch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_split", type=float, default=0.05, help="fraction of data used for validation")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--load_path", type=str, default='', help="load checkpoint network and continue")
    args, _ = parser.parse_known_args()
    print(args)
    main(args)
