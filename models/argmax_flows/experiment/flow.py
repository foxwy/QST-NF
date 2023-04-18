# Path
import os
import sys
import time

import torch
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from survae.distributions import DataParallelDistribution
from survae.utils import elbo_nats
from .utils import get_args_table, clean_dict

# Experiment
from .base import BaseExperiment

# Logging frameworks
from torch.utils.tensorboard import SummaryWriter
import wandb


def add_exp_args(parser):

    # Train params
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
    parser.add_argument('--resume', type=str, default=None)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=400)
    parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--log_tb', type=eval, default=False)
    parser.add_argument('--log_wandb', type=eval, default=False)


class FlowExperiment(BaseExperiment):

    log_base = './log'
    no_log_keys = ['project', 'name',
                   'log_tb', 'log_wandb',
                   'check_every', 'eval_every',
                   'device', 'parallel']

    def __init__(self, args,
                 model_id, optim_id,
                 train_loader, eval_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch, f):

        # Edit args
        if args.eval_every is None:
            args.eval_every = args.epochs
        if args.check_every is None:
            args.check_every = args.epochs
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join(model_id)

        # Move model
        model = model.to(args.device)
        if args.parallel == 'dp':
            model = DataParallelDistribution(model)

        # Init parent
        log_path = os.path.join(self.log_base, model_id, args.name)
        super(FlowExperiment, self).__init__(model=model,
                                             optimizer=optimizer,
                                             scheduler_iter=scheduler_iter,
                                             scheduler_epoch=scheduler_epoch,
                                             log_path=log_path,
                                             eval_every=args.eval_every,
                                             check_every=args.check_every)

        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.f = f

        # Init logging
        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        if args.log_tb:
            self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
            self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)
        if args.log_wandb:
            wandb.init(config=args_dict, project=args.project, id=args.name, dir=self.log_path)

    def log_fn(self, epoch, train_dict, eval_dict):

        # Tensorboard
        if self.args.log_tb:
            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)

        # Weights & Biases
        if self.args.log_wandb:
            for metric_name, metric_value in train_dict.items():
                wandb.log({'base/{}'.format(metric_name): metric_value}, step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    wandb.log({'eval/{}'.format(metric_name): metric_value}, step=epoch+1)

    def resume(self):
        resume_path = os.path.join(self.log_base, self.data_id, self.model_id, self.optim_id, self.args.resume, 'check')
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]
            if epoch in self.eval_epochs:
                eval_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    eval_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else: eval_dict = None
            self.log_fn(epoch, train_dict=train_dict, eval_dict=eval_dict)

    def run(self):
        if self.args.resume: self.resume()
        super(FlowExperiment, self).run(epochs=self.args.epochs)

    def train_fn(self, epoch):
        self.model.train()
        self.model.to("cuda:0")
        loss_sum = 0.0
        loss_count = 0
        for x in self.train_loader:
            self.optimizer.zero_grad()
            loss = elbo_nats(self.model, x.to(self.args.device))
            loss.backward()
            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, NLL: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'nll': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x in self.eval_loader:
                loss = elbo_nats(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, NLL: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')

            times = []
            for k in range(10):
                torch.cuda.synchronize()
                t0 = time.time()
                self.model.sample(1)
                torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1 - t0)
            print('Times mean: {:.6f} std: {:.6f}\n'.format(np.mean(times), np.std(times)))

            self.f.write('Times mean: {:.6f} std: {:.6f}\n'.format(np.mean(times), np.std(times)))
            self.f.write('nll test: {:.6f}\n\n'.format(loss_sum/loss_count))
            self.f.flush()

        return {'nll': loss_sum/loss_count}

    def sample_fn_mp(self, group_N):
        samples = self.model.sample(group_N)
        logp = self.model.log_prob(samples)
        samples = samples.detach().cpu().to(torch.int).numpy().reshape(len(samples), -1)
        P = logp.exp().detach().cpu().numpy()

        return [samples, P]

    def sample_fn(self, epoch):
        if (epoch + 1) % 100 == 0:
            self.model.eval()
            with torch.no_grad():
                group_N = 2 * 10**3
                params = [group_N] * (10 // group_N)

                cpu_counts = mp.cpu_count()
                if len(params) < cpu_counts:
                    cpu_counts = len(params)

                time_b = time.perf_counter()  # sample time
                print('---begin sample---')
                with Pool(cpu_counts) as pool:
                    results = pool.map(self.sample_fn_mp, params)

                samples, P = results[0][0], results[0][1]
                for k in range(len(results) - 1):
                    sa, lP = results[k+1][0], results[k+1][1]
                    samples = np.vstack((samples, sa))
                    P = np.hstack((P, lP))

                print('sample time:', time.perf_counter() - time_b)

                #samples, s_idx = np.unique(samples, axis=0, return_index=True)
                #P = P[s_idx]
                #samples, P = array_posibility_unique(samples)
                #cF = Ficalc.cFidelity(samples, P, 10**3)
                #print('cfdelity:', cF, abs(cF - 1))
