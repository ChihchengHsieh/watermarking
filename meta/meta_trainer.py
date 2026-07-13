import time

import torch

import wandb
import copy
from evaluation import evaluation, meta_testing
from .meta_model import ProtoClassifier, ResModel
from .util import (
    TIMING_TABLE,
    BaseTrainerConfig,
    LR_Scheduler,
    MetricMeter,
    SLATrainerConfig,
)


class BaseDATrainer:
    def __init__(self, loaders, args, backbone="resnet34"):
        self.model = ResModel(backbone, output_dim=args.dataset["num_classes"]).cuda()
        self.params = self.model.get_params(args.lr)
        self.optimizer = torch.optim.SGD(
            self.params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
        self.lr_scheduler = LR_Scheduler(self.optimizer, args.num_iters)

        # `self.iter_loaders` is used to load the training data. However, during evaluation or testing,
        # we need to pass a specific data loader that is not available in an iterator.
        self.loaders = loaders
        self.iter_loaders = iter(loaders)

        # recording
        self.meter = MetricMeter()
        self.inner_meter = MetricMeter()

        # required arguments for DATrainer
        self.args = args
        self.backbone = backbone
        self.config = BaseTrainerConfig.from_args(args)
        self.momentum = args.momentum
        self.weight_decay=args.weight_decay
        self.tasks = args.tasks

    def get_source_loss(self, step, *data):
        return self.model.base_loss(*data)

    def get_target_loss(self, step, *data):
        return self.model.base_loss(*data)

    def logging(self, step, info, unit="min"):
        wandb.log(
            {
                **info,
                "iteration": step,
                f"running time ({unit})": (time.perf_counter() - self.meter.start_time)
                * TIMING_TABLE[unit],
            }
        )

    def meta_test(self):
        val_acc = evaluation(self.loaders.loaders["target_validation"], self.model)
        t_acc = meta_testing(self.loaders.loaders["target_validation"],self.loaders.loaders["target_unlabeled_test"], self.model, self.backbone, self.args.dataset["num_classes"])
        if val_acc >= self.meter.best_val_acc:
            self.meter.best_val_acc = val_acc
            self.meter.counter = 0
            self.meter.best_acc = t_acc
        else:
            self.meter.counter += 1
        return val_acc, t_acc
    
    def meta_evaluate(self):
        val_acc = evaluation(self.loaders.loaders["target_validation"], self.new_model)
        t_acc = evaluation(self.loaders.loaders["target_unlabeled_test"], self.new_model)
        if val_acc >= self.inner_meter.best_val_acc:
            self.inner_meter.best_val_acc = val_acc
            self.inner_meter.counter = 0
            self.inner_meter.best_acc = t_acc
        else:
            self.inner_meter.counter += 1
        return val_acc, t_acc
    
    def meta_source_loss(self, step, *data):
        return self.new_model.base_loss(*data)
    
    def meta_target_loss(self, step, *data):
        return self.new_model.base_loss(*data)
    
    def meta_unl_target_loss(self, step, task, *data):
        return 0#"""
    
    def inner_training_step(self,step, task, sx, sy, tx, ty, ux):
        
        if task == "UDA":
            s_loss = self.meta_source_loss(step, sx, sy)
            t_loss = 0
            unl_loss = self.meta_unl_target_loss(step, task, ux, sx, sy)
        elif task == "SSL":
            s_loss = 0
            t_loss = self.meta_target_loss(step, tx, ty)
            unl_loss = self.meta_unl_target_loss(step, task, ux, tx, ty)
        else:
            s_loss = self.meta_source_loss(step, sx, sy)
            t_loss = self.meta_target_loss(step, tx, ty)
            unl_loss = self.meta_unl_target_loss(step, task, ux, tx, ty)#"""

        loss = s_loss + t_loss + unl_loss
        loss.backward(create_graph=True, retain_graph=True)

        return s_loss, unl_loss

    def outer_training_step(self, step, task, sx, sy, tx, ty, ux):
        
        if task == "UDA":
            outer_s_loss = self.meta_source_loss(step, sx, sy)
            outer_t_loss = 0
            outer_unl_loss = self.meta_unl_target_loss(step, task, ux, sx, sy)
        elif task == "SSL":
            outer_s_loss = 0
            outer_t_loss = self.meta_target_loss(step, tx, ty)
            outer_unl_loss = self.meta_unl_target_loss(step, task, ux, tx, ty)
        else:
            outer_s_loss = self.meta_source_loss(step, sx, sy)
            outer_t_loss = self.meta_target_loss(step, tx, ty)
            outer_unl_loss = self.meta_unl_target_loss(step, task, ux, tx, ty)#"""

        outer_loss = outer_s_loss + outer_t_loss + outer_unl_loss
        return outer_loss
    
    def meta_training_step(self, step, inner_step, first_order, task, *data):
        sx, sy, tx, ty, ux = data
        ### Prepare support and query data ###
        if task == "UDA":
            ###     UDA     ###
            im_data_sx_support, im_data_sx_query = sx.split(sx.size(0)//2, dim=0)
            gt_labels_sy_support, gt_labels_sy_query = sy.split(sy.size(0)//2, dim=0)
            im_data_tx_support, im_data_tx_query = 0,0
            gt_labels_ty_support, gt_labels_ty_query = 0,0
            im_data_ux_support, im_data_ux_query = ux[0].split(ux[0].size(0)//2, dim=0)

        if task == "SSL":          
            ###     SSL     ###
            im_data_sx_support, im_data_sx_query = 0,0
            gt_labels_sy_support, gt_labels_sy_query  = 0,0
            im_data_tx_support, im_data_tx_query = tx.split(tx.size(0)//2, dim=0)
            gt_labels_ty_support, gt_labels_ty_query = ty.split(ty.size(0)//2, dim=0)
            im_data_ux_support, im_data_ux_query = ux[0].split(ux[0].size(0)//2, dim=0)

        if task == "SL_Ds":
            ###     Supervised Learning (with Ds)   ###
            im_data_sx_support, im_data_sx_query = sx.split(sx.size(0)//2, dim=0)
            gt_labels_sy_support, gt_labels_sy_query = sy.split(sy.size(0)//2, dim=0)
            im_data_tx_support, im_data_tx_query = 0,0
            gt_labels_ty_support, gt_labels_ty_query = 0,0
            im_data_ux_support, im_data_ux_query = 0,0
            
        if task == "SL_Dt":
            ###     Supervised Learning (with Dt)   ###
            im_data_sx_support, im_data_sx_query = 0,0
            gt_labels_sy_support, gt_labels_sy_query  = 0,0
            im_data_tx_support, im_data_tx_query = tx.split(tx.size(0)//2, dim=0)
            gt_labels_ty_support, gt_labels_ty_query = ty.split(ty.size(0)//2, dim=0)
            im_data_ux_support, im_data_ux_query = 0,0#"""

        if task == "cdac":
            ### CDAC ###
            im_data_sx_support, im_data_sx_query = sx.split(sx.size(0)//2, dim=0)
            gt_labels_sy_support, gt_labels_sy_query = sy.split(sy.size(0)//2, dim=0)
            im_data_tx_support, im_data_tx_query = tx.split(tx.size(0)//2, dim=0)
            gt_labels_ty_support, gt_labels_ty_query = ty.split(ty.size(0)//2, dim=0)
            ### Because for cdac ux is a vector of 3 images ###
            im_data_ux_support = []
            im_data_ux_query = []
            for i in range(len(ux)):
                #print(i)
                support, query = ux[i].split(ux[i].size(0)//2, dim=0)
                im_data_ux_support.append(support)
                im_data_ux_query.append(query)

        ### inner loop ###
        for _ in range(inner_step):
            inner_lr= self.optimizer.param_groups[0]['lr']
            lbl_loss, unl_loss = self.inner_training_step(step, task, im_data_sx_support, gt_labels_sy_support, im_data_tx_support, gt_labels_ty_support, im_data_ux_support)
            
            #assert self.optimizer.param_groups[0]['params'][0].grad is not 
            for name, param in self.new_model.named_params():
                grad = param.grad
                if grad is None:
                    continue
                if first_order==True:
                    grad = grad.detach().data
                self.new_model.set_param(self.new_model, name, param - inner_lr * grad)

            
        ### Outer loop validation ###
        outer_loss = self.outer_training_step(step,task,im_data_sx_query,gt_labels_sy_query,im_data_tx_query,gt_labels_ty_query,im_data_ux_query)
        return lbl_loss, unl_loss, outer_loss
    
    def meta_train(self):
        self.model.train()
        self.meter.start_time = time.perf_counter()
        inner_step = 1
        first_order = True
        for step in range(0, self.config.num_iters + 1):
            (sx, sy), (tx, ty), ux = next(self.iter_loaders)

            task_losses = []
            meta_loss = 0


            tasks =["cdac"]
            if step % 50 == 0:
                tasks = ["UDA"]
            if step % 50 == 25:
                tasks = ["SSL"]
            for task in tasks:
                self.new_model = ResModel(self.backbone, output_dim=self.args.dataset["num_classes"]).cuda()
                self.new_model.copy(self.model, same_var=True)
                self.new_model.train()
                lbl_loss, unl_loss, outer_loss = self.meta_training_step(step, inner_step, first_order, task, sx, sy, tx, ty, ux )
                
                meta_loss+=outer_loss

                # logging
                if step % self.config.log_interval == 0:
                    self.logging(   
                        step,
                        {
                            "LR": self.lr_scheduler.get_lr(),
                            "labeled loss": lbl_loss,
                            "unlabeled loss": unl_loss,
                        },
                    )
                wandb.run.summary["inner_best_test_accuracy"] = self.meter.best_acc

            (meta_loss/len(tasks)).backward(create_graph=False, retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # early-stopping & evaluation
            self.lr_scheduler.step()
            if step >= self.config.early and step % self.config.eval_interval == 0:
                eval_acc, t_acc = self.meta_test()
                self.logging(
                    step,
                    {
                        "evaluation accuracy": eval_acc,
                        "test accuracy": t_acc,
                        "outer loss": meta_loss/len(self.tasks),
                    },
                )
                wandb.run.summary["best_test_accuracy"] = self.meter.best_acc

            # early-stopping
            # Here we set a huge number to plot the whole testing procedure.
            # Change it to a reasonable value for early-stopping
            if self.meter.counter > 10000 or step == self.config.num_iters:
                break

            del self.new_model



class UnlabeledDATrainer(BaseDATrainer):
    def __init__(self, loaders, args, backbone="resnet34", unlabeled_method="mme"):
        super().__init__(loaders, args, backbone)
        self.unlabeled_method = unlabeled_method

    def unlabeled_training_step(self, step, ux):
        self.optimizer.zero_grad()
        unlabeled_loss_fn = getattr(self.model, f"{self.unlabeled_method}_loss")
        u_loss = unlabeled_loss_fn(step, *ux)
        u_loss.backward()
        self.optimizer.step()

        return u_loss.item()
    
    def meta_unl_target_loss(self, step, task, ux, tx, ty):
        if (task == "UDA") or (task =="SSL"):
            unlabeled_loss_fn =getattr(self.new_model, "mixup_loss")
            u_loss = unlabeled_loss_fn(step, ux, tx, ty)
        else:
            unlabeled_loss_fn = getattr(self.new_model, f"{task}_loss")
            u_loss = unlabeled_loss_fn(step, *ux)
        return u_loss

    def meta_training_step(self, step, *data):
        s_loss, t_loss, u_loss = super().meta_training_step(step, *data)
        return s_loss, t_loss, u_loss


def get_SLA_trainer(base_class):
    class SLADATrainer(base_class):
        def __init__(self, loaders, args, **kwargs):
            super().__init__(loaders, args, **kwargs)
            self.config = SLATrainerConfig.from_args(args)
            self.ppc = ProtoClassifier(args.dataset["num_classes"])

        def meta_source_loss(self, step, sx, sy):
            sf = self.new_model.get_features(sx)
            if step > self.config.warmup:
                sy2 = self.ppc(sf.detach(), self.config.T)
                s_loss = self.new_model.sla_loss(sf, sy, sy2, self.config.alpha)
            else:
                s_loss = self.new_model.feature_base_loss(sf, sy)
            return s_loss

        def ppc_update(self, step):
            if step == self.config.warmup:
                self.ppc.init(self.new_model, self.loaders.loaders["target_unlabeled_test"])
                self.lr_scheduler.refresh()

            if step > self.config.warmup and step % self.config.update_interval == 0:
                self.ppc.init(self.new_model, self.loaders.loaders["target_unlabeled_test"])

        def meta_training_step(self, step, *data):
            s_loss, t_loss, u_loss = super().meta_training_step(step, *data)
            self.ppc_update(step)

            return s_loss, t_loss, u_loss

    return SLADATrainer


def get_trainer(base_class, label_trick=None):
    match label_trick:
        case "SLA", *_:
            return get_SLA_trainer(base_class)
        case _:
            return base_class
