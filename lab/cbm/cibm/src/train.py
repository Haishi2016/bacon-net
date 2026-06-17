import datetime
import os
import random
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from cycler import cycler
from fire import Fire
from loguru import logger
from lovely_numpy import lo
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm, trange

import wandb
from dataset.CUB import CUBDataModule
from dataset.emb import EmbeddingsDataModule
from losses import est_MI, est_MI_cy, est_MI_binning, get_H_discretization, est_HC
from models import StochasticMLP, BasicMLP

wb_run = None
label_metric = accuracy_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def measure_intervention_for_groups(model, test_loader, DEVICE, is_random, concept_groups,
                                    num_groups_to_tti):
    preds = []
    gt = []
    for batch in test_loader:
        feats = batch[0].to(DEVICE)
        gt_concepts = batch[1].to(DEVICE)
        targets = batch[2].reshape(-1).to(DEVICE)
        if is_random:
            intervention_groups_idx = np.random.choice(range(len(concept_groups)),
                                                       size=num_groups_to_tti, replace=False)
        else:
            uncertainties = model.uncertainties(feats)
            uncertainties = [torch.mean(uncertainties[:, concept_groups[idx]])
                             for idx in range(len(concept_groups))]
            # the groups with lowest -log(p(0)) => smallest distance to 0 => higher uncertainty
            intervention_groups_idx = sorted(range(len(concept_groups)),
                                             key=lambda x: -uncertainties[x])[:num_groups_to_tti]
        allowed_idx = [concept_idx for idx in intervention_groups_idx
                       for concept_idx in concept_groups[idx]]
        allowed_idx = torch.tensor(allowed_idx).to(DEVICE)
        with torch.no_grad():
            logits = model.forward_tti(feats, gt_concepts, allowed_idx)
        preds.append(logits.cpu().argmax(dim=-1).numpy())
        gt.append(targets.cpu().numpy())
    return preds, gt


def measure_interventions(model, test_loader, DEVICE, num_trials=5, is_random=True):
    concept_groups = list(test_loader.dataset.concept_groups.values())
    model.eval()
    num_tti_groups_to_acc = dict()
    for num_groups_to_tti in trange(1, len(concept_groups) + 1):
        accuracies = []
        for _ in tqdm(range(num_trials), leave=False, desc='Running trials over test set'):
            preds, gt = measure_intervention_for_groups(model, test_loader, DEVICE, is_random,
                                                        concept_groups, num_groups_to_tti)
            test_acc = label_metric(np.concatenate(gt).reshape(-1),
                                    np.concatenate(preds).reshape(-1))
            accuracies.append(test_acc)
        num_tti_groups_to_acc[num_groups_to_tti] = accuracies
    return num_tti_groups_to_acc


def run_test(model, test_loader, num_concepts, DEVICE, blackbox, verbose, noise_level=0):
    test_pbar = enumerate(test_loader)
    if verbose == 2:
        test_pbar = tqdm(test_pbar, leave=False, desc='Testing epoch',
                         total=len(test_loader))
    preds = []
    gt = []
    model.eval()
    preds_concepts = torch.zeros((num_concepts,))
    for i, batch in test_pbar:
        feats = batch[0]
        if noise_level != 0:
            # Normalize the noise magnitude based on the input scale
            noise = torch.randn_like(feats) * (noise_level * feats.std())
            feats = feats + noise
        feats = feats.to(DEVICE)
        gt_concepts = batch[1].to(DEVICE)
        targets = batch[2].reshape(-1).to(DEVICE)
        with torch.no_grad():
            logits, max_prob, concept_preds = model(feats)
        preds.append(logits.cpu().argmax(dim=-1).numpy())
        gt.append(targets.cpu().numpy())

        if not blackbox:
            preds_concepts += (
                    (concept_preds > 0).int() == gt_concepts.to(DEVICE)
            ).detach().cpu().sum(dim=0)

    test_acc = label_metric(np.concatenate(gt).reshape(-1),
                            np.concatenate(preds).reshape(-1))
    return test_acc, preds_concepts / len(test_loader.dataset)


def train_model_per_class_bacon(dataloaders, model, optim, scheduler, verbose=0, epochs=20,
                                 logdir=None, anneal_epochs=None):
    """
    Per-class training for multi-tree BACON CBM.
    
    Key: Pre-load data per class in batches to avoid slow disk I/O.
    Only evaluates 2 trees per sample (positive + 1 random negative).
    """
    global wb_run
    DEVICE = next(model.parameters()).device
    train_dataloader = dataloaders['train']
    
    num_classes = model.num_classes
    train_dataset = train_dataloader.dataset
    batch_size = train_dataloader.batch_size
    
    # Pre-load: group all training data by class (do once)
    print("Pre-loading training data by class...")
    class_data = {c: [] for c in range(num_classes)}
    for idx in range(len(train_dataset)):
        feat, concept, label = train_dataset[idx]
        label = int(label)
        if label < num_classes:
            class_data[label].append((feat, concept, label))
    
    pbar = range(epochs) if verbose == 0 else tqdm(range(epochs), desc='Epoch')
    
    for epoch in pbar:
        # Anneal routing temperature if applicable
        if anneal_epochs and hasattr(model, 'anneal'):
            model.anneal(min(1.0, epoch / max(1, anneal_epochs)))
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        # For each class, train on batches of that class
        for class_idx in range(num_classes):
            data_list = class_data.get(class_idx, [])
            if not data_list:
                continue  # Skip classes with no training samples
            
            # Batch the pre-loaded data for this class
            for batch_start in range(0, len(data_list), batch_size):
                batch_data = data_list[batch_start:batch_start + batch_size]
                
                # Stack batch
                feats = torch.stack([d[0] for d in batch_data]).to(DEVICE)
                gt_concepts = torch.stack([d[1] for d in batch_data]).to(DEVICE)
                targets = torch.tensor([d[2] for d in batch_data], device=DEVICE, dtype=torch.long)
                
                model.train()
                
                # Forward: negative sampling uses 2 trees per sample
                logits, _, _ = model(feats, targets)
                
                # Loss
                loss = F.cross_entropy(logits, targets)
                
                # Backward
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_loss += loss.item()
                epoch_steps += 1
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        if verbose > 0 and epoch % max(1, epochs // 10) == 0:
            avg_loss = epoch_loss / max(1, epoch_steps)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")
            
            if wb_run is not None:
                wb_run.log({"epoch": epoch, "loss": avg_loss})
    
    # NO EVALUATION - The point is to never evaluate 200 trees!
    if verbose > 0:
        print("Training complete.")
    
    return 0.0


def train_model(dataloaders, model, optim, scheduler, verbose=0, MI_const=1, jensen=False,
                blackbox=False, sz=1000, beta=0.5, beta_lr=-1e-2, epochs=20,
                num_bins_mi=2000, logdir=None, collect_MIs=True, merge_train_val=False,
                use_HC=False, use_kl_ce_objective=False, anneal_epochs=None):
    global wb_run
    DEVICE = next(model.parameters()).device
    train_dataloader, val_dataloader, test_dataloader = dataloaders['train'], \
        None, \
        dataloaders['test']

    if not merge_train_val:
        val_dataloader = dataloaders['val']

    train_losses = []
    val_losses = []
    train_MI = []
    val_MI = []

    label_acc = []
    best_label_acc = -1

    MI_XC = []
    MI_CY = []
    MI_ZY = []
    MI_XZ = []
    MI_ZC = []
    pbar = range(epochs) if verbose == 0 else tqdm(range(epochs), desc='Progress')
    for epoch in pbar:

        # BACON routing schedule: anneal Sinkhorn temperature broad -> sharp over
        # the first ``anneal_epochs`` epochs, then hold in the committed regime.
        if anneal_epochs and hasattr(model, 'anneal'):
            model.anneal(min(1.0, epoch / max(1, anneal_epochs)))

        train_loss_labels = 0
        train_loss_concepts = 0
        MI_constraint_loss = 0

        train_pbar = enumerate(dataloaders['train'])
        if verbose == 2:
            train_pbar = tqdm(train_pbar, leave=False, desc='Train epoch',
                              total=len(dataloaders['train']))
        preds = []
        gt = []
        all_c_activations = []

        preds_concepts = torch.zeros((train_dataloader.dataset.num_concepts,))
        for i, batch in train_pbar:
            model.train()
            feats = batch[0].to(DEVICE)
            gt_concepts = batch[1].to(DEVICE)
            targets = batch[2].reshape(-1).to(DEVICE)
            logits, max_prob, concept_preds = model(feats, targets)
            
            # L = (1-beta) * KL[p(c|z) || q(c|z)] + H(p(y|c), q(y|c))
            if use_kl_ce_objective and not blackbox:
                batch_concepts_loss = F.binary_cross_entropy_with_logits(
                    concept_preds.float(),
                    gt_concepts.float().to(DEVICE),
                    reduction='none'
                ).sum(dim=1).mean()
                concept_loss = (1 - beta) * batch_concepts_loss
                label_loss = F.cross_entropy(logits, targets)
                
                loss = concept_loss + label_loss
                train_loss_labels += label_loss.item()
                train_loss_concepts += batch_concepts_loss.item()
                
                preds.append(logits.cpu().argmax(dim=-1).numpy())
                gt.append(targets.cpu().numpy())
                all_c_activations.append(concept_preds.detach().cpu())
                
                preds_concepts += ((concept_preds > 0).int() == gt_concepts.to(DEVICE)
                                   ).detach().cpu().sum(dim=0)
            else:
                # L = CE(y) + BCE(c) + beta * (MI_const - I(X;C))
                loss = F.cross_entropy(logits, targets)
                train_loss_labels += loss.item()
                preds.append(logits.cpu().argmax(dim=-1).numpy())
                gt.append(targets.cpu().numpy())
                all_c_activations.append(concept_preds.detach().cpu())

                if collect_MIs:
                    mi_xz_, mi_zy_, mi_zc_ = est_MI_binning(model, train_dataloader.dataset,
                                                            num_bins=num_bins_mi,
                                                            batch_size=train_dataloader.batch_size)
                    MI_XZ.append(mi_xz_)
                    MI_ZY.append(mi_zy_)
                    MI_ZC.append(mi_zc_)
                    MI_CY.append(est_MI_cy(model, train_dataloader.dataset, samples=sz).item())

                if not blackbox:
                    batch_concepts_loss = F.binary_cross_entropy_with_logits(
                        concept_preds.float(),
                        gt_concepts.float().to(DEVICE),
                        reduction='none'
                    ).sum(dim=1).mean()
                    if use_HC:
                        loss += (1 + beta) * batch_concepts_loss
                    else:
                        loss += batch_concepts_loss
                    train_loss_concepts += batch_concepts_loss.item()
                    preds_concepts += ((concept_preds > 0).int() == gt_concepts.to(DEVICE)
                                       ).detach().cpu().sum(dim=0)

                if max_prob is not None:  # stochastic model
                    mi_loss = None
                    if use_HC:
                        H_C = est_HC(model, train_dataloader.dataset,
                                     sz=min(sz, len(train_dataloader.dataset)), jensen=jensen)
                        mi_loss = (1 - beta) * H_C
                        constraint_item = 0
                    else:
                        MI = est_MI(model, train_dataloader.dataset,
                                    sz=min(sz, len(train_dataloader.dataset)), jensen=jensen)
                        constraint = (MI_const - MI)
                        mi_loss = beta * constraint
                        constraint_item = constraint.item()
                        if collect_MIs:
                            MI_XC.append(max(0, MI.item()))
                else:
                    constraint_item = 0
                
                if not use_kl_ce_objective:
                    beta += beta_lr * constraint_item  # Lagrangian dual update
                    MI_constraint_loss += constraint_item

                if use_HC and not use_kl_ce_objective:
                    # H(C) gradient only wrt encoder params, not classifier
                    mi_loss.backward(retain_graph=True, inputs=list(model.pred_mu.parameters()) + list(model.pred_sigma.parameters()))
                elif max_prob is not None and not use_kl_ce_objective:
                    loss += mi_loss

            loss.backward()
            optim.step()
            optim.zero_grad()

        all_samples = torch.cat(all_c_activations, dim=0)
        H_C_train = get_H_discretization(all_samples, num_bins_mi)
        concepts_acc_this_epoch = (preds_concepts / len(train_dataloader.dataset)).mean()

        if scheduler is not None:
            scheduler.step()
        val_loss_labels = 0
        val_loss_concepts = 0
        val_MI_constraint_loss = 0
        if not merge_train_val:
            preds = []
            gt = []
            all_c_activations = []

            preds_concepts = torch.zeros((train_dataloader.dataset.num_concepts,))
            val_pbar = enumerate(dataloaders['val'])
            if verbose == 2:
                val_pbar = tqdm(val_pbar, leave=False, desc='Validation epoch',
                                total=len(dataloaders['val']))
            model.eval()
            for i, batch in val_pbar:
                feats = batch[0].to(DEVICE)
                gt_concepts = batch[1].to(DEVICE)
                targets = batch[2].reshape(-1).to(DEVICE)
                with torch.no_grad():
                    logits, max_prob, concept_preds = model(feats)
                preds.append(logits.cpu().argmax(dim=-1).numpy())
                gt.append(targets.cpu().numpy())
                all_c_activations.append(concept_preds)

                loss = F.cross_entropy(logits, targets)
                val_loss_labels += loss.item()

                if not blackbox:
                    batch_concepts_loss = F.binary_cross_entropy_with_logits(
                        concept_preds.float(),
                        gt_concepts.float().to(DEVICE),
                        reduction='none'
                    ).sum(dim=1).mean()
                    loss += batch_concepts_loss
                    val_loss_concepts += batch_concepts_loss.item()
                    preds_concepts += ((concept_preds > 0).int() == gt_concepts.to(DEVICE)
                                       ).detach().cpu().sum(dim=0)
                if max_prob is not None and not use_kl_ce_objective:
                    MI = est_MI(model, val_dataloader.dataset,
                                sz=min(sz, len(val_dataloader.dataset)), jensen=jensen,
                                requires_grad=False)
                    constraint = (MI_const - MI)
                    loss += beta * constraint
                    constraint_item = constraint.item()
                else:
                    constraint_item = 0
                val_MI_constraint_loss += constraint_item

            val_losses.append(
                [val_loss_labels / len(val_dataloader), val_loss_concepts / len(val_dataloader)])
            val_MI.append(val_MI_constraint_loss / len(val_dataloader))
            concepts_acc_this_epoch = (preds_concepts / len(val_dataloader.dataset)).mean()

            all_samples = torch.cat(all_c_activations, dim=0)
            H_C_val = get_H_discretization(all_samples, num_bins_mi)

        train_losses.append([train_loss_labels / len(train_dataloader),
                             train_loss_concepts / len(train_dataloader)])

        train_MI.append(MI_constraint_loss / len(train_dataloader))
        label_acc_this_epoch = label_metric(np.concatenate(gt).reshape(-1),
                                            np.concatenate(preds).reshape(-1))

        log_msg = f'Train losses: ({train_losses[-1][0]:.6}, {train_losses[-1][1]:.6})' + \
                  (
                      f' Val losses: ({val_losses[-1][0]:.6}, {val_losses[-1][1]:.6}) ' if not merge_train_val else ' ') + \
                  f'Concepts: {100 * concepts_acc_this_epoch:.6f} | ' + \
                  f'Labels: {100 * label_acc_this_epoch} | ' + \
                  f'Beta: {beta} ' + \
                  f'H_C_train: {H_C_train} ' + ('' if merge_train_val else f'H_C_val: {H_C_val} ')
        logger.info(log_msg)
        if wb_run is not None:
            wb_run.log(data={
                'train_loss_labels': train_losses[-1][0],
                'train_loss_concepts': train_losses[-1][-1],
                'val_loss_labels': val_losses[-1][0] if not merge_train_val else 0,
                'val_loss_concepts': val_losses[-1][-1] if not merge_train_val else 0,
                'val_acc_concepts': 100 * concepts_acc_this_epoch,
                'val_acc_labels': 100 * label_acc_this_epoch,
                'H_C_train': H_C_train,
                'H_C_val': H_C_val if not merge_train_val else 0,
                'beta': beta
            }, step=epoch)
        if best_label_acc < label_acc_this_epoch:
            torch.save(model.state_dict(), os.path.join(logdir, "model.pth"))
        best_label_acc = max(label_acc_this_epoch, best_label_acc)
        label_acc.append(label_acc_this_epoch)

    test_acc, preds_concepts = run_test(model, dataloaders['test'],
                                        train_dataloader.dataset.num_concepts,
                                        DEVICE, blackbox, verbose, noise_level=0)

    model.load_state_dict(torch.load(os.path.join(logdir, "model.pth")))
    model.eval()

    return np.array(train_losses), np.array(val_losses), train_MI, val_MI, \
        test_acc, preds_concepts, \
        MI_XC, MI_CY, MI_XZ, MI_ZY, MI_ZC


def run_experiment(model_arch, dataset_name, is_blackbox=False, is_stochastic=True, lr=0.001,
                   wd=0.0, epochs=20, num_runs=10, decoy_p=0, num_decoy_concepts=0,
                   num_removed_concepts=0,
                   verbose=1, MI_const=1, activation='relu',
                   train_backbone=False, optimizer='adam',
                   beta=0.5, beta_lr=-1e-2, batch_size=128,
                   samples_mi=200, collect_MIs=True, merge_train_val=False, use_scheduler=True,
                   seed=0, measure_robustness=False, measure_intervention=True, use_HC=False,
                   log_to_wandb=False, use_kl_ce_objective=False, log_base='logs',
                   head_type='mlp', head_lr_mult=1.0, bacon_aggregator='lsp.full_weight',
                   anneal_frac=0.667, bacon_tree_layout='left', bacon_concept_dropout=0.0,
                   bacon_sinkhorn_final_temp=0.1, bacon_use_negative_sampling=False):
    global wb_run
    if log_to_wandb:
        wb_run = wandb.init(project='CBM_IB')
    global label_metric
    set_seed(seed)

    now = datetime.datetime.now()
    logdir = now.strftime("%m%d%H%M%S")
    logpath = os.path.join(log_base, logdir)
    os.makedirs(logpath, exist_ok=True)
    if 'emb' in dataset_name:
        dm = EmbeddingsDataModule(dataset_name.split('_')[0], decoy_p)
        label_metric = partial(f1_score, average='macro')
    else:
        dm = CUBDataModule(embed_image=not train_backbone,
                           merge_train_val=merge_train_val,
                           n_decoys=num_decoy_concepts,
                           n_removed=num_removed_concepts)
        label_metric = accuracy_score
    logger.info(f"Selected target metric: {label_metric}")
    num_concepts, num_classes = dm.train_dataset.num_concepts, dm.train_dataset.num_classes
    dataloaders = {'train': dm.train_dataloader(batch_size=batch_size),
                   'test': dm.test_dataloader(batch_size=batch_size)}
    if not merge_train_val:
        dataloaders['val'] = dm.val_dataloader(batch_size=batch_size)
    test_accuracies = []

    test_label_losses = np.zeros((num_runs, epochs))
    test_concept_losses = np.zeros((num_runs, epochs))

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    robustness_to_noise_labels = {nl: [] for nl in noise_levels}  # noise_level -> [labels acc]
    robustness_to_noise_concepts = {nl: [] for nl in noise_levels}  # noise_level -> [concepts acc]

    concepts_accs = torch.zeros((num_runs, num_concepts))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Selected device: {str(device)}')
    for i in range(num_runs):
        set_seed(seed + i)
        model, optim, scheduler = configure_training(train_backbone, is_stochastic, model_arch, num_concepts,
                       num_classes, activation, optimizer, use_scheduler, lr, wd, device, epochs,
                       head_type=head_type, head_lr_mult=head_lr_mult,
                       bacon_aggregator=bacon_aggregator,
                       bacon_tree_layout=bacon_tree_layout,
                       bacon_concept_dropout=bacon_concept_dropout,
                       bacon_sinkhorn_final_temp=bacon_sinkhorn_final_temp,
                       bacon_use_negative_sampling=bacon_use_negative_sampling)
        anneal_epochs = int(epochs * anneal_frac) if head_type in ('bacon', 'bacon_multi') else None
        
        # Use per-class training for multi-tree BACON
        if head_type == 'bacon_multi':
            test_acc = train_model_per_class_bacon(dataloaders, model, optim, scheduler,
                                                   verbose=verbose, epochs=epochs,
                                                   logdir=logpath, anneal_epochs=anneal_epochs)
            # For bacon_multi, create dummy MI and loss arrays for compatibility
            train_losses = np.zeros((epochs, 2))
            test_losses = np.zeros((epochs, 2))
            train_MI = np.zeros((epochs, 5))
            test_MI = np.zeros((epochs, 5))
            concepts_acc = torch.zeros(num_concepts)
            MI_XC = MI_CY = MI_XZ = MI_ZY = MI_ZC = None
        else:
            train_losses, test_losses, train_MI, test_MI, \
                test_acc, concepts_acc, MI_XC, \
                MI_CY, MI_XZ, MI_ZY, MI_ZC = train_model(dataloaders, model,
                                                         optim,
                                                         scheduler,
                                                         verbose=verbose,
                                                         blackbox=is_blackbox,
                                                         epochs=epochs,
                                                         MI_const=MI_const, logdir=logpath,
                                                         beta=beta, beta_lr=beta_lr, sz=samples_mi,
                                                         merge_train_val=merge_train_val,
                                                         collect_MIs=collect_MIs,
                                                         use_HC=use_HC,
                                                         use_kl_ce_objective=use_kl_ce_objective,
                                                         anneal_epochs=anneal_epochs)

        test_accuracies.append(test_acc)
        concepts_accs[i] = concepts_acc
        if not merge_train_val:
            test_label_losses[i] = test_losses[:, 0]
            test_concept_losses[i] = test_losses[:, 1]

        if measure_robustness:
            logger.info("Running robustness tests (noise injection)")
            for noise_level in noise_levels:
                nl_acc, nl_preds_concepts = run_test(model, dataloaders['test'],
                                                     num_concepts, device,
                                                     is_blackbox,
                                                     verbose, noise_level)
                robustness_to_noise_labels[noise_level].append(nl_acc)
                robustness_to_noise_concepts[noise_level].append(nl_preds_concepts.mean())

    logger.info("Label accuracies")
    logger.info(lo(test_accuracies))
    logger.info("Concept accuracies")
    logger.info(f"{concepts_accs.mean()}±{concepts_accs.mean(1).std()}")

    if measure_intervention:
        num_tti_groups_to_acc = measure_interventions(model, dataloaders['test'], device)

        with open(os.path.join(logpath, 'interventions.txt'), 'w') as f:
            for n_groups, accuracies in num_tti_groups_to_acc.items():
                print(n_groups, accuracies, file=f)
        fig, ax = plt.subplots()
        acc_tti = [(k, np.mean(acc), np.std(acc)) for k, acc in num_tti_groups_to_acc.items()]
        acc_tti.sort()
        acc_tti_means = np.array([m for _, m, _ in acc_tti])
        acc_tti_stds = np.array([std for _, _, std in acc_tti])
        ax.plot(acc_tti_means)
        plt.fill_between(range(len(acc_tti_means)), acc_tti_means - acc_tti_stds,
                         acc_tti_means + acc_tti_stds, alpha=0.5)

        ax.set_xlabel('Number of intervened groups')
        ax.set_ylabel('Target accuracy')
        plt.savefig(os.path.join(logpath, "interventions.pdf"))
        plt.close(fig)
        normalized_auc = 0
        prev_value = test_acc
        for i, m, s in acc_tti:
            normalized_auc += (m - prev_value)
            prev_value = m
        normalized_auc /= len(acc_tti)
        logger.info(f"Interventions AUC = {np.mean([m for i, m, s in acc_tti])}")
        logger.info(f"Normalized Interventions AUC = {normalized_auc}")

    if measure_intervention and is_stochastic:
        run_and_plot_interventions_uncertainty(model, dataloaders, device, logpath, test_acc)

    if measure_robustness:
        with open(os.path.join(logpath, 'robustness.csv'), 'w') as f:
            print('noise_level,label_acc_mean,label_acc_std,concepts_acc_mean,concepts_acc_std',
                  file=f)
            for ((nl, label_acc), (_, concepts_acc)) in zip(robustness_to_noise_labels.items(),
                                                            robustness_to_noise_concepts.items()):
                label_acc = np.array(label_acc)
                concepts_acc = np.array(concepts_acc)
                print(
                    f"{nl},{label_acc.mean()},{label_acc.std()},{concepts_acc.mean()},{concepts_acc.std()}",
                    file=f
                )

    if not merge_train_val:
        plot_losses(logpath, test_label_losses, test_concept_losses)
    if collect_MIs:
        plot_mi(logpath, is_stochastic, MI_XC, MI_CY, MI_XZ, MI_ZY, MI_ZC)

    with open(os.path.join(logpath, 'info.txt'), 'w') as f:
        config = dict(model_arch=model_arch, dataset_name=dataset_name, is_blackbox=is_blackbox,
                      is_stochastic=is_stochastic, lr=lr,
                      wd=wd, epochs=epochs, num_runs=num_runs,
                      num_decoy_concepts=num_decoy_concepts,
                      num_removed_concepts=num_removed_concepts, decoy_p=decoy_p,
                      verbose=verbose, MI_const=MI_const, activation=activation,
                      train_backbone=train_backbone, optimizer=optimizer,
                      beta=beta, beta_lr=beta_lr,
                      batch_size=batch_size,
                      samples_mi=samples_mi, collect_MIs=collect_MIs,
                      merge_train_val=merge_train_val, use_scheduler=use_scheduler,
                      seed=seed, measure_robustness=measure_robustness,
                      measure_intervention=measure_intervention, use_HC=use_HC,
                      use_kl_ce_objective=use_kl_ce_objective,
                      head_type=head_type, head_lr_mult=head_lr_mult,
                      bacon_aggregator=bacon_aggregator, anneal_frac=anneal_frac,
                      bacon_tree_layout=bacon_tree_layout,
                      bacon_concept_dropout=bacon_concept_dropout,
                      bacon_sinkhorn_final_temp=bacon_sinkhorn_final_temp)
        print(config, file=f)
        print("Label accuracies", file=f)
        print(lo(test_accuracies), file=f)
        print("Concept accuracies", file=f)
        print(f"{concepts_accs.mean()}±{concepts_accs.mean(1).std()}", file=f)

    logger.info(f"Saved the plots, config and results to {logpath}")


def configure_training(train_backbone, is_stochastic, model_arch, num_concepts,
                       num_classes, activation, optimizer, use_scheduler, lr, wd, device, epochs,
                       head_type='mlp', head_lr_mult=1.0, bacon_aggregator='lsp.full_weight',
                       bacon_tree_layout='left', bacon_concept_dropout=0.0,
                       bacon_sinkhorn_final_temp=0.1, bacon_use_negative_sampling=False):
    backbone = None
    if train_backbone:
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50',
                                  pretrained=True)
        backbone.fc = torch.nn.Identity()
        backbone.to(device)
    if head_type == 'bacon':
        if is_stochastic:
            from bacon_head import StochasticBaconCBM
            model = StochasticBaconCBM(list(model_arch) + [num_concepts, num_classes],
                             activation, backbone=backbone,
                             aggregator=bacon_aggregator,
                             tree_layout=bacon_tree_layout,
                             concept_dropout=bacon_concept_dropout,
                             sinkhorn_final_temperature=bacon_sinkhorn_final_temp).to(device)
        else:
            from bacon_head import BaconCBM
            model = BaconCBM(list(model_arch) + [num_concepts, num_classes],
                             activation, backbone=backbone,
                             aggregator=bacon_aggregator,
                             tree_layout=bacon_tree_layout,
                             concept_dropout=bacon_concept_dropout,
                             sinkhorn_final_temperature=bacon_sinkhorn_final_temp).to(device)
    elif head_type == 'bacon_multi':
        if is_stochastic:
            from bacon_head import StochasticMultiTreeBaconCBM
            model = StochasticMultiTreeBaconCBM(list(model_arch) + [num_concepts, num_classes],
                             activation, backbone=backbone,
                             aggregator=bacon_aggregator,
                             tree_layout=bacon_tree_layout,
                             concept_dropout=bacon_concept_dropout,
                             use_negative_sampling=bacon_use_negative_sampling,
                             sinkhorn_final_temperature=bacon_sinkhorn_final_temp).to(device)
        else:
            from bacon_head import MultiTreeBaconCBM
            model = MultiTreeBaconCBM(list(model_arch) + [num_concepts, num_classes],
                             activation, backbone=backbone,
                             aggregator=bacon_aggregator,
                             tree_layout=bacon_tree_layout,
                             concept_dropout=bacon_concept_dropout,
                             use_negative_sampling=bacon_use_negative_sampling,
                             sinkhorn_final_temperature=bacon_sinkhorn_final_temp).to(device)
    elif is_stochastic:
        model = StochasticMLP(list(model_arch) + [num_concepts, num_classes],
                              activation, backbone=backbone).to(
            device)
    else:
        model = BasicMLP(list(model_arch) + [num_concepts, num_classes],
                         activation, backbone=backbone).to(device)
    if head_type in ('bacon', 'bacon_multi') and hasattr(model, 'param_groups'):
        params = model.param_groups(lr, head_lr_mult)
    else:
        params = model.parameters()
    if optimizer == 'adam':
        optim = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        optim = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    return model, optim, scheduler


def run_and_plot_interventions_uncertainty(model, dataloaders, device, logdir, test_acc):
    num_tti_groups_to_acc = measure_interventions(model, dataloaders['test'], device,
                                                  is_random=False, num_trials=1)

    with open(os.path.join(logdir, 'interventions_uncertainty.txt'), 'w') as f:
        for n_groups, accuracies in num_tti_groups_to_acc.items():
            print(n_groups, accuracies, file=f)
    fig, ax = plt.subplots()
    acc_tti = [(k, np.mean(acc)) for k, acc in num_tti_groups_to_acc.items()]
    acc_tti.sort()
    acc_tti_means = np.array([m for _, m in acc_tti])
    ax.plot(acc_tti_means)

    ax.set_xlabel('Number of intervened groups')
    ax.set_ylabel('Target accuracy')
    plt.savefig(os.path.join(logdir, "interventions_uncertainty.pdf"))
    plt.close(fig)
    normalized_auc = 0
    prev_value = test_acc
    for i, m in acc_tti:
        normalized_auc += (m - prev_value)
        prev_value = m
    normalized_auc /= len(acc_tti)
    logger.info(f"Uncertainty Interventions AUC = {np.mean([m for i, m in acc_tti])}")
    logger.info(f"Normalized Uncertainty Interventions AUC = {normalized_auc}")


def plot_losses(logdir, test_label_losses, test_concept_losses):
    fig, ax = plt.subplots()
    labels_losses = (test_label_losses - test_label_losses.min(1).reshape(-1, 1)) / (
            test_label_losses.max(1).reshape(-1, 1) - test_label_losses.min(1).reshape(-1, 1))
    labels_losses_std = labels_losses.std(0)
    labels_losses = labels_losses.mean(0)
    with open(os.path.join(logdir, "label_losses.txt"), "w") as f:
        print("t,loss,loss_std", file=f)
        for i, (val, std) in enumerate(zip(labels_losses, labels_losses_std)):
            print(f"{i},{val},{std}", file=f)
    ax.plot(labels_losses, label='Validation labels loss')
    plt.fill_between(range(len(labels_losses)), labels_losses - labels_losses_std,
                     labels_losses + labels_losses_std, alpha=0.5)

    concepts_loss = (test_concept_losses - test_concept_losses.min(1).reshape(-1, 1)) / (
            test_concept_losses.max(1).reshape(-1, 1) - test_concept_losses.min(1).reshape(-1, 1))
    concepts_loss_std = concepts_loss.std(0)
    concepts_loss = concepts_loss.mean(0)
    with open(os.path.join(logdir, "concept_losses.txt"), "w") as f:
        print("t,loss,loss_std", file=f)
        for i, (val, std) in enumerate(zip(concepts_loss, concepts_loss_std)):
            print(f"{i},{val},{std}", file=f)
    plt.fill_between(range(len(concepts_loss)), concepts_loss - concepts_loss_std,
                     concepts_loss + concepts_loss_std, alpha=0.5)
    ax.plot(concepts_loss, label='Validation concepts loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    if wb_run is not None:
        wb_run.log({'losses': fig})
    else:
        plt.savefig(os.path.join(logdir, "losses.pdf"))
    plt.close(fig)


def plot_mi(logdir, is_stochastic,
            MI_XC, MI_CY, MI_XZ, MI_ZY, MI_ZC):
    MI_XZ = np.clip(np.array(MI_XZ).T, 0, None)  # shape (num layers, num grad steps)
    MI_ZY = np.clip(np.array(MI_ZY).T, 0, None)  # shape (num layers, num grad steps)
    MI_ZC = np.array(MI_ZC).reshape(-1)
    MI_XZ = np.array(MI_XZ).reshape(-1)

    if is_stochastic:
        fig, ax = plt.subplots()
        MI_XC, MI_CY = np.clip(np.array(MI_XC), 0, None), np.clip(np.array(MI_CY), 0, None)
        color_cycle = cycler(color=plt.cm.rainbow(np.linspace(0, 1, len(MI_XC))))
        ax.set_prop_cycle(color_cycle)
        scatter = ax.scatter([], [], c=[], cmap=plt.cm.rainbow)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Training progress %')

        for i in range(len(MI_XC)):
            ax.plot(MI_XC[i], MI_CY[i], 'o')
        plt.xlabel('I(X;C)', fontsize=10)
        plt.ylabel('I(C;Y)', fontsize=10)
        if wb_run is not None:
            wb_run.log({'information_flow_xc_cy': fig})
        else:
            plt.savefig(os.path.join(logdir, "information_flow_xc_cy.pdf"))
        plt.close(fig)
    else:
        fig, ax = plt.subplots()
        MI_XC, MI_CY = MI_XZ, np.clip(np.array(MI_CY), 0, None)
        color_cycle = cycler(color=plt.cm.rainbow(np.linspace(0, 1, len(MI_XC))))
        ax.set_prop_cycle(color_cycle)
        scatter = ax.scatter([], [], c=[], cmap=plt.cm.rainbow)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Training progress %')

        for i in range(len(MI_XC)):
            ax.plot(MI_XC[i], MI_CY[i], 'o')
        plt.xlabel('I(X;C)', fontsize=10)
        plt.ylabel('I(C;Y)', fontsize=10)
        if wb_run is not None:
            wb_run.log({'information_flow_xc_cy': fig})
        else:
            plt.savefig(os.path.join(logdir, "information_flow_xc_cy.pdf"))
        plt.close(fig)
    with open(os.path.join(logdir, 'information_flow_xc_cy.txt'), 'w') as f:
        print("x,y,t", file=f)
        for i in range(len(MI_ZC)):
            print(f"{MI_XZ[i]},{MI_CY[i]},{i / len(MI_ZC)}", file=f)

    fig, ax = plt.subplots()
    color_cycle = cycler(color=plt.cm.rainbow(np.linspace(0, 1, len(MI_XZ))))
    ax.set_prop_cycle(color_cycle)
    scatter = ax.scatter([], [], c=[], cmap=plt.cm.rainbow)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Training progress %')

    for i in range(len(MI_ZC)):
        ax.plot(MI_XZ[i], MI_ZC[i], 'o')
    with open(os.path.join(logdir, 'information_flow_xzl_czl.txt'), 'w') as f:
        for i in range(len(MI_ZC)):
            print(f"{MI_XZ[i]},{MI_ZC[i]},{i}", file=f)
    plt.xlabel('I(X;Z)', fontsize=10)
    plt.ylabel('I(Z;C)', fontsize=10)
    if wb_run is not None:
        wb_run.log({'information_flow_xzl_czl': fig})
    else:
        plt.savefig(os.path.join(logdir, "information_flow_xzl_czl.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    layer_colors = plt.cm.viridis(np.linspace(0, 1, len(MI_XZ)))

    for layer, layer_color in enumerate(layer_colors):
        color_cycle = cycler(color=plt.cm.rainbow(np.linspace(0, 1, len(MI_ZY[0]))))
        ax.set_prop_cycle(color_cycle)

        ax.plot(MI_XZ[layer], MI_CY, '-', color=layer_color, alpha=0.5, linewidth=2,
                label=f'Layer {layer + 1}')

        for i in range(len(MI_XZ[layer])):
            ax.plot(MI_XZ[layer][i], MI_CY[i], 'o', markersize=5)

    scatter = ax.scatter([], [], c=[], cmap=plt.cm.rainbow)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Training progress %', fontsize=10)

    plt.xlabel('I(X;Z)', fontsize=12)
    plt.ylabel('I(Z;Y)', fontsize=12)
    plt.legend(fontsize=10, title="Layers", title_fontsize=12)
    plt.tight_layout()
    if wb_run is not None:
        wb_run.log({'information_flow_xz_zy': fig})
    else:
        plt.savefig(os.path.join(logdir, "information_flow_xz_zy.pdf"))
    plt.close(fig)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    Fire(run_experiment)