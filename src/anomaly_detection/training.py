import json
import pickle
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from src.anomaly_detection.eval.eval import evaluate, evaluate_torch, ocsvm_max_train
from src.anomaly_detection.utils import WANDB_ENTITY, WANDB_PROJECT

import wandb


def train_model(model, data, config, kfolds=5, eval=True):
    """
    Train the model
    :param model: Model to train
    :param data: Data
    :param kfolds: Number of splits in KFold
    :return: Scores and if eval is True, EM and MV values, EM and MV curves, time and alpha axis, maximum
    """
    # Prepare the data for KFold
    kf = KFold(n_splits=kfolds)
    y_scores = np.zeros_like(data[:, 0])

    # Prepare the evaluation results
    em_vals = []
    mv_vals = []
    em_curves = []
    mv_curves = []
    t = -1
    axis_alpha = -1
    amax = -1

    for iter, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Training fold {iter + 1} / {kfolds}")
        wandb.init(
            group=json.dumps(config),
            name=f"{json.dumps(config)}_fold_{iter + 1}",
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            tags=["pokus"],  # TODO: Remove this, when testing of wandb is done
            config=config
        )

        # Fit the model
        if model.__class__.__name__ == "OneClassSVM":
            model.fit(data[train_index][:min(ocsvm_max_train, len(data[train_index]) - 1)])
        else:
            model.fit(data[train_index])
        # Save the model
        with open(f"models/{json.dumps(config)}_fold_{iter + 1}.pkl", "wb") as fp:
            pickle.dump(model, fp)
        # Load in the future as:
        # with open(f"models/{json.dumps(config)}_fold_{iter + 1}.pkl", "rb") as fp:
        #     model = pickle.load(fp)

        # Predict the anomalies in the data
        y_scores[test_index] = model.score_samples(data[test_index])

        if eval:
            # Evaluate the model
            print("Evaluating the model...")
            em_val, mv_val, em_curve, mv_curve, t_, axis_alpha_, amax_ = evaluate(model, data[train_index], data[test_index], averaging=10)

            for em in em_curve:
                wandb.log({"EM": em})
            wandb.log({"EM_final": em_val})
            for mv in mv_curve:
                wandb.log({"MV": mv})
            wandb.log({"MV_final": mv_val})

            em_vals.append(em_val)
            mv_vals.append(mv_val)
            em_curves.append(em_curve)
            mv_curves.append(mv_curve)
            t = t_
            axis_alpha = axis_alpha_
            amax = max(amax, amax_)

        wandb.finish()

    # Average the evaluation results
    if eval:
        em_val = np.mean(em_vals)
        mv_val = np.mean(mv_vals)
        em_curve = np.mean(em_curves, axis=0)
        mv_curve = np.mean(mv_curves, axis=0)

        return y_scores, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax

    return y_scores


def train_torch_model(model, data_loader, config, num_epochs=10, lr=1e-5, kfolds=5, eval=True):
    """
    Train the torch model
    :param model: Model to train
    :param data_loader: Data loader
    :param config: Configuration of the model
    :param num_epochs: Number of epochs
    :param lr: Learning rate
    :param kfolds: Number of splits in KFold
    :param eval: Evaluate the model
    :return: Scores and if eval is True, EM and MV values, EM and MV curves, time and alpha axis, maximum
    """
    # Prepare the data for KFold
    kf = KFold(n_splits=kfolds)

    # y_scores needs to be (data_len, seq_len)
    y_scores = np.zeros_like(data_loader.dataset.data[:, 0].cpu().numpy())

    # Prepare the evaluation results
    em_vals = []
    mv_vals = []
    em_curves = []
    mv_curves = []
    t = -1
    axis_alpha = -1
    amax = -1

    for iter, (train_index, test_index) in enumerate(kf.split(data_loader.dataset.data)):
        print(f"Training fold {iter + 1} / {kfolds}")
        wandb.init(
            group=json.dumps(config),
            name=f"{json.dumps(config)}_fold_{iter + 1}",
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            tags=["pokus"],  # TODO: Remove this, when testing of wandb is done
            config=config
        )

        # Create subset data loaders
        train_loader = DataLoader(Subset(data_loader.dataset, train_index), batch_size=data_loader.batch_size)
        test_loader = DataLoader(Subset(data_loader.dataset, test_index), batch_size=data_loader.batch_size)

        # Reset model parameters
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        # Fit the model
        model.fit(train_loader, test_loader, num_epochs=num_epochs, lr=lr)
        # Save the model
        model.save_model(f"models/{json.dumps(config)}_fold_{iter + 1}")
        # Load in the future as:
        # model.load_model(f"models/{json.dumps(config)}_fold_{iter + 1}")

        # Predict the anomalies in the data
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                from_idx = i * data_loader.batch_size
                to_idx = min((i + 1) * data_loader.batch_size, len(test_index))
                y_scores[test_index[from_idx:to_idx]] = model.decision_function(batch)

        if eval:
            # Evaluate the model
            print("Evaluating the model...")
            em_val, mv_val, em_curve, mv_curve, t_, axis_alpha_, amax_ = evaluate_torch(model, train_loader, test_loader, num_epochs=num_epochs, lr=lr, averaging=10)

            for em in em_curve:
                wandb.log({"EM": em})
            wandb.log({"EM_final": em_val})
            for mv in mv_curve:
                wandb.log({"MV": mv})
            wandb.log({"MV_final": mv_val})

            em_vals.append(em_val)
            mv_vals.append(mv_val)
            em_curves.append(em_curve)
            mv_curves.append(mv_curve)
            t = t_
            axis_alpha = axis_alpha_
            amax = max(amax, amax_)

        wandb.finish()

    # Average the evaluation results
    if eval:
        em_val = np.mean(em_vals)
        mv_val = np.mean(mv_vals)
        em_curve = np.mean(em_curves, axis=0)
        mv_curve = np.mean(mv_curves, axis=0)

        return y_scores, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax

    return y_scores
