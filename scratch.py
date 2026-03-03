from regression import LogisticRegressor
from regression import BaseRegressor
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

import numpy as np
# import os
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score
import random
import time
import copy


# import data
# data/nsclc.csv
# Load data
X_train, X_val, y_train, y_val = utils.loadDataset(
    features=[
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ],
    split_percent=0.8,
    split_seed=42
)

# Scale the data, since values vary across feature. Note that we
# fit on the training data and use the same scaler for X_val.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)


# log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=100, batch_size=100)
log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
log_model.train_model(X_train, y_train, X_val, y_val)
y_prob = log_model.make_prediction(X_val)
print("Val AUC:", roc_auc_score(y_val, y_prob))
print("Val BCE:", log_model.loss_function(y_val, y_prob))
print("Val Acc:", accuracy_score(y_val, (y_prob >= 0.5).astype(int)))

# log_model.plot_loss_history()








# import os


# # For testing purposes: run a focused hyperparameter search over
# # only learning_rate, max_iter, and batch_size (random sampling).
# def eval_model_on_val(model, X_val, y_val):
#     y_prob = model.make_prediction(X_val)
#     val_loss = model.loss_function(y_val, y_prob)
#     try:
#         val_auc = roc_auc_score(y_val, y_prob)
#     except ValueError:
#         val_auc = float('nan')
#     y_pred = (y_prob >= 0.5).astype(int)
#     val_acc = accuracy_score(y_val, y_pred)
#     return {'loss': val_loss, 'auc': val_auc, 'acc': val_acc}


# def focused_random_search(num_trials=12, seed=0):
#     random.seed(seed)
#     np.random.seed(seed)
#     trials = []
#     best = None

#     # Sampling helpers for only the three hyperparams requested
#     def sample_lr():
#         return 10 ** np.random.uniform(-5, -2)  # 1e-5 .. 1e-2

#     def sample_batch():
#         return int(random.choice([8, 16, 32, 64]))

#     def sample_max_iter():
#         return int(random.choice([50, 100, 200, 500]))

#     for t in range(num_trials):
#         lr = sample_lr()
#         batch = sample_batch()
#         epochs = sample_max_iter()

#         model = LogisticRegressor(num_feats=X_train.shape[1],
#                                   learning_rate=lr,
#                                   tol=0.001,
#                                   max_iter=epochs,
#                                   batch_size=batch)
#         model.reset_model()
#         start = time.time()
#         model.train_model(X_train, y_train, X_val, y_val)
#         elapsed = time.time() - start

#         metrics = eval_model_on_val(model, X_val, y_val)
#         trial = {
#             'trial': t,
#             'lr': lr,
#             'batch_size': batch,
#             'max_iter': epochs,
#             'val_loss': metrics['loss'],
#             'val_auc': metrics['auc'],
#             'val_acc': metrics['acc'],
#             'time_s': elapsed,
#             'model_W': model.W.copy(),
#             'loss_hist_train': np.array(model.loss_hist_train),
#             'loss_hist_val': np.array(model.loss_hist_val)
#         }
#         trials.append(trial)

#         # update best by AUC (ignore NaN)
#         if best is None or (not np.isnan(trial['val_auc']) and trial['val_auc'] > best['val_auc']):
#             best = trial

#         print(f"Trial {t+1}/{num_trials}: lr={lr:.2e}, batch={batch}, max_iter={epochs} -> "
#               f"AUC={trial['val_auc']:.4f}, loss={trial['val_loss']:.4f}, acc={trial['val_acc']:.4f} ({elapsed:.1f}s)")

#     # sort by AUC (descending, placing NaNs at the end)
#     trials_sorted = sorted(trials, key=lambda x: (np.nan if np.isnan(x['val_auc']) else -x['val_auc']))
#     print("\nTop 5 trials by val AUC:")
#     for i, tr in enumerate(trials_sorted[:5]):
#         print(f"{i+1}. AUC={tr['val_auc']:.4f}, loss={tr['val_loss']:.4f}, acc={tr['val_acc']:.4f} "
#               f"lr={tr['lr']:.2e}, batch={tr['batch_size']}, max_iter={tr['max_iter']}, time={tr['time_s']:.1f}s")

#     return best, trials


# # Run the focused search and save best results
# best_trial, all_trials = focused_random_search(num_trials=12, seed=42)
# print("\nBest trial:", best_trial)

# # Ensure artifacts dir exists
# os.makedirs('artifacts', exist_ok=True)

# # Save best model weights and plot the best trial's loss history
# best_weights = best_trial['model_W']
# np.save('artifacts/best_logreg_weights.npy', best_weights)
# print("Saved best weights to artifacts/best_logreg_weights.npy")

# # Save best loss history plot
# fig, axs = plt.subplots(2, figsize=(8, 8))
# fig.suptitle('Best Trial Loss History')
# if best_trial['loss_hist_train'] is not None and len(best_trial['loss_hist_train']) > 0:
#     axs[0].plot(np.arange(len(best_trial['loss_hist_train'])), best_trial['loss_hist_train'])
# axs[0].set_title('Training')
# if best_trial['loss_hist_val'] is not None and len(best_trial['loss_hist_val']) > 0:
#     axs[1].plot(np.arange(len(best_trial['loss_hist_val'])), best_trial['loss_hist_val'])
# axs[1].set_title('Validation')
# plt.xlabel('Steps')
# fig.tight_layout()
# out_path = os.path.join('artifacts', 'loss_history_best.png')
# fig.savefig(out_path)
# print(f"Saved best trial loss plot to: {out_path}")
