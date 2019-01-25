import h5py
from utils.eval_utils import get_uncertainty_precision, plot_precision_recall_curve, compute_metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from matplotlib import colors

recall_do, npv_do, acc_do, precision_do, recall_do_, auc_do, t_1 = compute_metrics(run_name='dropout_mean_uncertainty')
recall_dc, npv_dc, acc_dc, precision_dc, recall_dc_, auc_dc, t_2 = compute_metrics(run_name='dropout_pred_uncertainty')

# recall_do, npv_do, acc_do, precision_do, auc_do, t_1 = compute_metrics(run_name='camvid_dropout')
# recall_dc, npv_dc, acc_dc, precision_dc, auc_dc, t_2 = compute_metrics(run_name='camvid_dropconnect')
# recall_dca, npv_dca, acc_dca, precision_dca, auc_dca, t_3 = compute_metrics(run_name='camvid_all_dropconnect')


fig, axs = plt.subplots(1, 4)
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=None)

ax = axs[0]
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
# ax.step(recall_do, precision_do, color='r', alpha=0.2, where='post')
ax.plot(recall_do_, precision_do, color='r')
# ax.fill_between(recall_do, precision_do, alpha=0.2, color='r')

# ax.step(recall_dc, precision_dc, color='g', alpha=0.2, where='post')
ax.plot(recall_dc_, precision_dc, color='b')
# ax.fill_between(recall_dc, precision_dc, alpha=0.2, color='g')

# ax.step(recall_dca, precision_dca, color='b', alpha=0.2, where='post')
# ax.plot(recall_dca, precision_dca, color='g')
# ax.fill_between(recall_dca, precision_dca, alpha=0.2, color='b')

ax.legend(['dropout, AUC={0:.3f}'.format(auc_do),
           'dropconnect, AUC={0:.3f}'.format(auc_dc)])
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.05])


ax = axs[1]
ax.plot(t_1, recall_do, 'o-', c='r', markersize=2)
ax.plot(t_2, recall_dc, 'o-', c='b', markersize=2)
# ax.plot(t_3, recall_dca, 'o-', c='g', markersize=2)
ax.set_ylabel('Recall')
ax.set_xlabel('T')

ax = axs[2]
ax.plot(t_1, npv_do, 'o-', c='r', markersize=2)
ax.plot(t_2, npv_dc, 'o-', c='b', markersize=2)
# ax.plot(t_3, npv_dca, 'o-', c='g', markersize=2)
ax.set_ylabel('NPV')
ax.set_xlabel('T')

ax = axs[3]
ax.plot(t_1, acc_do, 'o-', c='r', markersize=2)
ax.plot(t_2, acc_dc, 'o-', c='b', markersize=2)
# ax.plot(t_3, acc_dca, 'o-', c='g', markersize=2)
ax.set_ylabel('Uncertainty Accuracy')
ax.set_xlabel('T')

print()


# cmap = colors.ListedColormap(['red', 'green'])
# bounds = [0, 0.5, 1]
# norm = colors.BoundaryNorm(bounds, cmap.N)
# correct = (all_mask == all_pred).astype(int)
# plt.imshow(correct[1], cmap=cmap, norm=norm)
# plt.title('right vs. wrong')
#

# # uncertainty_measure = get_uncertainty_precision(all_mask, all_pred, all_var)
# precision, recall, average_precision1 = plot_precision_recall_curve(all_mask, all_pred, all_var)
# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})
# plt.step(recall, precision, color='b', alpha=0.2, where='post')
# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
#
#
# run_name = 'camvid_dropconnect'
# h5f = h5py.File(run_name + '_bayes.h5', 'r')
# all_mask = h5f['y'][:]
# all_pred = h5f['y_pred'][:]
# all_var = h5f['y_var'][:]
# h5f.close()
#
# p_uw_dc, p_cr_dc, precision_dc = compute_metrics(all_mask, all_pred, all_var)
# precision, recall, average_precision2 = plot_precision_recall_curve(all_mask, all_pred, all_var)
# # step_kwargs = ({'step': 'post'}
# #                if 'step' in signature(plt.fill_between).parameters
# #                else {})
# # plt.step(recall, precision, color='r', alpha=0.2, where='post')
# # plt.fill_between(recall, precision, alpha=0.2, color='r', **step_kwargs)
#
#
# run_name = 'camvid_all_dropconnect'
# h5f = h5py.File(run_name + '_bayes.h5', 'r')
# all_mask = h5f['y'][:]
# all_pred = h5f['y_pred'][:]
# all_var = h5f['y_var'][:]
# h5f.close()
#
# p_uw_dca, p_cr_dca, precision_dca = compute_metrics(all_mask, all_pred, all_var)
# precision, recall, average_precision3 = plot_precision_recall_curve(all_mask, all_pred, all_var)
# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})
# plt.step(recall, precision, color='g', alpha=0.2, where='post')
# plt.fill_between(recall, precision, alpha=0.2, color='g', **step_kwargs)
#
#
# plt.legend(['AP_dropout={0:.2f}'.format(average_precision1),
#             'AP_dropconnect={0:.2f}'.format(average_precision2),
#             'AP_all_dropconnect={0:.2f}'.format(average_precision3)])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.show()
# print()
