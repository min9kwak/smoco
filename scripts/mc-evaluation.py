import numpy as np
from datasets.brain import BrainProcessor
from easydict import EasyDict as edict
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import tqdm


config = edict()
config.root = 'D:/data/ADNI'
config.data_info = 'labels/data_info.csv'
config.mci_only = True
config.n_splits = 10
config.n_cv = 0
config.data_type = 'pet'

####
result_fin = {}
for random_state in tqdm.tqdm(range(2021, 2031)):
    for n_cv in range(0, 10):
        config.random_state = random_state
        config.n_cv = n_cv

        # load data
        data_processor = BrainProcessor(root=config.root,
                                        data_info=config.data_info,
                                        data_type=config.data_type,
                                        mci_only=config.mci_only,
                                        random_state=config.random_state)
        datasets = data_processor.process(n_splits=config.n_splits, n_cv=config.n_cv)

        mc = datasets['test']['mc']
        y = datasets['test']['y']

        result_dict = {'all': {}, '20': {}, '37': {}}

        # all
        auroc_ = roc_auc_score(y, mc)

        fpr, tpr, thresholds = roc_curve(y, mc)
        dist = fpr ** 2 + (1 - tpr) ** 2
        threshold = thresholds[np.argmin(dist)]

        cm = confusion_matrix(y_true=y, y_pred=mc > threshold)
        n00, n01, n10, n11 = cm.reshape(-1, ).tolist()
        accuracy_ = (n00 + n11) / (n00 + n01 + n10 + n11)
        sensitivity_ = n11 / (n11 + n10 + 1e-7)
        specificity_ = n00 / (n00 + n01 + 1e-7)
        precision_ = n11 / (n11 + n01 + 1e-7)
        f1_ = (2 * precision_ * sensitivity_) / (precision_ + sensitivity_ + 1e-7)
        gmean_ = np.sqrt(sensitivity_ * specificity_)
        result_dict['all'] = dict(acc=accuracy_, auroc=auroc_, sens=sensitivity_, spec=specificity_,
                                  prec=precision_, f1=f1_, gmean=gmean_, threshold=threshold)

        for cutoff in [20, 37]:

            pred = np.array([m > cutoff for m in mc], dtype=float)
            true = np.array(y, dtype=float)

            cm = confusion_matrix(true, pred)
            n00, n01, n10, n11 = cm.reshape(-1, ).tolist()

            accuracy_ = (n00 + n11) / (n00 + n01 + n10 + n11)
            sensitivity_ = n11 / (n11 + n10 + 1e-7)
            specificity_ = n00 / (n00 + n01 + 1e-7)
            precision_ = n11 / (n11 + n01 + 1e-7)
            f1_ = (2 * precision_ * sensitivity_) / (precision_ + sensitivity_ + 1e-7)
            gmean_ = np.sqrt(sensitivity_ * specificity_)

            result_dict[str(cutoff)] = dict(acc=accuracy_, auroc=auroc_, sens=sensitivity_, spec=specificity_,
                                            prec=precision_, f1=f1_, gmean=gmean_, threshold=float(cutoff))

        result_fin[random_state*10+n_cv] = result_dict

#
summary = {'all': {}, '20': {}, '37': {}}
metrics = result_dict['all'].keys()
for k in summary.keys():
    for m in metrics:
        summary[k][m] = []

for random_state, result_ in result_fin.items():
    for cutoff, result in result_.items():
        for m, v in result.items():
            summary[cutoff][m].append(v)

from copy import deepcopy
summary_fin = deepcopy(summary)
for cutoff, res in summary.items():
    for m, values in res.items():
        mean, std = np.mean(values), np.std(values)
        r = f'{mean*100:.2f} ({std*100:.2f})'
        summary_fin[cutoff][m] = r

for cutoff, res in summary_fin.items():
    print(cutoff, ' -----')
    for m, v in res.items():
        print(m, '\t', v)