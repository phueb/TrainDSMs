import numpy as np
from bayes_opt import BayesianOptimization

from two_process_nlp import config


def calc_accuracy(eval_sims_mat, eval_probes, eval_candidates_mat, verbose=False):
    """
    eval_sims has same shape as eval_candidates_mat (to save memory)
    """
    assert eval_sims_mat.shape == eval_candidates_mat.shape
    num_correct = 0
    for eval_sims_row in eval_sims_mat:
        for correct_id in range(config.Eval.min_num_relata):
            if np.all(eval_sims_row[config.Eval.min_num_relata:] < eval_sims_row[correct_id]):  # there can be multiple correct  # TODO test
                num_correct += 1
    res = num_correct / (len(eval_probes) * config.Eval.min_num_relata)
    #
    if verbose:
        for eval_sims_row, eval_candidates_row, eval_probe in zip(
                eval_sims_mat, eval_candidates_mat, eval_probes):
            print('------------')
            print(eval_probe)
            print('------------')
            for correct_id in range(config.Eval.min_num_relata):
                print(eval_candidates_row[config.Eval.min_num_relata:])
                print(eval_sims_row[config.Eval.min_num_relata:])
                print(eval_candidates_row[correct_id])
                print(eval_sims_row[correct_id])
                #
                if np.all(eval_sims_row[config.Eval.min_num_relata:] < eval_sims_row[correct_id]):  # there can be multiple correct  # TODO test
                    print('correct')
                else:
                    print('false')
                print('-')
            print()
    return res


def calc_cluster_score(calc_signals, sims_mean, verbose=True):

    def calc_probes_fs(thr):
        tp, tn, fp, fn = calc_signals(thr)
        precision = np.divide(tp + 1e-10, (tp + fp + 1e-10))
        sensitivity = np.divide(tp + 1e-10, (tp + fn + 1e-10))  # aka recall
        fs = 2 * (precision * sensitivity) / (precision + sensitivity)

        # TODO debug f-score  - it is below 0.5

        print('prec={:.2f} sens={:.2f}, | tp={} tn={} | fp={} fn={}'.format(precision, sensitivity, tp, tn, fp, fn))
        return fs

    def calc_probes_ck(thr):
        """
        cohen's kappa
        """
        tp, tn, fp, fn = calc_signals(thr)
        totA = np.divide(tp + tn, (tp + tn + fp + fn))
        #
        pyes = ((tp + fp) / (tp + fp + tn + fn)) * ((tp + fn) / (tp + fp + tn + fn))
        pno = ((fn + tn) / (tp + fp + tn + fn)) * ((fp + tn) / (tp + fp + tn + fn))
        #
        randA = pyes + pno
        ck = (totA - randA) / (1 - randA)
        # print('totA={:.2f} randA={:.2f}'.format(totA, randA))
        return ck

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals(thr)
        specificity = np.divide(tn + 1e-10, (tn + fp + 1e-10))
        sensitivity = np.divide(tp + 1e-10, (tp + fn + 1e-10))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    if verbose:
        print('Finding best thresholds between using bayesian-optimization...')
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    if config.Eval.matching_metric == 'F1':
        fun = calc_probes_fs
    elif config.Eval.matching_metric == 'BalAcc':
        fun = calc_probes_ba
    elif config.Eval.matching_metric == 'CohensKappa':
        fun = calc_probes_ck
    else:
        raise AttributeError('rnnlab: Invalid arg to "metric".')
    bo = BayesianOptimization(fun, {'thr': (-1.0, 1.0)}, verbose=verbose)
    bo.explore({'thr': [sims_mean]})  # keep bayes-opt at version 0.6 because 1.0 occasionally returns 0.50 wrongly
    bo.maximize(init_points=2, n_iter=config.Eval.num_opt_steps,
                acq="poi", xi=0.01, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = fun(best_thr)
    res = np.mean(results)
    return res