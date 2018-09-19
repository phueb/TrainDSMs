import matplotlib.pyplot as plt


def make_classifier_figs(model):
    if not os.path.isfile(model.hc_data_path):
        model.run_classifier()  # saves results to disk

    def make_classifier_accuracy_fig():
        """
        Returns fig showing accuracy of a classifier tasked to map hidden representations of probes to categories
        """
        # load data
        npzfile = np.load(model.hc_data_path)
        train_hca_traj = npzfile['train_hca_cat_traj_mat'].mean(axis=0)
        test_hca_traj = npzfile['test_hca_cat_traj_mat'].mean(axis=0)
        # fig
        fig, ax = plt.subplots(1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_ylim([0, 110])
        ax.set_xlabel('Training steps', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylabel = 'Accuracy'
        ylabel += ' (shuffled cats, ' if AppConfigs.HC_PARAMS_DICT['shuffle_cats'] else ' ('
        ylabel += 'proto)' if AppConfigs.HC_PARAMS_DICT['rep_type'] == 'proto' else 'exemplar)'
        ax.set_ylabel(ylabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.yaxis.grid(True)
        # plot
        ax.plot(train_hca_traj, '-', linewidth=FigsConfigs.LINEWIDTH, label='train')
        ax.plot(test_hca_traj, '-', linewidth=FigsConfigs.LINEWIDTH, label='test')
        ax.legend(loc='best')
        plt.tight_layout()
        return fig

    def make_classifier_accuracy_breakdown_fig():
        """
        Returns fig showing accuracy broken down by category of a classifier tasked to map hidden representations
         of probes to categories
        """
        start = time.time()

        # load data
        npzfile = np.load(model.hc_data_path)
        train_hca_cat_traj_mat = npzfile['train_hca_cat_traj_mat']
        test_hca_cat_traj_mat = npzfile['test_hca_cat_traj_mat']
        x_mat = npzfile['x_mat']
        # fig
        fig, axarr = plt.subplots(model.hub.probe_store.num_cats, 1,
                                  figsize=(FigsConfigs.MAX_FIG_WIDTH, 3 * model.hub.probe_store.num_cats),
                                  dpi=FigsConfigs.DPI)
        for ax, train_hca_cat_traj, test_hca_cat_traj, cat, x in zip(
                axarr, train_hca_cat_traj_mat, test_hca_cat_traj_mat, model.hub.probe_store.cats, x_mat):
            ax.set_ylim([0, 110])
            max_x = np.max(x_mat[:, -1])
            ax.set_xlim([0, max_x])
            ax.set_xlabel('Training steps', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            ylabel = 'Accuracy'
            ylabel += ' (shuffled cats, ' if AppConfigs.HC_PARAMS_DICT['shuffle_cats'] else ' ('
            ylabel += 'proto)' if AppConfigs.HC_PARAMS_DICT['rep_type'] == 'proto' else 'exemplar)'
            ax.set_ylabel(ylabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='both', top='off', right='off')
            ax.yaxis.grid(True)
            # plot
            ax.plot(x, train_hca_cat_traj, '-', linewidth=FigsConfigs.LINEWIDTH, label='{} (train)'.format(cat))
            ax.plot(x, test_hca_cat_traj, '-', linewidth=FigsConfigs.LINEWIDTH, label='{} (test)'.format(cat))
            ax.legend()
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_classifier_conf_mat_fig():
        """
        Returns fig showing confusionmatrix of classifier trained to map hidden representations of probes to categories
        """
        print('Making data for classifier (rep_type={}, shuffle_cats={})...'.format(
            AppConfigs.HC_PARAMS_DICT['rep_type'], AppConfigs.HC_PARAMS_DICT['shuffle_cats']))
        start = time.time()

        # load data
        npzfile = np.load(model.hc_data_path)
        cm = npzfile['cm']
        # fig
        fig, ax = plt.subplots(1, 1, figsize=(FigsConfigs.MAX_FIG_WIDTH, FigsConfigs.MAX_FIG_WIDTH),
                               dpi=FigsConfigs.DPI)
        # plot
        sns.heatmap(cm, ax=ax, square=True, annot=False, cbar_kws={"shrink": .5}, cmap='jet', vmin=0, vmax=100)
        # axis 2 (needs to be below plot for axes to be labeled)
        ax.set_yticklabels([cat + ' (truth)' for cat in sorted(model.hub.probe_store.cats, reverse=True)], rotation=0)
        ax.set_xticklabels(model.hub.probe_store.cats, rotation=90)
        ax.set_title('Confusion Matrix')
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 50, 100])
        cbar.set_ticklabels(['0%', '50%', '100%'])
        cbar.set_label('Hits & False Alarms')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_classifier_accuracy_fig(),
            make_classifier_accuracy_breakdown_fig(),
            make_classifier_conf_mat_fig()]
    return figs
