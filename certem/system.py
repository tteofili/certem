import os
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
from pathlib import Path
import graphviz
import seaborn as sns
from certa.utils import merge_sources


# custom plot for saliencies
def custom_plot(df, name):
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    df.plot(kind = 'bar',ax=ax1)
    img_path = 'data_new/img/'+name+'.png'
    plt.savefig(img_path)
    plt.close()
    return  [f1, img_path]

# datasets
datasets = [name for name in os.listdir("data")]
datasets.remove('img')

#training data
datadirs = dict()
datadirs['AB'] = 'datasets/abt_buy'
datadirs['BA'] = 'datasets/beers'
datadirs['IA'] = 'datasets/itunes_amazon'
train_dfs = dict()

for dataset in datasets:
    datadir = datadirs[dataset]
    lsource = pd.read_csv(datadir + '/tableA.csv')
    rsource = pd.read_csv(datadir + '/tableB.csv')
    gt = pd.read_csv(datadir + '/train.csv')
    train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])
    train_dfs[dataset] = train_df

# data selection widgets
datasets_dropdown = widgets.Dropdown(
    options=datasets,
    value=datasets[0],
    description='Dataset',
    disabled=False,
)

gt_filter = widgets.RadioButtons(
    options=['Any', 'NO-MATCH', 'MATCH'],
    description='Label',
    disabled=False
)

pred_filter = widgets.RadioButtons(
    options=['Any', 'NO-MATCH', 'MATCH'],
    description='Prediction',
    disabled=False
)

sys_label = widgets.Label(
    value='ER Systems'
)
de_cb = widgets.Checkbox(
    value=True,
    description='DeepER',
    disabled=False,
    indent=False,
)
dm_cb = widgets.Checkbox(
    value=True,
    description='DeepMatcher',
    disabled=False,
    indent=False,
)
dt_cb = widgets.Checkbox(
    value=True,
    description='Ditto',
    disabled=False,
    indent=False,
)

box_layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center')

cf_name_dict = {'shapc' : 'SHAP-C', 'limec': 'LIME-C', 'certa': 'CERTA', 'dice_random' : 'DiCE'}

# color maps
cb = sns.light_palette("blue", as_cmap=True)
cr = sns.light_palette("red", as_cmap=True)
cg = sns.light_palette("green", as_cmap=True)

def highlight_prediction(x, columns=['DeepER', 'DeepMatcher', 'Ditto', 'prediction', 'label', 'match_score']):
    rh = f"background-color:red"
    gh = f"background-color:green"
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    for column in columns:
        if column in x.columns:
            ig = x[column] > 0.5
            ir = x[column] < 0.5
            df1.loc[ir, column] = rh
            df1.loc[ig, column] = gh
    return df1


out2 = widgets.Output()


def f(dataset, deeper, dm, ditto, pred_filter, gt_filter):
    out2.clear_output()
    samples = pd.read_csv('data_new/' + dataset + '/samples.csv').drop(['ltable_id', 'rtable_id'], axis=1)
    if not deeper:
        samples = samples.drop(['DeepER'], axis=1)
    if not dm:
        samples = samples.drop(['DeepMatcher'], axis=1)
    if not ditto:
        samples = samples.drop(['Ditto'], axis=1)
    if gt_filter == 'NO-MATCH':
        samples = samples[samples['label'] == 0]
    if gt_filter == 'MATCH':
        samples = samples[samples['label'] == 1]
    if pred_filter == 'NO-MATCH':
        samples = samples[samples['DeepER'] < 0.5]
    if pred_filter == 'MATCH':
        samples = samples[('DeepER' in samples.columns and samples['DeepER'] > 0.5) | (
                    'DeepMatcher' in samples.columns and samples['DeepMatcher'] > 0.5) | (
                                      'Ditto' in samples.columns and samples['Ditto'] > 0.5)]
    samples = samples.loc[:, ~samples.columns.str.contains('^Unnamed')]
    explain_buttons = []
    for idx in samples.index:
        explain_button = widgets.Button(description="Explain Item " + str(idx))

        def on_explain_clicked(b):
            out2.clear_output()
            saliencies = dict()
            cfs = dict()
            item_idx = int(b.description[-1])
            if deeper:
                saliency = pd.read_csv('data_new/' + dataset + '/DeepER/certa.csv')['explanation'].iloc[item_idx]
                first_cf = pd.read_csv('data_new/' + dataset + '/DeepER/' + str(item_idx) + '/certa.csv').iloc[0]
                saliencies['DeepER'] = saliency
                cfs['DeepER'] = first_cf.copy()
            if dm:
                saliency = pd.read_csv('data_new/' + dataset + '/DeepMatcher/certa.csv')['explanation'].iloc[item_idx]
                first_cf = pd.read_csv('data_new/' + dataset + '/DeepMatcher/' + str(item_idx) + '/certa.csv').iloc[0]
                saliencies['DeepMatcher'] = saliency
                cfs['DeepMatcher'] = first_cf.copy()
            if ditto:
                saliency = pd.read_csv('data_new/' + dataset + '/Ditto/certa.csv')['explanation'].iloc[item_idx]
                first_cf = pd.read_csv('data_new/' + dataset + '/Ditto/' + str(item_idx) + '/certa.csv').iloc[0]
                saliencies['Ditto'] = saliency
                cfs['Ditto'] = first_cf.copy()
            saliencies_box = []
            saliency_dfs = []
            for k in saliencies.keys():
                saliency_df = pd.DataFrame(eval(saliencies[k]), index=[0])
                cnv, path = custom_plot(saliency_df, dataset + '_' + k + '_' + str(item_idx))
                img = widgets.Image(value=open(path, 'rb').read(), format='png')

                saliency_df['model'] = k
                saliency_dfs.append(saliency_df)

                inspect_button = widgets.Button(description='Inspect ' + k)

                def inspect_button_click(ib):
                    selected_model = ib.description[8:]
                    single_pred = samples.iloc[item_idx]
                    for sm in ['DeepER', 'DeepMatcher', 'Ditto']:
                        if sm != selected_model:
                            single_pred = single_pred.drop(sm)
                    expl_data_df = pd.read_csv('data_new/' + dataset + '/' + selected_model + '/certa.csv')
                    pnn_df = pd.DataFrame(eval(expl_data_df['explanation'].iloc[item_idx]), index=[0])

                    pss_dict = eval(expl_data_df['summary'].iloc[item_idx])
                    pss_dfs = dict()
                    for k, v in pss_dict.items():
                        attrs = k.split('/')
                        no_attrs = len(attrs)
                        if not no_attrs in pss_dfs:
                            pss_dfs[no_attrs] = {' '.join(attrs): v}
                        else:
                            cd = pss_dfs[no_attrs]
                            cd.update({' '.join(attrs): v})
                            pss_dfs[no_attrs] = cd
                    pss_outs = []
                    for k, v in pss_dfs.items():
                        pssk_out = widgets.Output()
                        with pssk_out:
                            display(pd.DataFrame(v, index=[0]).style.background_gradient(cmap=cg, axis=1, low=0.1,
                                                                                         high=0.6))
                        pss_outs.append(pssk_out)
                    pred_out = widgets.Output()
                    with pred_out:
                        display(pd.DataFrame(single_pred).T.style.apply(highlight_prediction, axis=None))
                    out_pnn = widgets.Output()
                    with out_pnn:
                        display(pnn_df.style.background_gradient(cmap=cr, axis=1, low=0.1, high=0.6))

                    item_data_path = 'data_new/' + dataset + '/' + selected_model + '/' + str(item_idx) + '/'
                    tr_files = [f for f in Path(item_data_path).iterdir() if f.match("triangle_*.csv")]
                    tr_dfs = []
                    for tr_file in tr_files:
                        tr_dfs.append(pd.read_csv(tr_file).drop(['Unnamed: 0'], axis=1))
                    lt_files = [f for f in Path(item_data_path).iterdir() if f.match("lattice_*.dot")]
                    tr_slider = widgets.IntSlider(value=0, min=0, max=len(tr_files) - 1, step=1,
                                                  description='Triangle:', disabled=False, continuous_update=False,
                                                  orientation='horizontal',
                                                  readout=True, readout_format='d')

                    def tr_slide(slide):

                        out_df = widgets.Output()
                        with out_df:
                            display(tr_dfs[slide].style.apply(highlight_prediction, axis=None))
                        try:
                            display(widgets.VBox([widgets.Image(
                                value=graphviz.Source.from_file(lt_files[slide]).pipe(format='png'), format='png'), out_df],
                                                 layout=box_layout))
                        except:
                            pass

                    tr_out = widgets.interactive_output(tr_slide, {'slide': tr_slider})

                    with out2:
                        display(widgets.VBox([widgets.Label(selected_model + ' Prediction'), pred_out,
                                              widgets.Label('Probability of Necessity'),
                                              out_pnn,
                                              widgets.Label('Probability of Sufficiency'),
                                              widgets.VBox(pss_outs, layout=box_layout),
                                              widgets.HBox([tr_slider]), tr_out], layout=box_layout))

                inspect_button.on_click(inspect_button_click)

                debug_button = widgets.Button(description="Debug " + k)

                def on_debug_clicked(b):
                    # out2.clear_output()
                    selected_model = b.description[6:]

                    single_pred = samples.loc[item_idx]
                    for sm in ['DeepER', 'DeepMatcher', 'Ditto']:
                        if sm != selected_model:
                            single_pred = single_pred.drop(sm)
                    pred_out = widgets.Output()
                    with pred_out:
                        display(pd.DataFrame(single_pred).T.style.apply(highlight_prediction, axis=None))

                    expl_data_df = pd.read_csv('data_new/' + dataset + '/' + selected_model + '/certa.csv')

                    pnn_df = pd.DataFrame(eval(expl_data_df['explanation'].iloc[item_idx]), index=[0])
                    out_pnn = widgets.Output()
                    with out_pnn:
                        display(pnn_df.style.background_gradient(cmap=cr, axis=1, low=0.1, high=0.6))

                    pss_dict = eval(expl_data_df['summary'].iloc[item_idx])
                    pss_dfs = dict()
                    for k, v in pss_dict.items():
                        attrs = k.split('/')
                        no_attrs = len(attrs)
                        if not no_attrs in pss_dfs:
                            pss_dfs[no_attrs] = {' '.join(attrs): v}
                        else:
                            cd = pss_dfs[no_attrs]
                            cd.update({' '.join(attrs): v})
                            pss_dfs[no_attrs] = cd
                    pss_outs = []
                    for k, v in pss_dfs.items():
                        pssk_out = widgets.Output()
                        with pssk_out:
                            display(pd.DataFrame(v, index=[0]).style.background_gradient(cmap=cg, axis=1, low=0.1,
                                                                                         high=0.6))
                        pss_outs.append(pssk_out)

                    if float(single_pred[selected_model]) > 0.5:
                        perturb = 'mask'
                    else:
                        perturb = 'copy'

                    saliency_graphs = []
                    topk_slider = widgets.IntSlider(value=0, min=0, max=7, step=1, description='Top K:', disabled=False,
                                                    continuous_update=False, orientation='horizontal',
                                                    readout=True, readout_format='d')
                    sgt = 'certa'
                    try:
                        sg_path = 'data_new/' + dataset + '/' + selected_model + '/' + str(
                            item_idx) + '/sg/' + sgt + '_' + perturb + '.png'
                        saliency_graph = widgets.VBox([widgets.Image(value=open(sg_path, 'rb').read(),
                                                                     format='png', width=400, height=240, ),
                                                       widgets.Label(value=sgt),
                                                       widgets.Label('Debug Data'), topk_slider], layout=box_layout)
                        saliency_graphs.append(saliency_graph)
                    except:
                        pass

                    def debug_data(top_k):
                        saliency = pnn_df.to_dict(orient='list')
                        explanation_attributes = sorted(saliency, key=saliency.get, reverse=True)[:top_k]
                        train_df = train_dfs[dataset]
                        search_rows = dict()
                        train_rows = pd.DataFrame()
                        for search_column in explanation_attributes:
                            search_value = single_pred[search_column]
                            search_rows[search_column] = search_value
                            res = train_df[train_df[search_column].str.contains(search_value)]
                            train_rows = pd.concat([train_rows, res], axis=0).drop_duplicates()

                        out_result_rows = widgets.Output()
                        out_search_rows = widgets.Output()
                        if len(search_rows) > 0:
                            with out_search_rows:
                                display(pd.DataFrame(search_rows, index=[0]))
                        if len(train_rows) > 0:
                            with out_result_rows:
                                display(train_rows.reset_index(drop=True).style.apply(highlight_prediction, axis=None))
                        display(widgets.VBox([out_search_rows, out_result_rows]))

                    data_debug_out = widgets.interactive_output(debug_data, {'top_k': topk_slider})

                    for sgt in ['mojito', 'landmark', 'shap']:
                        try:
                            sg_path = 'data_new/' + dataset + '/' + selected_model + '/' + str(
                                item_idx) + '/sg/' + sgt + '_' + perturb + '.png'
                            saliency_graph = widgets.VBox([widgets.Image(value=open(sg_path, 'rb').read(),
                                                                         format='png', width=400, height=240, ),
                                                           widgets.Label(value=sgt)], layout=box_layout)
                            saliency_graphs.append(saliency_graph)
                        except:
                            pass

                    cfm_outs = []
                    cfm_outs.append(widgets.Label('Counterfactual Examples'))
                    for cfm in ['certa', 'dice_random', 'shapc', 'limec']:
                        try:
                            cf_path = 'data_new/' + dataset + '/' + selected_model + '/' + str(
                                item_idx) + '/' + cfm + '.csv'
                            cfm_df = pd.read_csv(cf_path)
                            for c in ['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle', 'attr_count',
                                      'nomatch_score']:
                                if c in cfm_df.columns:
                                    cfm_df = cfm_df.drop([c], axis=1)
                            for rc in ['match_score', 'label']:
                                if rc in cfm_df.columns:
                                    cfm_df['prediction'] = cfm_df[rc].copy()
                                    cfm_df = cfm_df.drop([rc], axis=1)
                            cfm_out = widgets.Output()
                            with cfm_out:
                                display(cfm_df.loc[:, ~cfm_df.columns.str.contains('^Unnamed')].style.apply(
                                    highlight_prediction, axis=None))
                            cfm_outs.append(
                                widgets.VBox([widgets.Label(cf_name_dict[cfm]), cfm_out], layout=box_layout))
                        except:
                            pass
                    cf_metrics_label = widgets.Label('Counterfactual Metrics')
                    cfm_outs.append(cf_metrics_label)

                    cf_metrics_out = widgets.Output()
                    with cf_metrics_out:
                        display(pd.read_csv('data_new/' + dataset + '/' + selected_model + '/cf_metrics.csv'))
                    cfm_outs.append(cf_metrics_out)

                    saliency_tab = widgets.VBox(
                        [widgets.Label('Saliency Graphs'), widgets.HBox(saliency_graphs), data_debug_out],
                        layout=box_layout)
                    cf_tab = widgets.VBox(cfm_outs, layout=box_layout)
                    children = [saliency_tab, cf_tab]
                    debug_tab = widgets.Tab(children=children)
                    debug_tab.set_title(0, 'Saliency')
                    debug_tab.set_title(1, 'Counterfactual')
                    # debug_tab.children = children
                    # debug_tab.titles = ['Saliency', 'Counterfactual']

                    out2_data = widgets.VBox([widgets.Label(selected_model + ' Prediction'), pred_out,
                                              widgets.Label('Probability of Necessity'), out_pnn,
                                              widgets.Label('Probability of Sufficiency'),
                                              widgets.VBox(pss_outs, layout=box_layout),
                                              debug_tab], layout=box_layout)
                    with out2:
                        display(out2_data)

                debug_button.on_click(on_debug_clicked)
                saliencies_box.append(widgets.VBox([img, inspect_button, debug_button], layout=box_layout))

            cfs_df = pd.DataFrame.from_dict(cfs).T.drop(
                ['alteredAttributes', 'attr_count', 'copiedValues', 'droppedValues', 'triangle', 'nomatch_score'],
                axis=1)
            cfs_df['prediction'] = cfs_df['match_score'].copy()
            cfs_df = cfs_df.drop(['match_score'], axis=1)
            cfs_df = cfs_df.loc[:, ~cfs_df.columns.str.contains('^Unnamed')]
            out_cfs = widgets.Output()
            out_cfs.append_display_data(cfs_df.style.apply(highlight_prediction, axis=None))

            o_out = widgets.Output()
            if len(saliency_dfs) > 0:
                saliencies_df = pd.concat(saliency_dfs, axis=0, ignore_index=True)
                with o_out:
                    display(saliencies_df.style.background_gradient(cmap=cb, axis=1, low=0.1, high=0.6))
            out2.clear_output()
            out2_data = widgets.VBox([o_out, widgets.HBox(saliencies_box), out_cfs], layout=box_layout)

            with out2:
                display(out2_data)

        explain_button.on_click(on_explain_clicked)
        explain_buttons.append(explain_button)

    explain_buttons_box = widgets.HBox(explain_buttons)
    samples_out = widgets.Output()
    with samples_out:
        display(samples.style.apply(highlight_prediction, axis=None))
    display(widgets.HBox([samples_out, widgets.VBox([explain_buttons_box])], layout=box_layout))

out = widgets.interactive_output(f, {'dataset': datasets_dropdown, 'deeper': de_cb, 'dm': dm_cb, 'ditto':dt_cb, 'pred_filter': pred_filter, 'gt_filter': gt_filter})


first_box = widgets.VBox([datasets_dropdown])
second_box = widgets.HBox([sys_label, widgets.VBox([de_cb, dm_cb, dt_cb])])
third_box = widgets.VBox([gt_filter])
fourth_box = widgets.VBox([pred_filter])
top1 = widgets.HBox([first_box, second_box, third_box, fourth_box])
ui = widgets.VBox([top1, out, out2], layout=box_layout)