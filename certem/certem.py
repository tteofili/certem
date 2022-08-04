from pandas.io.common import file_exists
import os
import pandas as pd
from certa.utils import merge_sources
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np


def custom_plot(df, name):
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    df.plot(kind = 'bar',ax=ax1)
    img_path = '..data/img/'+name+'.png'
    plt.savefig(img_path)
    plt.close()
    return  [f1, img_path]

datasets = [name for name in os.listdir("..data") ]

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

def f(dataset, deeper, dm, ditto, pred_filter, gt_filter):
  samples = pd.read_csv('..data/'+dataset+'/samples.csv')
  if not deeper:
    samples = samples.drop(['DeepER'], axis=1)
  if not dm:
    samples = samples.drop(['DeepMatcher'], axis=1)
  if not ditto:
    samples = samples.drop(['Ditto'], axis=1)
  if gt_filter == 'NO-MATCH':
    samples = samples[samples['label']==0]
  if gt_filter == 'MATCH':
    samples = samples[samples['label']==1]
  if pred_filter == 'NO-MATCH':
    samples = samples[samples['DeepER']<0.5]
  if pred_filter == 'MATCH':
    samples = samples[('DeepER' in samples.columns and samples['DeepER']>0.5) | ('DeepMatcher' in samples.columns and samples['DeepMatcher']>0.5) | ('Ditto' in samples.columns and samples['Ditto']>0.5)]
  samples = samples.loc[:, ~samples.columns.str.contains('^Unnamed')]
  buttons = []
  out2.clear_output()
  for idx in range(len(samples)):
    button = widgets.Button(description="Explain Item "+str(idx))
    def on_button_clicked(b):
      out2.clear_output()
      saliencies = dict()
      cfs = dict()
      item_idx = int(b.description[-1])
      if deeper:
        saliency = pd.read_csv('..data/'+dataset+'/DeepER/certa.csv')['explanation'].iloc[item_idx]
        first_cf = pd.read_csv('..data/'+dataset+'/DeepER/'+str(idx)+'/certa.csv').iloc[0]
        saliencies['DeepER'] = saliency
        cfs['DeepER'] = first_cf
      if dm:
        saliency = pd.read_csv('..data/'+dataset+'/DeepMatcher/certa.csv')['explanation'].iloc[item_idx]
        first_cf = pd.read_csv('..data/'+dataset+'/DeepMatcher/'+str(idx)+'/certa.csv').iloc[0]
        saliencies['DeepMatcher'] = saliency
        cfs['DeepMatcher'] = first_cf
      if ditto:
        saliency = pd.read_csv('..data/'+dataset+'/Ditto/certa.csv')['explanation'].iloc[item_idx]
        first_cf = pd.read_csv('..data/'+dataset+'/Ditto/'+str(idx)+'/certa.csv').iloc[0]
        saliencies['Ditto'] = saliency
        cfs['Ditto'] = first_cf
      with out2:
        saliencies_box = []
        for k in saliencies.keys():
          saliency_df = pd.DataFrame(eval(saliencies[k]),index=[0])
          cnv, path = custom_plot(saliency_df, dataset+'_'+k+'_'+str(idx))
          img = widgets.Image(value=open(path, 'rb').read(), format='png')
          saliencies_box.append(widgets.VBox([img , widgets.Button(description='Inspect '+k)], layout=box_layout))
        cfs_df = pd.DataFrame.from_dict(cfs).T.drop(['alteredAttributes', 'attr_count', 'copiedValues', 'droppedValues', 'label', 'triangle', 'nomatch_score'], axis=1)
        cfs_df['prediction'] = cfs_df['match_score'].copy()
        cfs_df = cfs_df.drop(['match_score'], axis=1)
        cfs_df = cfs_df.loc[:, ~cfs_df.columns.str.contains('^Unnamed')]
        display(widgets.HBox(saliencies_box), cfs_df)
    button.on_click(on_button_clicked)
    buttons.append(button)
  buttons_box = widgets.HBox(buttons)
  display(samples, buttons_box, out2)

'''def f2(datasets_dropdown, de_cb, dm_cb, dt_cb, selected_items):
  for selected in selected_items:
    if selected:
      break
'''

out = widgets.interactive_output(f, {'dataset': datasets_dropdown, 'deeper': de_cb, 'dm': dm_cb, 'ditto':dt_cb, 'pred_filter': pred_filter, 'gt_filter': gt_filter})
#out2 = widgets.interactive_output(f2, {'dataset': datasets_dropdown, 'deeper': de_cb, 'dm': dm_cb, 'ditto':dt_cb, 'selected_item': selected_item})
out2 = widgets.Output()

first_box = widgets.VBox([datasets_dropdown])
second_box = widgets.HBox([sys_label, widgets.VBox([de_cb, dm_cb, dt_cb])])
third_box = widgets.VBox([gt_filter])
fourth_box = widgets.VBox([pred_filter])
top1 = widgets.HBox([first_box, second_box, third_box, fourth_box])
ui = widgets.VBox([top1, out, out2], layout=box_layout)
