import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback

def alter_data(explanations, l_tuple, r_tuple, predict_fn, agg=True, lprefix='ltable_', rprefix='rtable_',
        num_masked: int = 3, perturb: str = 'mask', mask_token: str = ''):
    data = pd.DataFrame()
    lt = l_tuple.copy()
    rt = r_tuple.copy()
    row = get_row(lt, rt)
    orig = predict_fn(row)[['nomatch_score', 'match_score']].values[0][1]
    margin = 0.5
    for explanation in explanations:
        saliency = explanation.copy()
        exp_type = saliency.pop('type')
        scores_d = []
        scores_c = []
        for tk in np.arange(num_masked):
            # get top k important attributes
            if not agg and tk >= len(saliency):
                break
            if exp_type == 'certa':
                if agg:
                    explanation_attributes = sorted(saliency, key=saliency.get, reverse=True)[:tk]
                else:
                    explanation_attributes = [sorted(saliency, key=saliency.get, reverse=True)[tk]]
            elif orig < 0.5:
                saliency = {k:v for k,v in saliency.items() }
                if agg:
                    explanation_attributes = sorted(saliency, key=saliency.get)[:tk]
                else:
                    explanation_attributes = [sorted(saliency, key=saliency.get)[tk]]
            else:
                saliency = {k:v for k,v in saliency.items() }
                if agg:
                    explanation_attributes = sorted(saliency, key=saliency.get, reverse=True)[:tk]
                else:
                    explanation_attributes = [sorted(saliency, key=saliency.get, reverse=True)[tk]]
            # alter those attributes
            if len(explanation_attributes) > 0:
              if perturb == 'mask':
                try:
                    lt = l_tuple.copy()
                    rt = r_tuple.copy()
                    modified_row = get_row(lt, rt)
                    for e in explanation_attributes:
                        modified_row[e] = mask_token
                    modified_tuple_prediction = predict_fn(modified_row)[['nomatch_score', 'match_score']].values[0]
                    score_drop = modified_tuple_prediction[1]
                    scores_d.append(score_drop)
                except Exception as e:
                    print(traceback.format_exc())
              elif perturb == 'copy':
                try:
                    lt = l_tuple.copy()
                    rt = r_tuple.copy()
                    modified_row = get_row(lt, rt)
                    for e in explanation_attributes:
                        if e.startswith(lprefix):
                            new_e = e.replace(lprefix, rprefix)
                        else:
                            new_e = e.replace(rprefix, lprefix)
                        modified_row[e] = modified_row[new_e]
                    modified_tuple_prediction = predict_fn(modified_row)[['nomatch_score', 'match_score']].values[0]
                    score_copy = modified_tuple_prediction[1]
                    scores_d.append(score_copy)
                except Exception as e:
                    print(traceback.format_exc())
        data[exp_type]= pd.Series(scores_d)

    data['prediction'] = orig
    data['margin'] = margin
    return data


def saliency_graph(saliency_df: pd.DataFrame, l_tuple, r_tuple, predict_fn, etype='certa', nm=9, color='skyblue',
                   so=False, mapping_dict=None):
    exp = saliency_df.copy().to_dict(orient='list')
    exp['type'] = etype
    single = alter_data([exp], l_tuple, r_tuple, predict_fn, num_masked=nm, agg=False)[etype]
    aggr = alter_data([exp], l_tuple, r_tuple, predict_fn, num_masked=nm)[etype]
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    my_cmap = plt.get_cmap("Blues")
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=60, fontsize=20)
    if mapping_dict is not None:
        saliency_df = saliency_df.rename(columns=mapping_dict)
    saliency_sorted_df = saliency_df.sort_values(saliency_df.last_valid_index(), axis=1, ascending=so)

    x = saliency_sorted_df.columns
    y = saliency_sorted_df.values[0]

    if etype == 'certa':
        plt.ylim(0.0, 1.0)
    else:
        plt.ylim(min(y) * 1.1, 1.0)
    plt.bar(x=x, height=y, label='Saliency', color=color)
    axes2 = plt.twinx()
    y1 = aggr
    y2 = single
    axes2.plot(x[:len(y1)], y1, '^g-', label='Aggregate', linewidth=5.0)
    axes2.plot(x[:len(y2)], y2, '^r-', label='Single', linewidth=5.0)

    plt.show()
    aggr = np.array(y1)
    sing = np.array(y2)

    if len(aggr) < len(saliency_df.columns):
        pl = len(saliency_df.columns) - len(aggr)
        aggr = np.pad(aggr, (0, pl))
    if len(sing) < len(saliency_df.columns):
        pl = len(saliency_df.columns) - len(sing)
        sing = np.pad(sing, (0, pl))

    saliency_sorted_df.loc[1] = aggr
    saliency_sorted_df.loc[2] = sing
    saliency_sorted_df['result'] = ['saliency', 'aggregate', 'single']

    return saliency_sorted_df