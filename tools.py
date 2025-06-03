import numpy as np
import pandas as pd
import polars as pl
from datetime import timedelta
from tqdm import tqdm

def recall_at(df_solution: pl.DataFrame, df_pred: pl.DataFrame, k=40):
    assert df_pred.group_by(['cookie']).agg(pl.col('node').count())['node'].max() <41 , 'send more then 40 nodes per cookie'
    assert 'node' in df_pred.columns, 'node columns does not exist'
    assert 'cookie' in df_pred.columns, 'cookie columns does not exist'
    assert df_pred.with_columns(v = 1).group_by(['cookie','node']).agg(pl.col('v').count())['v'].max() == 1 , 'more then 1 cookie-node pair'
    assert df_pred['cookie'].dtype == pl.Int64, 'cookie must be int64'
    assert df_pred['node'].dtype == pl.Int64, 'node must be int64'
    
    return  df_solution[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']], 
        how='left',
        on = ['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum()/pl.col(
                'value'
            ).count()
        ]
    )['value'].mean()

def main(solution_path: str, prediction_path: str, stage: int):
    return recall_at(pl.read_csv(solution_path).filter(stage=stage), pl.read_csv(prediction_path))

def load_data_actions(df_clickstream, df_event):
    # PREPARE TRAIN EVAL
    EVAL_DAYS_TRESHOLD = 14
    treshhold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)
    df_train = df_clickstream.filter(df_clickstream['event_date']<= treshhold)
    df_eval = df_clickstream.filter(df_clickstream['event_date']> treshhold)[['cookie', 'node', 'event']]
    df_eval = df_eval.join(df_train, on=['cookie', 'node'], how='anti')

    df_eval = df_eval.filter(
        pl.col('event').is_in(
            df_event.filter(pl.col('is_contact')==1)['event'].unique()
        )
    )
    df_eval = df_eval.filter(
            pl.col('cookie').is_in(df_train['cookie'].unique())
        ).filter(
            pl.col('node').is_in(df_train['node'].unique())
        )
    df_eval = df_eval.unique(['cookie', 'node'])

    return df_train, df_eval


def generate_lightfm_recs_mapper(model, item_ids, known_items, 
                                 user_features, item_features, N, 
                                 user_mapping, item_inv_mapping, 
                                 num_threads=1):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(user_id, item_ids, user_features=user_features, 
                             item_features=item_features, num_threads=num_threads)
        
        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]
        
        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]
    return _recs_mapper



def predict_user_top_items_lightfm(u_raw, lfm_model, lightfm_mapping, known_items, n_items, top_N):
    """Compute top-N for a single user raw ID."""
    u = lightfm_mapping['users_mapping'].get(u_raw, None)
    if u is None:
        return u_raw, []

    # 1) score all items
    scores = lfm_model.predict(u, np.arange(n_items), num_threads=1)
    # 2) mask seen
    scores[known_items[u]] = -np.inf
    # 3) top-N
    top_idxs = np.argpartition(-scores, top_N)[:top_N]
    sorted_idxs = top_idxs[np.argsort(-scores[top_idxs])]
    top_items = [lightfm_mapping['items_inv_mapping'][i] for i in sorted_idxs]
    return u_raw, top_items  
