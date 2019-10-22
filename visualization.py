from sql import read_table, db_engine
from helpers import filter_df_by_np_id, np_country, np_ppty, check_dates, decade_from_year_df, group_and_count

import seaborn as sns
import pandas as pd
from typing import Iterable


def plot_issues_time_id(time_gran: str, start_date: int, end_date: int, np_ids: Iterable = None,
                        country: str = None, df: pd.core.frame.DataFrame = None, batch_size: int = None,
                        ppty: str = None, ppty_value: str = None) -> None:
    issues_df = df

    # load data from SQL if needed
    if df is None:
        issues_df = read_table('impresso.issues', db_engine())

    # select specific np ids
    if np_ids is not None and len(np_ids) > 0:
        issues_df = filter_df_by_np_id(issues_df, np_ids)

    # select specific country
    if country is not None:
        countries = np_country(country)
        issues_df = filter_df_by_np_id(issues_df, countries)

    # select specific property
    if not (ppty is None or ppty_value is None):
        properties = np_ppty(ppty, ppty_value, db_engine())
        issues_df = filter_df_by_np_id(issues_df, properties)

    # check date values
    assert check_dates(start_date, end_date), 'Problem with start and end dates.'

    # select dates
    issues_df = issues_df.loc[(start_date <= issues_df['year']) & (issues_df['year'] <= end_date)]

    # take final list of np ids
    np_ids_filtered = issues_df.newspaper_id.unique()

    # check time_granularity is either 'year' or 'decade'
    assert (time_gran == 'decade' or time_gran == 'year'), "Time granularity must be either 'decade' or 'year'."

    # create it decade column if doesn't exist yet
    if time_gran == 'decade' and 'decade' not in df.columns:
        issues_df = decade_from_year_df(issues_df)

        # group and count for the histogram
    count_df, _, _ = group_and_count(issues_df, ['newspaper_id', time_gran], 'id', print_=False)

    # if batch_size not specified : plot all newspapers on the same figure
    if batch_size is None:
        sns.catplot(x=time_gran, y="count", hue="newspaper_id", kind="bar", data=count_df, height=5, aspect=2);

    # else plot by batches (no intelligent batching is done)
    else:
        assert (0 < batch_size and batch_size < 20), "Batch size must be between 1 and 19."
        catplot_by_batch_np(count_df, np_ids_filtered, time_gran, batch_size)


def catplot_by_batch_np(df, np_list, time_granularity='decade', max_cat=5):
    if len(df.newspaper_id.unique()) > max_cat:
        np_batch = [np_list[x:x + max_cat] for x in range(0, len(np_list), max_cat)]
    else:
        np_batch = [np_list]

    # Plot by batches
    for i, b in enumerate(np_batch):
        batch = filter_df_by_np_id(df, b)
        sns.catplot(x=time_granularity, y="count", hue="newspaper_id", kind="bar", data=batch, height=5, aspect=2)
