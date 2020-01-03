from impresso_stats.sql import read_table, db_engine
from impresso_stats.helpers import filter_df_by_np_id, np_country, np_ppty, check_dates, decade_from_year_df, group_and_count, filter_df

import numpy as np
import seaborn as sns
import pandas as pd
import dask
import dask.dataframe
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from typing import Iterable

from  matplotlib.ticker import FuncFormatter

# ---------------------- CONSTANTS ---------------------- #

LABEL_THRESHOLD_ROTATION = 30
LABEL_THRESHOLD_SELECT = 100
NUM_BARS_THRESHOLD = 350
MAX_CAT = 5
COLOR = 'salmon'
HEIGHT=5
FIG_HEIGHT=20
ASPECT=3
FIG_ASPECT=5

# ----------------------- ISSUES ----------------------- #
#plot_freq_issues
def plt_freq_issues_time(time_gran: str,
                        start_date: int = None,
                        end_date: int = None,
                        np_ids: Iterable = None,
                        country: str = None,
                        df: pd.core.frame.DataFrame = None,
                        ppty: str = None,
                        ppty_value: str = None,
                        batch_size: int = None) -> pd.core.frame.DataFrame:
    """
    General plotting function for issues frequency analysis (histogram).
    :param time_gran: granularity in time, either 'year' or 'decade'
    :param start_date: earliest date for analysis
    :param end_date: latests date for analysis
    :param np_ids: list (or pandas series) of newspapers ids on which to focus
    :param country: selected country code
    :param df: original data frame on which to build the histogram
    :param ppty: selected property on which to filter newspapers
    :param ppty_value: property value corresponding to the selected property
    :param batch_size: maximum number of newspapers represented on a single plot
    :return: Aggregated dataframe.
    """
    issues_df = df

    # load data from SQL if needed
    if df is None:
        issues_df = read_table('impresso.issues', db_engine())
    
    issues_df, np_ids_filtered = filter_df(issues_df, start_date, end_date, np_ids, country, ppty, ppty_value)

    # check time_granularity is either 'year' or 'decade'
    assert (time_gran == 'decade' or time_gran == 'year'), "Time granularity must be either 'decade' or 'year'."

    # create it decade column if doesn't exist yet
    if time_gran == 'decade' and 'decade' not in issues_df.columns:
        issues_df = decade_from_year_df(issues_df)

    # group and count for the histogram
    count_df = group_and_count(issues_df, 'newspaper_id', time_gran, 'id', print_=False)

    # if batch_size not specified : plot all newspapers on the same figure
    if batch_size is None:
        g = sns.catplot(x=time_gran, y="count", hue="newspaper_id", kind="bar", data=count_df, height=HEIGHT, aspect=ASPECT)

        plt_settings_FacetGrid(g, count_df, [time_gran, 'newspaper'], \
                               facet='freq', level='issues', hide_xtitle=True, log_y=False)

    # else plot by batches (no intelligent batching is done)
    else:
        assert (0 < batch_size and batch_size < 20), "Batch size must be between 1 and 19."
        catplot_by_batch_np(count_df, np_ids_filtered, time_gran, 'count', 'newspaper_id', batch_size,
                            display_x_label=False, y_label='Number of issues',
                            title='Issue frequency per newspaper, through time.')
        
    return count_df

#plot_freq_issues_batch
def catplot_by_batch_np(df: pd.core.frame.DataFrame,
                        np_list: Iterable,
                        xp: str,
                        yp: str,
                        huep: str,
                        max_cat: int = MAX_CAT,
                        rotation: int = None,
                        display_x_label: bool = True,
                        display_y_label: bool = True,
                        x_label: str = None,
                        y_label: str = None,
                        title: str = None) -> None:
    """
    Helper function for plotting by batches
    :param df: data frame to plot (no processing done on it)
    :param np_list: list of newspaper ids
    :param xp: x axis variable
    :param yp: y axis variable
    :param huep: hue variable
    :param max_cat: maximum number of categories represented on a single plot
    :param rotation: rotation for the labels on the x axis (optional)
    :param display_x_label: set to False if x axis title should be hidden
    :param display_y_label: set to False if y axis title should be hidden
    :param x_label: specify the title you want to set for the x axis
    :param y_label: specify the title you want to set for the y axis
    :param title: specify plot title
    :return: Nothing, only plots.
    """
    if len(df.newspaper_id.unique()) > max_cat:
        np_batch = [np_list[x:x + max_cat] for x in range(0, len(np_list), max_cat)]
    else:
        np_batch = [np_list]
    
    # Plot by batches
    for i, b in enumerate(np_batch):
        batch = filter_df_by_np_id(df, b)

        g = sns.catplot(x=xp, y=yp, hue=huep, kind="bar", data=batch, height=HEIGHT, aspect=ASPECT)

        plt_settings_FacetGrid(g, batch, [xp, 'newspaper'], facet='freq', level='issues', hide_xtitle=True, log_y=False)


# ----------------------- LICENCES ----------------------- #

def plot_licences(facet: str = 'newspapers',
                  df: pd.core.frame.DataFrame = None,
                  start_date: int = None,
                  end_date: int = None,
                  np_ids: Iterable = None,
                  country: str = None,
                  batch_size: int = None,
                  ppty: str = None,
                  ppty_value: str = None):
    """
    Plot frequency of the access right type per newspaper or per decade, on the given df.
    :param facet: either 'newspapers' or 'time' depending on the dimension one wants to explore
    :param df: pandas data frame on which to plot the licence frequencies after applying filters.
    :param start_date: earliest date for analysis
    :param end_date: latests date for analysis
    :param np_ids: list (or pandas series) of newspapers ids on which to focus
    :param country: selected country code
    :param batch_size: maximum number of newspapers represented on a single plot
    :param ppty: selected property on which to filter newspapers
    :param ppty_value: property value corresponding to the selected property
    :return: Nothing. Plots the bar plot(s) of licence frequency.
    """
    result_df = df.copy()

    # load data from SQL if needed
    if df is None:
        result_df = read_table('impresso.issues', db_engine())

    # apply all filters to get specific rows of df
    result_df, np_ids = filter_df(result_df, start_date=start_date, end_date=end_date, np_ids=np_ids,
                                  country=country, ppty=ppty, ppty_value=ppty_value)

    result_df = decade_from_year_df(result_df)

    if facet == 'newspapers':
        count_df = group_and_count(result_df, ['newspaper_id', 'access_rights'], 'id')
        plot_licences_np(count_df, np_ids, batch_size)

    elif facet == 'time':
        count_df = group_and_count(result_df, ['decade', 'access_rights'], 'id')
        plot_licences_time(count_df, np_ids, batch_size)
    else:
        print("Nothing can be done : facet parameter should be either 'newspapers', or 'time'.")


def plot_licences_time(count_df: pd.core.frame.DataFrame,
                       np_ids: Iterable,
                       batch_size: int = None) -> None:
    """
    Plots the number of issues per access right type per decade in the given df.
    Helper function for plot_licences.
    :param count_df: pandas data frame with columns 'count', 'decade', 'access_rights'
    :param np_ids: list of unique np ids in the df (useful for batch plotting)
    :param batch_size: max number of decades per plot
    :return: Nothing. Plots.
    """
    # if batch_size not specified : plot all newspapers on the same figure
    if batch_size is None:
        g = sns.catplot(x="decade", y="count", hue="access_rights", kind="bar", data=count_df, height=5, aspect=2)
        display_setup(g, rotation=30, display_x_label=False, y_label='Number of issues',
                      title='Issue frequency per access right type.')

    # else plot by batches (no intelligent batching is done)
    else:
        assert (0 < batch_size and batch_size < 20), "Batch size must be between 1 and 19."
        catplot_by_batch_np(count_df, np_ids, 'decade', 'count', 'access_rights', batch_size,
                            rotation=30, display_x_label=False, y_label='Number of issues',
                            title='Issue frequency per access right type.')


def plot_licences_np(count_df: pd.core.frame.DataFrame,
                     np_ids: Iterable,
                     batch_size: int = None) -> None:
    """
    Plots the number of issues par access right type per newspapers in the given df.
    Helper function for plot_licences.
    :param count_df: pandas data frame with columns 'count', 'newspapers_id', 'access_rights'
    :param np_ids: list of unique np ids in the df (useful for batch plotting)
    :param batch_size: max number of newspaper per plot
    :return: Nothing. Plots.
    """
    # if batch_size not specified : plot all newspapers on the same figure
    if batch_size is None:
        g = sns.catplot(x="newspaper_id", y="count", hue="access_rights", kind="bar", data=count_df, height=5, aspect=2)
        display_setup(g, rotation=30, y_label='Number of issues', title='Issue frequency per access right type.')

    # else plot by batches (no intelligent batching is done)
    else:
        assert (0 < batch_size and batch_size < 20), "Batch size must be between 1 and 19."
        catplot_by_batch_np(count_df, np_ids, 'newspaper_id', 'count', 'access_rights', batch_size,
                            y_label='Number of issues', rotation=30, title='Issue frequency per access right type.')
        
        
        
# ----------------------- CONTENT ITEMS ----------------------- #

def plt_freq_ci_filter(df: dask.dataframe.core.DataFrame, 
                grouping_col: list, 
                asc: bool =False, 
                hide_xtitle: bool =False, 
                log_y: bool =False, 
                types: list = None,
                start_date: int = None,
                end_date: int = None,
                np_ids: Iterable = None,
                country: str = None) -> pd.core.frame.DataFrame:
    """
    Similar function as plt_freq_ci, on which you can add a filter at the level of newspapers.
    Displays a bar plot of the number of content items aggregated at one or two dimension in given df, 
    after filtering according to parameters.
    :param df: dask data frame with column 'id' and column(s) in grouping_col
    :param grouping_col: list of column(s) on which to aggregate the count (typically : 'year', 'type', 
       'newspaper', or 'decade'). The list can contain one or two column names. If length is 2,
       the first value is used as x axis and the second one as categorical value. 
       It is not recommanded to use a time column (year or decade) as second column name. 
       Sorting will be done in the same order.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
                        (typically useful if x axis are years) (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :param types: list of content-item types ('ar', 'w', ...). The list can have length 1 to number of types.
    :param start_date: earliest date on which to filter
    :param end_date: latests date on which to filter
    :param np_ids: list (or pandas series) of newspapers ids to filter (i.e. to keep)
    :param country: selected country code (typically 'CH' or 'LU')
    :return: the aggregated df used for plotting
    """
    
    df_filtered = df
    
    # Apply filters at the level of content-items (on types)
    if types is not None:
        df_filtered = df_filtered[df_filtered.type.isin(types)]
        
    # Apply filters at the level of newspapers
    df_filtered, _ = filter_df(df_filtered, start_date, end_date, np_ids, country)
    
    # Plot ci frequency based on filtered df
    return plt_freq_ci(df_filtered, grouping_col, asc, hide_xtitle, log_y)


def plt_avg_tl_filter(df: dask.dataframe.core.DataFrame, 
                grouping_col: list, 
                asc: bool =False, 
                hide_xtitle: bool =False, 
                log_y: bool =False, 
                types: list = None,
                start_date: int = None,
                end_date: int = None,
                np_ids: Iterable = None,
                country: str = None) -> pd.core.frame.DataFrame:
    """
    Similar function as plt_avg_tl, on which you can add a filter at the level of newspapers.
    Displays a bar plot of the number of content items aggregated at one or two dimension in given df, 
    after filtering according to parameters.
    :param df: dask data frame with column 'id' and column(s) in grouping_col
    :param grouping_col: list of column(s) on which to aggregate the average (typically : 'year', 'type', 
       'newspaper', or 'decade'). The list can contain one or two column names. If length is 2,
       the first value is used as x axis and the second one as categorical value. 
       It is not recommanded to use a time column (year or decade) as second column name. 
       Sorting will be done in the same order.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
                        (typically useful if x axis are years) (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :param types: list of content-item types ('ar', 'w', ...). The list can have length 1 to number of types.
    :param start_date: earliest date on which to filter
    :param end_date: latests date on which to filter
    :param np_ids: list (or pandas series) of newspapers ids to filter (i.e. to keep)
    :param country: selected country code (typically 'CH' or 'LU')
    :return: the aggregated df used for plotting
    """
    
    df_filtered = df
    
    # Apply filters at the level of content-items (on types)
    if types is not None:
        df_filtered = df_filtered[df_filtered.type.isin(types)]
        
    # Apply filters at the level of newspapers
    df_filtered, _ = filter_df(df_filtered, start_date, end_date, np_ids, country)
    
    # Plot ci frequency based on filtered df
    return plt_avg_tl(df_filtered, grouping_col, asc, hide_xtitle, log_y)
    

def plt_freq_ci(df: dask.dataframe.core.DataFrame, 
                grouping_col: list, 
                asc: bool =False, 
                hide_xtitle: bool =False, 
                log_y: bool =False) -> pd.core.frame.DataFrame:
    """
    Displays a bar plot of the number of content items aggregated at one or two dimension in given df.
    Helper function for plt_freq_ci_filter.
    :param df: dask data frame with column 'id' and column(s) in grouping_col
    :param grouping_col: list of column(s) on which to aggregate the count (typically : 'year', 'type', 
       'newspaper', or 'decade'). The list can contain one or two column names. If length is 2,
       the first value is used as x axis and the second one as categorical value. 
       It is not recommanded to use a time column (year or decade) as second column name. 
       Sorting will be done in the same order.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
                        (typically useful if x axis are years) (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    n = len(grouping_col)
    
    if n==1:
        return plt_freq_ci_1d(df, grouping_col[0], asc, hide_xtitle, log_y)
    
    elif n==2:
        return plt_freq_ci_2d(df, grouping_col, hide_xtitle, log_y)
        
    else: 
        raise ValueError("grouping_col parameter must be a list of length 1 or 2.")
        
        
def plt_avg_tl(df: dask.dataframe.core.DataFrame, 
                grouping_col: list, 
                asc: bool =False, 
                hide_xtitle: bool =False, 
                log_y: bool =False) -> pd.core.frame.DataFrame:
    """
    Displays a bar plot of the average title length of content items, aggregated at one or two dimension in given df.
    Helper function for plt_avg_tl_filter.
    :param df: dask data frame with column 'title_length' and column(s) in grouping_col
    :param grouping_col: list of column(s) on which to aggregate the average (typically : 'year', 'type', 
       'newspaper', or 'decade'). The list can contain one or two column names. If length is 2,
       the first value is used as x axis and the second one as categorical value. 
       It is not recommanded to use a time column (year or decade) as second column name. 
       Sorting will be done in the same order.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
       (typically useful if x axis are years) (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    n = len(grouping_col)
    
    if n==1:
        return plt_avg_tl_1d(df, grouping_col[0], asc, hide_xtitle, log_y)
    
    elif n==2:
        return plt_avg_tl_2d(df, grouping_col, hide_xtitle, log_y)
        
    else: 
        raise ValueError("grouping_col parameter must be a list of length 1 or 2.")
        

def plt_freq_ci_1d(df: dask.dataframe.core.DataFrame, 
                   grouping_col: str, 
                   asc: bool =False, 
                   hide_xtitle: bool =False, 
                   log_y: bool =False) -> pd.core.frame.DataFrame:
    """
    Displays a bar plot of the number of content items aggregated at one dimension in given df.
    :param df: dask data frame with columns 'id', grouping_col, or already aggregated with count column 'freq'
    :param grouping_col: column on which to aggregate the count (typically : 'year', 'type', 
                        'newspaper', or 'decade' if column is added by calling decade_from_year_df 
                        from helpers for example.
    :param asc: if set to True, plots bar in ascending order (default is descending order)
    :param hide_xtitle: if set to True, doesn't display title for x axis 
                        (typically useful if x axis are years) (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    
    return plt_generic_1d(df, grouping_col, 'freq', asc, hide_xtitle, log_y)


def plt_avg_tl_1d(df: dask.dataframe.core.DataFrame, 
                   grouping_col: str, 
                   asc: bool =False, 
                   hide_xtitle: bool =False, 
                   log_y: bool =False) -> pd.core.frame.DataFrame:
    """
    Displays a bar plot of the average title length aggregated at one dimension in given df.
    :param df: dask data frame with columns 'id', grouping_col, or already aggregated with count column 'avg'
    :param grouping_col: column on which to aggregate the count (typically : 'year', 'type', 
        'newspaper', or 'decade' if column is added by calling decade_from_year_df from helpers for example.)
    :param asc: if set to True, plots bar in ascending order (default is descending order)
    :param hide_xtitle: if set to True, doesn't display title for x axis(typically useful if x axis are years) 
        (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    
    return plt_generic_1d(df, grouping_col, 'avg', asc, hide_xtitle, log_y)


def plt_generic_1d(df: dask.dataframe.core.DataFrame, 
                   grouping_col: str, 
                   facet: str ='freq',
                   asc: bool =False, 
                   hide_xtitle: bool =False, 
                   log_y: bool =False) -> pd.core.frame.DataFrame:
    """
    Displays a bar plot either of the number of content items or of the average title_length (depending 
    on parameter 'facet'), aggregated at one dimension in given df.
    :param df: dask data frame with columns 'id' of 'title_length' (depending on parameter facet), grouping_col, 
        or already aggregated with column 'freq' or 'avg' (equal to value of parameter 'facet')
    :param grouping_col: column on which to aggregate (typically : 'year', 'type', 'newspaper', or 'decade' 
        if column is added by calling decade_from_year_df from helpers for example.)
    :param freq_avg: if set to 'freq', counts the number of content item ID, if set to 'avg', computes the average 
        title_length (further development may add other facets). Disclaimer : for the average facet, it doesn't 
        consider NaN values.
    :param asc: if set to True, plots bar in ascending order (default is descending order)
    :param hide_xtitle: if set to True, doesn't display title for x axis 
        (typically useful if x axis are years) (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    
    assert facet in ['freq', 'avg'], 'Parameter facet should be a string of value either "freq" or "avg"'
    
    # Perfom the group by and count operation and convert to pandas df
    if facet in df.columns:
        agg_df = df
    else :
        if facet=='freq':
            agg_df = df.groupby(grouping_col).id.count().compute().reset_index(name=facet)

        elif facet=='avg':
            agg_df = df.groupby(grouping_col).title_length.mean().compute().reset_index(name=facet)
    
        # Sort by avg descending (default), or other if specified (time / ascending)
        if grouping_col == 'year' or grouping_col=='decade' :
            agg_df.sort_values(by=grouping_col, inplace=True, ascending=True)

            # Fill potential gaps in time
            time_step = 1 if grouping_col == 'year' else 10
            idx = np.arange(agg_df[grouping_col].min(), agg_df[grouping_col].max()+time_step, step=time_step)
            agg_df = agg_df.set_index(grouping_col).reindex(idx).reset_index().fillna({facet:0}).fillna(method='ffill')

        else:
            agg_df.sort_values(by=facet, inplace=True, ascending=asc)
    
    num_xlabels = len(agg_df[grouping_col])
    
    # Plot figure
    plt.figure(figsize=(FIG_HEIGHT,FIG_ASPECT))

    g = sns.barplot(x=grouping_col, y=facet, data=agg_df, color=COLOR);

    plt_settings_Axes(g, agg_df, grouping_col, facet, hide_xtitle, log_y)

    return agg_df


def plt_settings_Axes(g: matplotlib.axes.SubplotBase, 
                      count_df: dask.dataframe.core.DataFrame,
                      grouping_col: list,
                      facet: str,
                      hide_xtitle: bool,
                      log_y: bool) -> None:
    '''
    Helper function for plot settings, used in function plt_generic_1d.
    Modifies parameter g for setting titles, axis, formats, etc.
    :param g: matplotlib Axes which will be modified directly in the function.
    :param count_df: pandas dataframe which is plotted.
    :param grouping_col: column for x axis.
    :param facet: paramter passed by function plt_generic_1d, giving information 
    on whether we are plotting and average or a count.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
    :param log_y: if set to True, plot in logarithmic scale (for y axis)
    :return: nothing. changes are done directly by modifiying parameter g.
    '''
    
    assert facet in ['freq', 'avg'], 'Parameter facet should be a string of value either "freq" or "avg"'

    # SET X AXIS
    # Labels
    # no particular setup if number of labels is less than the first threshold
    num_xlabels = len(count_df[grouping_col])

    if num_xlabels < LABEL_THRESHOLD_ROTATION :
        g.set_xticklabels(count_df[grouping_col])
        
    # rotate by 90 degrees if number of labels is between first and second threshold
    elif num_xlabels <  LABEL_THRESHOLD_SELECT :
        g.set_xticklabels(count_df[grouping_col], rotation=90)
       
    # display only certain labels (and rotate by 45 degrees) if number of labels is higher
    else :        
        number_of_steps = num_xlabels/50
        
        l = np.arange(0, num_xlabels, number_of_steps)
        
        pos = (l / num_xlabels) * (max(g.get_xticks())-min(g.get_xticks()))
        g.set_xticks(pos);
        g.set_xticklabels(count_df[grouping_col].iloc[l], rotation=45);
    
    # Title
    # option to remove the x axis title (when its obvious, e.g. for the years)
    if hide_xtitle:
        g.set_xlabel('');
    else:
        g.set_xlabel(grouping_col);
    
    # SET Y AXIS
    # log scale option
    if log_y :
        g.set_yscale("log")
        if facet=='freq':
            g.set_ylabel('# content items (log scale)')
        elif facet=='avg':
            g.set_ylabel('title length (log scale)')
    
    else :       
        if facet=='freq':
            g.set_ylabel('# content items')
        elif facet=='avg':
            g.set_ylabel('title length')

        
    # Labels
    ylabels = ['{:,.0f}'.format(y) for y in g.get_yticks()]
    g.set_yticklabels(ylabels)
    
    # Plot Title
    if facet=='freq':
        g.set_title('Number of content items by %s' % grouping_col)
    elif facet=='avg':
        g.set_title('Average title length of content items by %s' % grouping_col)

    
def plt_freq_ci_2d(df: dask.dataframe.core.DataFrame,
                   grouping_col: list,
                   hide_xtitle: bool =False,
                   log_y: bool =False) -> pd.core.frame.DataFrame:
    
    """
    Displays a categorical plot of the number of content items aggregated at two dimension in given df.
    :param df: dask data frame with column 'id' and column(s) in grouping_col, or with column 'freq' 
        if already aggregated
    :param grouping_col: list of column(s) on which to aggregate the count (typically : 'year', 'type', 
       'newspaper', or 'decade'). The list must contain two column names :the first value is used as x axis 
       and the second one as categorical value. 
       It is not recommanded to use a time column (year or decade) as second column name. 
       Sorting will be done in the same order.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
                        (typically useful if x axis are years) (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    
    return plt_generic_2d(df, grouping_col, 'freq', hide_xtitle, log_y)


def plt_avg_tl_2d(df: dask.dataframe.core.DataFrame,
                   grouping_col: list,
                   hide_xtitle: bool =False,
                   log_y: bool =False) -> pd.core.frame.DataFrame:
    
    """
    Displays a categorical plot of the average title length of content items, aggregated at two dimension in given df.
    :param df: dask data frame with column 'title_length' and column(s) in grouping_col, or with column 'avg' 
        if already aggregated
    :param grouping_col: list of column(s) on which to aggregate the average (typically : 'year', 'type', 
       'newspaper', or 'decade'). The list must contain two column names :the first value is used as x axis 
       and the second one as categorical value. 
       It is not recommanded to use a time column (year or decade) as second column name. 
       Sorting will be done in the same order.
    :param hide_xtitle: if set to True, doesn't display title for x axis (typically useful if x axis are years) 
        (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    
    return plt_generic_2d(df, grouping_col, 'avg', hide_xtitle, log_y)


def plt_generic_2d(df: dask.dataframe.core.DataFrame,
                   grouping_col: list,
                   facet: str,
                   hide_xtitle: bool =False,
                   log_y: bool =False) -> pd.core.frame.DataFrame:
    
    """
    Displays a categorical plot of the number of content items or the average title length of content items,
        depending on parameter 'facet', aggregated at two dimension in given df.
    :param df: dask data frame with column 'id' or 'title_length' (depending on parameter 'facet') 
        and column(s) in grouping_col, or with column facet (i.e. 'freq' or 'avg'), if already aggregated.
    :param grouping_col: list of column(s) on which to aggregate (typically : 'year', 'type', 
       'newspaper', or 'decade'). The list must contain two column names :the first value is used as x axis 
       and the second one as categorical value. 
       It is not recommanded to use a time column (year or decade) as second column name. 
       Sorting will be done in the same order.
    :param hide_xtitle: if set to True, doesn't display title for x axis (typically useful if x axis are years) 
        (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    
    assert facet in ['freq', 'avg'], 'Parameter facet should be a string of value either "freq" or "avg"'

    assert len(grouping_col)==2, "grouping_col parameter must be a list of length 2."
    assert not (grouping_col[1]=='year' or grouping_col[1]=='decade'), "Time cannot be used as categorical variable."
    
    # Aggregation operation: groupby and mean or count
    if facet in df.columns:
        agg_df = df.copy()
    else :
        if facet=='freq':
            agg_df = df.groupby(grouping_col).id.count().compute().rename(facet)#.reset_index(name=facet)

        elif facet=='avg':
            agg_df = df.groupby(grouping_col).title_length.mean().compute().rename(facet)#.reset_index(name=facet)
    

        # Fill potential gaps in time if aggregating at time dimensionn
    
        my_dict = {}
        if grouping_col[0]=='year' or grouping_col[0]=='decade' :

            time_step = 1 if grouping_col[0] == 'year' else 10
            max_date = agg_df.reset_index()[grouping_col[0]].max()
            min_date = agg_df.reset_index()[grouping_col[0]].min()
            idx = np.arange(min_date, max_date+ time_step, time_step)
    
            for idx1 in agg_df.index.get_level_values(1).unique():
                sub_df = agg_df.xs(idx1,level=grouping_col[1]).reset_index()
                
                sub_df = sub_df.set_index(grouping_col[0]).reindex(idx).reset_index().fillna({facet:0}).fillna(method='ffill')

                my_dict[idx1] = sub_df

            agg_df = pd.concat(my_dict.values(), keys=my_dict.keys()).reset_index()

            agg_df = agg_df.drop(['level_1'], axis=1).rename(columns={'level_0': grouping_col[1]})
            
        else :
            agg_df = agg_df.reset_index()
    
    
    # Check if df is not too big for plotting
    if len(agg_df) > NUM_BARS_THRESHOLD :
        raise ValueError("The total number of bars to plot exceeds limit: "+ str(NUM_BARS_THRESHOLD) + \
                         "(you have: "+ str(len(agg_df)) +"). Not able to plot figure. \
                         Please reduce by filtering the dataframe on some features.") 
    
    # Sort by count descending (default), or other if specified (time / ascending)
    aggr_dim = grouping_col[0]
    cat_dim = grouping_col[1]
    
    ascending_0 = grouping_col[0]=='year' or grouping_col[0]=='decade'
        
    agg_df.sort_values(by=[grouping_col[0], facet], inplace=True, ascending=[ascending_0, False])
    
    # Plot
    g = sns.catplot(x=grouping_col[0], y=facet, data=agg_df, \
                hue=grouping_col[1], kind='bar', height=HEIGHT, aspect=ASPECT)
    
    # Plot settings
    plt_settings_FacetGrid(g, agg_df, grouping_col, facet, 'content items', hide_xtitle, log_y)
    
    return agg_df


def plt_settings_FacetGrid(g: sns.axisgrid.FacetGrid, 
                           count_df: dask.dataframe.core.DataFrame, 
                           grouping_col: list,
                           facet: str,
                           level: str,
                           hide_xtitle: bool, 
                           log_y: bool) -> None:
    '''
    Helper function for plot settings, used in function plt_generic_2d.
    Modifies parameter g for setting titles, axis, formats, etc.
    :param g: seaborn FacetGrid which will be modified directly in the function.
    :param count_df: pandas dataframe which is plotted.
    :param grouping_col: list of columns for category and x axis.
    :param facet: parameter passed by function plt_generic_2d, giving information 
        on whether we are plotting and average or a count.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
    :param log_y: if set to True, plot in logarithmic scale (for y axis)
    :return: nothing. changes are done directly by modifiying parameter g.
    '''
    
    assert facet in ['freq', 'avg'], 'Parameter facet should be a string of value either "freq" or "avg"'
    assert level in ['issues', 'content items'], 'Parameter level should be a string of value either \
    "issues" or "content items"'
    assert not (level=='issues' and facet=='avg'), 'You cannot compute an average at the issue level.\
    Please check your parameters are matching.'

    axis_col = grouping_col[0]
    
    # SET X AXIS
    # Labels
    # no particular setup if number of labels is less than the first threshold
    
    num_xlabels = count_df[axis_col].nunique()
    
    if num_xlabels < LABEL_THRESHOLD_ROTATION :
        g.set_xticklabels(count_df[axis_col].unique())
        
    # rotate by 90 degrees if number of labels is between first and second threshold
    elif num_xlabels < LABEL_THRESHOLD_SELECT :
        g.set_xticklabels(count_df[axis_col].unique(), rotation=90)
       
    # display only certain labels (and rotate by 45 degrees) if number of labels is higher
    else :        
        number_of_steps = int(num_xlabels/50)

        l = np.arange(0, num_xlabels, number_of_steps)
        
        my_xticklabels=[]
        for x in g.axes[0,0].get_xticklabels():
            my_xticklabels.append(int(x.get_text()))
        
        pos = (l / num_xlabels) * (max(my_xticklabels)-min(my_xticklabels))
        g.set(xticks=pos)
        g.set_xticklabels(count_df[axis_col].unique()[l], rotation=45);
    
    # Title
    # option to remove the x axis title (when its obvious, e.g. for the years)
    xtitle = '' if hide_xtitle else axis_col;
    
    # SET Y AXIS
    # log scale option
    if log_y :
        g.set(yscale="log")
    
    # Title
    if facet=='freq':
        ytitle = '# %s (log scale)' % level if log_y else '# %s' % level
    elif facet=='avg':
        ytitle = 'title length (log scale)' if log_y else 'title length' 
        
    # Labels        
    for ax in g.axes[0]:
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
    
    # Plot Titles
    g.set_axis_labels(x_var=xtitle, y_var=ytitle);
    if facet=='freq':
        g.ax.set_title('Number of %s by %s per %s' % (level, grouping_col[0], grouping_col[1]))

    elif facet=='avg':
        g.ax.set_title('Average title length of content items by %s per %s' % (grouping_col[0], grouping_col[1]))

