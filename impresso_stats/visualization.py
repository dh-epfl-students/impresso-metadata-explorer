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

# ----------------------- ISSUES ----------------------- #
#plot_freq_issues
def plot_issues_time_id(time_gran: str,
                        start_date: int = None,
                        end_date: int = None,
                        np_ids: Iterable = None,
                        country: str = None,
                        df: pd.core.frame.DataFrame = None,
                        ppty: str = None,
                        ppty_value: str = None,
                        batch_size: int = None) -> None:
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
    :return: Nothing, but plots the histogram(s) of issue frequency.
    """
    issues_df = df

    # load data from SQL if needed
    if df is None:
        issues_df = read_table('impresso.issues', db_engine())

    issues_df, np_ids_filtered = filter_df(issues_df, start_date, end_date, np_ids, country, ppty, ppty_value)

    # check time_granularity is either 'year' or 'decade'
    assert (time_gran == 'decade' or time_gran == 'year'), "Time granularity must be either 'decade' or 'year'."

    # create it decade column if doesn't exist yet
    if time_gran == 'decade' and 'decade' not in df.columns:
        issues_df = decade_from_year_df(issues_df)

    # group and count for the histogram
    count_df, _, _ = group_and_count(issues_df, ['newspaper_id', time_gran], 'id', print_=False)

    # if batch_size not specified : plot all newspapers on the same figure
    if batch_size is None:
        g = sns.catplot(x=time_gran, y="count", hue="newspaper_id", kind="bar", data=count_df, height=5, aspect=2)
        display_setup(g, display_x_label=False, y_label='Number of issues',
                      title='Issue frequency per newspaper, through time.')

    # else plot by batches (no intelligent batching is done)
    else:
        assert (0 < batch_size and batch_size < 20), "Batch size must be between 1 and 19."
        catplot_by_batch_np(count_df, np_ids_filtered, 'decade', 'count', 'newspaper_id', batch_size,
                            display_x_label=False, y_label='Number of issues',
                            title='Issue frequency per newspaper, through time.')

#plot_freq_issues_batch
def catplot_by_batch_np(df: pd.core.frame.DataFrame,
                        np_list: Iterable,
                        xp: str,
                        yp: str,
                        huep: str,
                        max_cat: int = 5,
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
        g = sns.catplot(x=xp, y=yp, hue=huep, kind="bar", data=batch, height=5, aspect=2)
        display_setup(g, rotation, display_x_label, display_y_label, x_label, y_label, title)

# plt settings
def display_setup(g: sns.axisgrid.FacetGrid,
                  rotation: int = None,
                  display_x_label: bool = True,
                  display_y_label: bool =True,
                  x_label: str = None,
                  y_label: str = None,
                  title: str = None) -> None:
    """
    Common setup of seaborn plots : set titles and other stuff.
    :param g: the seaborn FacetGrid on which to do settings.
    :param rotation: degree of rotation for the x axis labels.
    :param display_x_label: boolean, set to False if you want to hide the x axis title. (default is True)
    :param display_y_label: boolean, set to False if you want to hide the y axis title. (default is True)
    :param x_label: specify particular x axis title
    :param y_label: specify particular y axis title
    :param title: specify title
    :return: Nothing. Modifies the g parameter.
    """

    # set axis titles
    if not display_x_label:
        g.set_xlabels('')
    elif x_label is not None:
        g.set_xlabels(x_label)

    if not display_y_label:
        g.set_ylabels('')
    elif y_label is not None:
        g.set_ylabels(y_label)

    # set rotation on x axis
    if rotation is not None:
        g.set_xticklabels(rotation=rotation)

    # set title DOESNT WORK
    if title is not None:
        g.set_titles(title)


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
        count_df, _, _ = group_and_count(result_df, ['newspaper_id', 'access_rights'], 'id')
        plot_licences_np(count_df, np_ids, batch_size)

    elif facet == 'time':
        count_df, _, _ = group_and_count(result_df, ['decade', 'access_rights'], 'id')
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

label_threshold_rotation = 30
label_threshold_select = 100
num_bars_threshold = 350

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
        

def plt_freq_ci_1d(df: dask.dataframe.core.DataFrame, 
                   grouping_col: str, 
                   asc: bool =False, 
                   hide_xtitle: bool =False, 
                   log_y: bool =False) -> pd.core.frame.DataFrame:
    """
    Displays a bar plot of the number of content items aggregated at one dimension in given df.
    :param df: dask data frame with columns 'id', grouping_col, or already aggregated with count column 'ci_count'
    :param grouping_col: column on which to aggregate the count (typically : 'year', 'type', 
                        'newspaper', or 'decade' if column is added by calling decade_from_year_df 
                        from helpers for example.
    :param asc: if set to True, plots bar in ascending order (default is descending order)
    :param hide_xtitle: if set to True, doesn't display title for x axis 
                        (typically useful if x axis are years) (default is False)
    :param log_y: if set to True, plot in logarithmic scale (for y axis) (default is False)
    :return: the aggregated df used for plotting
    """
    
    # Perfom the group by and count operation and convert to pandas df
    if 'ci_count' in df.columns:
        count_df = df
    else :
        count_df = df.groupby(grouping_col).id.count().compute().reset_index(name='ci_count')
    
    # Sort by count descending (default), or other if specified (time / ascending)
    if grouping_col == 'year' or grouping_col=='decade' :
        count_df.sort_values(by=grouping_col, inplace=True, ascending=True)
    else:
        count_df.sort_values(by='ci_count', inplace=True, ascending=asc)
    
    num_xlabels = len(count_df[grouping_col])
    
    # Plot figure
    plt.figure(figsize=(20,5))

    g = sns.barplot(x=grouping_col, y="ci_count", data=count_df, color='salmon');
    
    plt_settings_Axes(g, count_df, grouping_col, hide_xtitle, log_y)
    
    return count_df


def plt_settings_Axes(g: matplotlib.axes.SubplotBase, 
                      count_df: dask.dataframe.core.DataFrame,
                      grouping_col: list,
                      hide_xtitle: bool,
                      log_y: bool) -> None:
    '''
    Helper function for plot settings, used in function plt_freq_ci_1d.
    Modifies parameter g for setting titles, axis, formats, etc.
    :param g: matplotlib Axes which will be modified directly in the function.
    :param count_df: pandas dataframe which is plotted.
    :param grouping_col: column for x axis.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
    :param log_y: if set to True, plot in logarithmic scale (for y axis)
    :return: nothing. changes are done directly by modifiying parameter g.
    '''
    # SET X AXIS
    # Labels
    # no particular setup if number of labels is less than the first threshold
    num_xlabels = len(count_df[grouping_col])

    if num_xlabels < label_threshold_rotation:
        g.set_xticklabels(count_df[grouping_col])
        
    # rotate by 90 degrees if number of labels is between first and second threshold
    elif num_xlabels <  label_threshold_select:
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
        g.set_ylabel('# content items (log scale)')
    
    else :
        # Title
        g.set_ylabel('# content items')
        
    # Labels
    ylabels = ['{:,.0f}'.format(y) for y in g.get_yticks()]
    g.set_yticklabels(ylabels)
    
    # Plot Title
    g.set_title('Number of content items by %s' % grouping_col)
    
    
def plt_freq_ci_2d(df: dask.dataframe.core.DataFrame,
                   grouping_col: list,
                   hide_xtitle: bool =False,
                   log_y: bool =False) -> pd.core.frame.DataFrame:
    
    """
    Displays a categorical plot of the number of content items aggregated at two dimension in given df.
    :param df: dask data frame with column 'id' and column(s) in grouping_col, or with column 'ci_count' if already aggregated
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
    
    assert len(grouping_col)==2, "grouping_col parameter must be a list of length 2."
    assert not (grouping_col[1]=='year' or grouping_col[1]=='decade'), "Time cannot be used as categorical variable."
    
    if 'ci_count' in df.columns:
        count_df = df
    else :
        count_df = df.groupby(grouping_col).id.count().compute().reset_index(name='ci_count')
    
    if len(count_df) > num_bars_threshold:
        raise ValueError("The total number of bars to plot exceeds limit: "+ num_bars_thershold +". Not able to plot figure.\
        Please reduce by filtering the dataframe on some features.") 
    
    aggr_dim = grouping_col[0]
    cat_dim = grouping_col[1]
    
    ascending_0 = grouping_col[0]=='year' or grouping_col[0]=='decade'
    
    
    # Sort by count descending (default), or other if specified (time / ascending)
    count_df.sort_values(by=[grouping_col[0], 'ci_count'], inplace=True, ascending=[ascending_0, False])
    
    g = sns.catplot(x=grouping_col[0], y="ci_count", data=count_df, \
                hue=grouping_col[1], kind='bar', height=5, aspect=3)
    
    plt_settings_FacetGrid(g, count_df, grouping_col, hide_xtitle, log_y)
    
    return count_df


def plt_settings_FacetGrid(g: sns.axisgrid.FacetGrid, 
                           count_df: dask.dataframe.core.DataFrame, 
                           grouping_col: list, 
                           hide_xtitle: bool, 
                           log_y: bool) -> None:
    '''
    Helper function for plot settings, used in function plt_freq_ci_2d.
    Modifies parameter g for setting titles, axis, formats, etc.
    :param g: seaborn FacetGrid which will be modified directly in the function.
    :param count_df: pandas dataframe which is plotted.
    :param grouping_col: list of columns for category and x axis.
    :param hide_xtitle: if set to True, doesn't display title for x axis 
    :param log_y: if set to True, plot in logarithmic scale (for y axis)
    :return: nothing. changes are done directly by modifiying parameter g.
    '''
    axis_col = grouping_col[0]
    
    # SET X AXIS
    # Labels
    # no particular setup if number of labels is less than the first threshold
    
    num_xlabels = count_df[axis_col].nunique()
    
    if num_xlabels < label_threshold_rotation:
        g.set_xticklabels(count_df[axis_col])
        
    # rotate by 90 degrees if number of labels is between first and second threshold
    elif num_xlabels <  label_threshold_select:
        g.set_xticklabels(count_df[axis_col], rotation=90)
       
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
    ytitle = '# content items (log scale)' if log_y else '# content items'
        
    # Labels        
    for ax in g.axes[0]:
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
    
    # Plot Titles
    g.set_axis_labels(x_var=xtitle, y_var=ytitle);
    g.ax.set_title('Number of content items by %s per %s' % (grouping_col[0], grouping_col[1]))

