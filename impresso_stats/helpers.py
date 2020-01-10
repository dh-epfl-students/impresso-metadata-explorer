from typing import Iterable
from datetime import date

import numpy as np
import pandas as pd
import dask
import dask.dataframe
import sqlalchemy
import warnings

from impresso_stats.sql import read_table, db_engine


def np_by_language(newspapers_languages_df: pd.core.frame.DataFrame,
                   languages_df: pd.core.frame.DataFrame,
                   language: str) -> pd.core.series.Series:
    """ Get IDs of newspapers having a specific language.
        Helper function for function np_ppty.
        :param pd.core.frame.DataFrame newspapers_languages_df: newspapers dataframe with languages info (language ID).
        :param pd.core.frame.DataFrame languages_df: dataframe containing info on each language (matching ID and language value).
        :param str language: language value on which we want to select newspapers (e.g. "fr").
        :return: filtered pandas series containing the IDs of the newspapers data frame for the selected language value.
        """

    if not (language in languages_df.code.unique()):
        raise ValueError("Chose a language among existing ones in db.")

    lang_id = languages_df.loc[languages_df['code'] == language]['id']
    lang_id = next(iter(lang_id), 'no match')
    
    return newspapers_languages_df.loc[newspapers_languages_df['language_id'] == lang_id]['newspaper_id']


def np_by_property(newspapers_metadata_df: pd.core.frame.DataFrame,
                   meta_properties_df: pd.core.frame.DataFrame,
                   property_name: str,
                   filter_: str) -> pd.core.series.Series:
    """ Get IDs of newspapers having a specific property value.
        Helper function for function np_ppty.
        :param pd.core.frame.DataFrame newspapers_metadata_df: newspapers dataframe with properties info.
        :param pd.core.frame.DataFrame meta_properties_df: dataframe containing info on each property (matching value and ID).
        :param str property_name: Property name on which we want to select newspapers.
        :param str filter_: Property value on which we want to select newspapers.
        :return: pandas series containing the newspaper's IDs for the selected property value.
        """

    if not (property_name in meta_properties_df.name.unique()):
        raise ValueError("Can't recognize selected property. Please chose one among existing ones in db meta_properties.")
        
    # Find ID
    prop_id = meta_properties_df.loc[meta_properties_df['name'] == property_name]['id']
    prop_id = next(iter(prop_id), 'no match')

    return newspapers_metadata_df.loc[(newspapers_metadata_df['property_id'] == prop_id)
                                      & (newspapers_metadata_df['value'] == filter_)]['newspaper_id']


def np_ppty(ppty_name: str, ppty_val: str, engine: sqlalchemy.engine.base.Engine) -> pd.core.frame.DataFrame:
    """ Get the id of newspapers having a specific property value.
        Loads data frames related to propertied ans calls function np_by_property. (also works with 'language')
        :param str ppty_name: Property name on which we want to select newspapers, or 'language'.
        :param str ppty_val: Property value on which we want to select newspapers.
        :param sqlalchemy.engine.base.Engine engine: sql engine for loading dataframes.
        :return: Pandas series containing the newspaper's ids for the selected property value.
        """
    if ppty_name=='language':
        newspapers_languages_df = read_table('newspapers_languages', engine)
        languages_df = read_table('languages', engine)
        result = np_by_language(newspapers_languages_df,languages_df, ppty_val)
        
    else:
        newspapers_metadata_df = read_table('newspapers_metadata', engine)
        meta_properties_df = read_table('meta_properties', engine)
        result = np_by_property(newspapers_metadata_df, meta_properties_df, ppty_name, ppty_val)
        
    return result


def np_country(code: str) -> pd.core.frame.DataFrame:
    """ Get the id of newspapers corresponding to a specific country.
        Calls function np_ppty.
        :param str code: Country code (e.g. 'CH' for Switzerland) .
        :return: Pandas series containing the newspaper's ids for the selected country value.
        """
    return np_ppty('countryCode', code, db_engine())


def check_dates(start_date: int, end_date: int) -> bool:
    """ Check validity of combination of start and end date, based on some criterions.
    :param int start_date: earliest date
    :param int end_date: latest date
    :return: True if pair is valid, false otherwise
    """

    assert(start_date is not None and end_date is not None), "Start and End dates should not be None."

    # End date must be after start date
    if start_date > end_date:
        return False

    # Start and end dates must be before today's year
    if start_date > date.today().year or end_date > date.today().year:
        return False

    return True


def decade_from_year_df(df: pd.core.frame.DataFrame, 
                        dask_df: bool = False) -> pd.core.frame.DataFrame:
    """ Created a column 'decade' based on the columns 'year' of a pandas data frame.
    :param pd.core.frame.DataFrame df: Data frame to be modified
    :param dask_df: set tu True if the dataframe is dask (and not pandas)
    :return: new pandas data frame with column decade.
    """
    if 'decade' in df.columns:
        return df
    elif 'year' in df.columns:
        result_df = df.copy()
        if dask_df is True :
            result_df['decade'] = result_df.year.apply(lambda y: y - y % 10, meta=('int'))
        else :
            result_df['decade'] = result_df.year.apply(lambda y: y - y % 10)
        return result_df
    else:
        raise ValueError("Decade columns already there, or year columns not there.")


def filter_df_by_np_id(df: pd.core.frame.DataFrame,
                       selected_np: Iterable) -> pd.core.series.Series:
    # Param selected_np should be a list or pd.Series => check if we have a more specific type for these
    """ Select only the rows of a data frame, corresponding to some newspapers ID.
        :param pd.core.frame.DataFrame df: Source data frame.
        :param Iterable selected_np: List or pandas Series containing the newspapers id which will be kept.
        :return: Pandas series containing the rows of the source data frame for the selected newspapers.
        """
    assert len(selected_np) > 0, "Given list of selected newspapers has length 0."

    # Column is usually not named the same in issues df and content item df
    # e.g. 'newspaper_id' in issues and 'newspaper' in ci.
    # add another condition if you use it with another column names for newspaper IDs.
    if 'newspaper_id' in df.columns:
        return df[df['newspaper_id'].isin(selected_np)]
    elif 'newspaper' in df.columns:
        return df[df['newspaper'].isin(selected_np)]
    else:
        raise ValueError("Columns 'newspaper_id' or 'newspaper' is not there. Check your columns' name.")


def check_all_column_count(df: pd.core.frame.DataFrame,
                           count_df: pd.core.frame.DataFrame,
                           grouping_columns: Iterable,
                           column_select: str,
                           print_: bool) -> (pd.core.frame.DataFrame, bool, Iterable):
    """ Check whether all columns of a data frame have the same count value after operations group by and count.
        Helper function for group_and_count function.
        :param pd.core.frame.DataFrame df: Source data frame.
        :param pd.core.frame.DataFrame count_df: Data frame after the group by and count operations.
        :param Iterable grouping_columns: List of columns on which the df has been groups by.
        :param str column_select: Reference column to which we compare others.
        :param bool print_: Whether we print some info in case of differences.
        :return: Tuple of three values : the data frame with a count column,
        a boolean indicating if all columns have the same counts (for all rows),
        a list containing the name of the columns for which some count values are different than the selected column.
        """

    all_same = True
    value_to_check = count_df[column_select]
    column_different_count = []

    # Check if all columns have the same count number
    for idx, col in enumerate(df.columns):
        if col not in grouping_columns:
            this_count = count_df[col]

            if not value_to_check.equals(this_count):
                all_same = False
                boolean_df = value_to_check.eq(this_count)
                column_different_count.append(col)
                if print_:
                    print("Column {} does not have the same count number : "
                          "lines {}.".format(col, boolean_df[boolean_df].index.values))

    # Convert pd.Series to pd.core.frame.DataFrame (for plotting for example)
    value_to_check = value_to_check.reset_index(name='count')

    return value_to_check, all_same, column_different_count


def group_and_count(df: pd.core.frame.DataFrame,
                    grouping_col: str,
                    time_gran: str,
                    column_select: str) -> (pd.core.frame.DataFrame, bool, Iterable):
    """ Perform group by and count on a data set.
        :param pd.core.frame.DataFrame df: Source data frame.
        :param str grouping_col: columns which the df should be grouped by (with the time).
        :param str time_gran: 'year' or 'decade' depending on the precision you want.
        :param str column_select: Reference column that we keep for the count values.
        :return: Tuple of three values : the result data frame with a count column,
        a boolean indicating if all columns have the same counts (for all rows),
        a list containing the name of the columns for which some count values are different than the selected column.
        """

    assert time_gran=='year' or time_gran=='decade', "Value of 'time_gran' parameter should be either 'year' of 'decade'."
    
    count_df = df.groupby([grouping_col, time_gran]).count()
    
    # Keep only the column we need
    count_df = count_df[column_select].rename('count')
    
    # For each index of the first level (grouping_col, usually the newspaper_id),
    # check if a time value is missing. 
    time_step = 1 if time_gran == 'year' else 10
    max_date = count_df.reset_index()[time_gran].max()
    min_date = count_df.reset_index()[time_gran].min()
    idx = np.arange(min_date, max_date+ time_step, time_step)

    my_dict = {}
    for idx0 in count_df.index.get_level_values(0).unique():
        sub_df = count_df.xs(idx0,level=grouping_col).reset_index()
        
        sub_df = sub_df.set_index(time_gran).reindex(idx).reset_index().fillna({'count':0}).fillna(method='ffill')

        my_dict[idx0] = sub_df
        
    result = pd.concat(my_dict.values(), keys=my_dict.keys()).reset_index()
    result = result.drop(['level_1'], axis=1).rename(columns={'level_0': grouping_col})
    
    return result


def filter_df(df: pd.core.frame.DataFrame,
              start_date: int = None,
              end_date: int = None,
              np_ids: Iterable = None,
              country: str = None,
              ppty: str = None,
              ppty_value: str = None) -> (pd.core.frame.DataFrame, pd.core.series.Series):
    """
    Returns a filtered data frame depending on the parameters
    :param df: original data frame to be filtered
    :param start_date: earliest date we want in the final df
    :param end_date: latests date we want in the final df
    :param np_ids: list of newspapers id to keep (drop all others)
    :param country: select by country (eg 'CH', 'LU')
    :param str ppty_name: Property name on which we want to select newspapers.
    :param str ppty_val: Property value on which we want to select newspapers.
    :return: Filtered data frame.
    """
    result_df = df.copy()

    # select specific np ids
    if np_ids is not None and len(np_ids) > 0:
        result_df = filter_df_by_np_id(result_df, np_ids)

    # select specific country
    if country is not None:
        countries = np_country(country)
        result_df = filter_df_by_np_id(result_df, countries)

    # select specific property
    if ppty is not None and ppty_value is not None:
        properties = np_ppty(ppty, ppty_value, db_engine())
        result_df = filter_df_by_np_id(result_df, properties)

    # check date values
    if start_date is not None and end_date is not None:
        assert check_dates(start_date, end_date), 'Problem with start and end dates.'

        # select dates
        result_df = result_df[(start_date <= result_df['year']) & (result_df['year'] <= end_date)]

    # take final list of np ids
    # Column is usually not named the same in issues df and content item df
    # e.g. 'newspaper_id' in issues and 'newspaper' in ci.
    # add another condition if you use it with another column names for newspaper IDs.
    if 'newspaper_id' in result_df.columns:
        np_ids_filtered = result_df.newspaper_id.unique()
    elif 'newspaper' in result_df.columns:
        np_ids_filtered = result_df.newspaper.unique()
    else:
        warnings.warn("Columns 'newspaper_id' or 'newspaper' is not there. Check your columns' name. \
        Second returned value is empty.")

    return result_df, np_ids_filtered


# ----------------------------# LICENCES #---------------------------- #

def license_stats_table(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Gives a table with statistics about the access rights per newspaper id in given df
    :param df: pandas data frame containing columns 'newspaper_id' and 'access-rights'
    :return: pandas data frame with rates on each access right level for each np
    """
    
    ar_df = df.groupby(['newspaper_id', 'access_rights']).count()
    ar_df = ar_df['id'].rename('count')
    ar_df = ar_df.reset_index()
    

    # Pivot to get a count per access right level & set the index correctly
    ar_df = ar_df.pivot_table('count', ['newspaper_id'], 'access_rights')
    ar_df.reset_index(drop=False, inplace=True)

    # Change NaN values to zeros
    ar_df = ar_df.fillna(0)

    # Compute the rate of access right per level per np
    ar_df['Total'] = ar_df[['Closed', 'OpenPrivate', 'OpenPublic']].sum(axis=1)
    ar_df['rate_Closed'] = ar_df['Closed'] / ar_df['Total']
    ar_df['rate_OpenPrivate'] = ar_df['OpenPrivate'] / ar_df['Total']
    ar_df['rate_OpenPublic'] = ar_df['OpenPublic'] / ar_df['Total']
    
    # Format columns
    
    # Remove useless decimals
    ar_df['Closed'] = ar_df['Closed'].map('{:,.0f}'.format)
    ar_df['OpenPrivate'] = ar_df['OpenPrivate'].map('{:,.0f}'.format)
    ar_df['OpenPublic'] = ar_df['OpenPublic'].map('{:,.0f}'.format)
    ar_df['Total'] = ar_df['Total'].map('{:,.0f}'.format)
    
    # Troncate rates
    ar_df['rate_Closed'] = ar_df['rate_Closed'].map(lambda x: '{:,.4%}'.format(x) if x>0 else 0)
    ar_df['rate_OpenPrivate'] = ar_df['rate_OpenPrivate'].map(lambda x: '{:,.4%}'.format(x) if x>0 else 0)
    ar_df['rate_OpenPublic'] = ar_df['rate_OpenPublic'].map(lambda x: '{:,.4%}'.format(x) if x>0 else 0)
    
    return ar_df


def multiple_ar_np(df: pd.core.frame.DataFrame) -> Iterable:
    """
    Finds the newspapers ids of newspapers which have issues with several different access right types
    :param df: pandas data frame with has 'access_rights' and 'newspapers_id' columns, in which to find the
    newspapers id (typically the issues data frame)
    :return: array of all
    """
    # Count number of different access rights per newspapers and mark the ones which have more that 1 access
    # right policy
    nb_ar_np = (df.groupby('newspaper_id')['access_rights'].nunique() > 1).reset_index(name='value')

    # Get ids of the newspapers which have several access right levels
    return nb_ar_np[nb_ar_np['value']].newspaper_id.unique()


# ----------------------------# CONTENT ITEMS #---------------------------- #

def licenses_ci_df(ci_df:  dask.dataframe.core.DataFrame) -> None:
    '''
    Helper function to get the access rights at the level of content items :
    Loads and join the sql table on issues, with the df passed as parameter.
    Warning : this function is very sensitive to any change in the column names 
    for the df passed as parameter and for the one in sql.
    :param df: dask data frame with column 'id' having format: NNN-yyyy-mm-dd-t-ixxxx 
    from which we can extract the issue_id : NNN-yyyy-mm-dd-t
    :return: Merged dataframe, with access right for each content item'''
    
    # Read df with access rights (as issues level) from sql
    engine = db_engine()
    issues_df = read_table('impresso.issues', engine)
    
    # Rename column 'id' to avoid name conflicts 
    issues_df = issues_df.rename(columns={"id": "issue_id"})
    
    # Extract issue_id from content item id
    ci_df['issue_id'] = ci_df.id.apply(lambda x: x[:x.rfind("-")], meta=str)
    
    # Merge dfs
    merged_df = ci_df.merge(issues_df, on='issue_id', suffixes=('_ci', '_issue'))
    
    # Select relevant columns and rename column year for compatibility with package functions
    merged_df = merged_df[['id', 'newspaper','decade', 'year_issue', 'month', 'day',  'type', 'n_tokens', \
                                   'title_length', 'access_rights']].rename(columns={'year_issue':'year'})
    
    return merged_df
