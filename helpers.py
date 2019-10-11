import pandas as pd
from typing import Iterable


def np_by_language(newspapers_languages_df: pd.core.frame.DataFrame,
                   languages_df: pd.core.frame.DataFrame,
                   language: str) -> pd.core.series.Series:

    assert language in languages_df.code.unique(), "Chose a language among existing ones in db."

    # Find ID
    lang_id = languages_df.loc[languages_df['code'] == language]['id'].item()
    return newspapers_languages_df.loc[newspapers_languages_df['language_id'] == lang_id]['newspaper_id']


def np_by_property(newspapers_metadata_df: pd.core.frame.DataFrame,
                   meta_properties_df: pd.core.frame.DataFrame,
                   property_name: str,
                   filter_: str) -> pd.core.series.Series:
    """ Get the id of newspapers having a specific property value.
        :param pd.core.frame.DataFrame newspapers_metadata_df: Newspapers data frame with properties info.
        # TODO BELOW !!!!!
        :param pd.core.frame.DataFrame meta_properties_df: Data frame after the group by and count operations.
        :param str property_name: Reference column to which we compare others.
        :param str filter_: Whether we print some info in case of differences.
        :return: Tuple of three values : the data frame with a count column,
        a boolean indicating if all columns have the same counts (for all rows),
        a list containing the name of the columns for which some count values are different than the selected column.
        """
    assert property_name in meta_properties_df.name.unique(), "Can't recognize selected property. \
    Please chose one among existing ones in db meta_properties."

    # Find ID
    prop_id = meta_properties_df.loc[meta_properties_df['name'] == property_name]['id'].item()

    # TODO : check if filter is one of values for given property ?

    return newspapers_metadata_df.loc[(newspapers_metadata_df['property_id']==prop_id)\
                                     & (newspapers_metadata_df['value']==filter_)]['newspaper_id']


def filter_df_by_np_id(df: pd.core.frame.DataFrame,
                       selected_np: Iterable) -> pd.core.series.Series:
    # Param selected_np should be a list or pd.Series => check if we have a more specific type for these
    """ Select only the rows if a data frame corresponding to some newspapers ID.
        :param pd.core.frame.DataFrame df: Source data frame.
        :param Iterable selected_np: List or pandas Series containing the newspapers id which will be kept.
        :return: Pandas series containing the rows of the source data frame for the selected newspapers.
        """
    assert len(selected_np) > 0, "Given list of selected newspapers has length 0."

    return df.loc[df['newspaper_id'].isin(selected_np)]


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

    for idx, col in enumerate(df.columns):
        if col not in grouping_columns:
            this_count = count_df[col]
            # print(this_count)
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
                    grouping_columns: Iterable,
                    column_select: str,
                    print_: bool = True) -> (pd.core.frame.DataFrame, bool, Iterable):
    """ Perform group by and count on a data set.
        :param pd.core.frame.DataFrame df: Source data frame.
        :param Iterable grouping_columns: List of columns which the df should be grouped by.
        :param str column_select: Reference column that we keep for the count values.
        :param bool print_: Whether we print some info in case of count differences between the columns.
        :return: Tuple of three values : the data frame with a count column,
        a boolean indicating if all columns have the same counts (for all rows),
        a list containing the name of the columns for which some count values are different than the selected column.
        """

    count_df = df.groupby(grouping_columns).count()
    return check_all_column_count(df, count_df, grouping_columns, column_select, print_)



#
