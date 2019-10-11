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
    assert len(selected_np) > 0, "Given list of selected newspapers has length 0."

    return df.loc[df['newspaper_id'].isin(selected_np)]


def check_all_column_count(df: pd.core.frame.DataFrame,
                           count_df: pd.core.frame.DataFrame,
                           grouping_columns: Iterable,
                           column_select: str,
                           print_: bool) -> (pd.core.frame.DataFrame, bool, Iterable):
    all_same = True
    value_to_check = count_df[column_select]
    column_different_count = []

    for idx, col in enumerate(df.columns) :
        if col not in grouping_columns :
            this_count = count_df[col]
            # print(this_count)
            if not value_to_check.equals(this_count) :
                all_same = False
                boolean_df = value_to_check.eq(this_count)
                column_different_count.append(col)
                if print_ :
                    print("Column {} does not have the same count number : "
                          "lines {}.".format(col, boolean_df[boolean_df].index.values))

    # Convert pd.Series to pd.Dataframe (for plotting for example)
    value_to_check = value_to_check.reset_index(name='count')
    return value_to_check, all_same, column_different_count


def group_and_count(df: pd.core.frame.DataFrame,
                    grouping_columns: Iterable,
                    column_select: str,
                    print_=True) -> (pd.core.frame.DataFrame, bool, Iterable) :
    count_df = df.groupby(grouping_columns).count()
    # print("ONE", count_df.head())
    return check_all_column_count(df, count_df, grouping_columns, column_select, print_)



#
