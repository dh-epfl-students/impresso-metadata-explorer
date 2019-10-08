import pandas as pd

def np_by_language(newspapers_languages_df, languages_df, language):

    assert language in languages_df.code.unique(), "Can't recognize selected languages. Please chose one among existing languages in db."

    # Find ID
    lang_id = languages_df.loc[languages_df['code']==language]['id'].item()
    return newspapers_languages_df.loc[newspapers_languages_df['language_id']==lang_id]['newspaper_id']


def np_by_property(newspapers_metadata_df, meta_properties_df, property_name, filter_):

    assert property_name in meta_properties_df.name.unique(), "Can't recognize selected property. \
    Please chose one among existing ones in db meta_properties."

    # Find ID
    prop_id = meta_properties_df.loc[meta_properties_df['name']==property_name]['id'].item()

    # TODO : check if filter is one of values for given property ?

    return newspapers_metadata_df.loc[(newspapers_metadata_df['property_id']==prop_id)\
                                     & (newspapers_metadata_df['value']==filter_)]['newspaper_id']

def filter_df_by_np_id(df, selected_np):
    assert len(selected_np) > 0, "Given list of selected newspapers has length 0."

    return df.loc[df['newspaper_id'].isin(selected_np)]

def check_all_column_count(df, count_df, grouping_columns, column_select, print_):
    all_same = True
    value_to_check = count_df[column_select]
    column_different_count = []

    for idx, col in enumerate(df.columns) :
        if not col in grouping_columns :
            this_count = count_df[col]
            #print(this_count)
            if not value_to_check.equals(this_count) :
                all_same = False
                boolean_df = value_to_check.eq(this_count)
                column_different_count.append(col)
                if print_ :
                    print("Column {} does not have the same count number : lines {}."\
                    .format(col, boolean_df[boolean_df].index.values))

    # Convert pd.Series to pd.Dataframe (for plotting for example)
    value_to_check = value_to_check.reset_index(name='count')
    return value_to_check, all_same, column_different_count


def group_and_count(df, grouping_columns, column_select, print_=True) :
    count_df = df.groupby(grouping_columns).count()
    #print("ONE", count_df.head())
    return check_all_column_count(df, count_df, grouping_columns, column_select, print_)



#
