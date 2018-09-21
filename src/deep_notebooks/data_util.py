"""
@author Claas Voelcker

Utils module containing some data utility functions
"""


def get_categorical_data(spn, df, dictionary, header=1, types=False, date=False):
    """

    :param spn:
    :param df:
    :param dictionary:
    :param header:
    :param types:
    :param date:
    :return:
    """
    context = dictionary['context']
    categoricals = context.get_categoricals()
    df_numerical = df.copy(deep=True)
    for i in categoricals:
        transformed = dictionary['features'][i]['encoder'].transform(
            df_numerical.values[:, i])
        df_numerical.iloc[:, i] = transformed

    numerical_data = df_numerical.values

    categorical_data = {}
    for i in categoricals:
        data = df_numerical.groupby(context.feature_names[i])
        data = [data.get_group(x).values for x in data.groups]
        categorical_data[i] = data

    return numerical_data, categorical_data
