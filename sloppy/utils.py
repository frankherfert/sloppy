import pandas as pd
import os
from IPython.core.display import display, HTML
import datetime


def set_pd_options():
    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 200,
            'max_seq_items': 50,         # Max length of printed sequence
            'precision': 6,
            'show_dimensions': False
        },
        # 'mode': {
        #     'chained_assignment': None   # Controls SettingWithCopyWarning
        # }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+
    print('pandas options updated')


def display_df(df, column_level=1):    
    """
    requires 'from IPython.core.display import display, HTML'
    """
    max_col_length = len(max(df.columns, key=len))
    
    style = """
        <style>
        th.rotate {
            height: height_strpx;
            white-space: nowrap;
        }

        th.rotate > div {
            transform: 
                translate(25px, 51px)
                rotate(315deg);
            width: 30px;
        }

        th.rotate > div > span {
            border-bottom: 1px solid #ccc;
            padding: 5px 10px;
        }
        </style>""".replace('height_str', '140') #str(15*max_col_length))

    dfhtml = style + df.to_html()

    try:
        colnames = df.columns.get_level_values(column_level).values
    except IndexError as e:
        colnames = df.columns.values

    for name in colnames:        
        dfhtml = dfhtml.replace('<th>{0}</th>'.format(name),
                                '<th class="rotate"><div><span>{0}</span></div></th>'.format(name))

    display(HTML(dfhtml))


def downcast_numeric_columns(df, columns=[]):
    """

    """
    numeric_columns = df.loc[:, columns].select_dtypes('number').columns.tolist()
    int_columns     = df.loc[:, columns].select_dtypes('int').columns.tolist()
    float_columns   = df.loc[:, columns].select_dtypes('float').columns.tolist()

    max_string_length = max([len(col) for col in numeric_columns])+2

    for col in numeric_columns:
        print("downcasting:", col.ljust(max_string_length), 'from', memory_usage(df[col]).rjust(8), end=' ')
        if col in int_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col in float_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        print(memory_usage(df[col]).rjust(8))
        
    return df


def del_columns(df, columns):
    """
    Deletes columns one by one from the DataFrame. Easier to use during development compared to df.drop(columns)
    """
    
    for column in columns:
        if column in df.columns:
            del df[column]
            print("Deleted:          ", column)
        else:
            print("Not in DataFrame: ",column)
    
    return df


#### Files
def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.2f %s" % (num, x)
        num /= 1024.0


def get_file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


def show_path_file_sizes(path):
    """
    Shows sizes for all files in path
    """
    files = os.listdir(path)
    max_len = len(max(files, key=len))
    
    for f in files:
        print(f.ljust(max_len), get_file_size(f'{path}/{f}'))  


def memory_usage(df_or_series):
    """
    Returns the size of a DataFrame in megabytes.
    """
    if type(df_or_series)==pd.core.frame.DataFrame:
        size = round(df_or_series.memory_usage(index=True, deep=True).sum(),2)
    elif type(df_or_series)==pd.core.frame.Series:
        size = round(df_or_series.memory_usage(index=True, deep=True),2)
    
    return convert_bytes(size)


### features
def get_catg_cols(df):
    catg_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    
    return catg_cols


def get_features_list(df, prefix:str = 'ft_', suffix:str=None, sort_results = True):
    """
    Returns list of continous or categorical features from DataFrame.
    :prefix: 'cont' or 'catg'
    """
    
    column_list = [col for col in df.columns if col.startswith(prefix)]
    
    if suffix:
        column_list = [col for col in column_list
                       if ((col.find(suffix)>0) or (col.endswith(suffix)))]
    
    if sort_results:
        column_list = sorted(column_list)
    
    return column_list


def clean_feature_names(df, include: 'list or all'='all', exclude:list = None) -> pd.DataFrame:
    """
    Cleans feature names for easier column handling
    - replaces whitespaces and special character with underscores
    - removes duplicate prefixes
    """
    if include=='all':
        new_columns = [col.replace('ft_ft_', 'ft_').replace('-', '_') for col in df.columns]

        df.columns = new_columns
    else:
        print('not implemented yet')
    
    return df

### rest
def get_datetime_str(up_to='second'):
    if up_to=='second':
        s = str(datetime.datetime.now())[0:19]
        
    s = s.replace('-', '').replace(' ', '_').replace(':', '')
    return s


def round_to_nearest_int(value, base=25):
    """
    Rounds a number to the nearest base as integer value
    """
    try:
        return int(base * round(float(value)/base))
    except:
        return np.nan
