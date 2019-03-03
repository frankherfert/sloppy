import pandas as pd
from IPython.core.display import display, HTML


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
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+
    print('pandas options updated')


def show_compact_df(df, column_level=1):    
    """
    requires 'from IPython.core.display import display, HTML'
    """
    
    style = """
        <style>
        th.rotate {
            /* Something you can count on */
            height: 140px;
            white-space: nowrap;
        }

        th.rotate > div {
            transform: 
                /* Magic Numbers */
                translate(25px, 51px)
                rotate(315deg);
            width: 30px;
        }

        th.rotate > div > span {
            border-bottom: 1px solid #ccc;
            padding: 5px 10px;
        }
        </style>"""

    dfhtml = style + df.to_html()

    try:
        colnames = df.columns.get_level_values(column_level).values
    except IndexError as e:
        colnames = df.columns.values

    for name in colnames:        
        dfhtml = dfhtml.replace('<th>{0}</th>'.format(name),
                                '<th class="rotate"><div><span>{0}</span></div></th>'.format(name))

    display(HTML(dfhtml))


def memory_usage(df_or_series):
    """
    Returns the size of a DataFrame in megabytes.
    """
    if type(df_or_series)==pd.core.frame.DataFrame:
        size = round(df_or_series.memory_usage(index=True, deep=True).sum()*1e-6,2)
    elif type(df_or_series)==pd.core.frame.Series:
        size = round(df_or_series.memory_usage(index=True, deep=True)*1e-6,2)

    size_str = "{:,.2f}".format(size)
    
    return size_str


def downcast_numeric_columns(df, column_type="int"):
    """

    """
    max_string_length = max([len(col) for col in list_of_columns])

    numeric_columns = full.select_dtypes('number').columns
    int_columns     = full.select_dtypes('int').columns
    float_columns   = full.select_dtypes('float').columns

    for col in numeric_columns:
        print("downcasting:", col.ljust(max_string_length), 'from', memory_usage(df[col]).rjust(8), end=' ')
        if col in int_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col in float_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        print(memory_usage(df[col]).rjust(8))
        
    return df

def set_learning_rate_with_resets(iteration, start=0.1, min_learning_rate=0.001, decay=0.99, reset_every=100, verbose=False):
    """
    LGB suitable learning rate decay
    Returns a decaying learning rate that will be reset to higher values at intervals.
    This can help to overcome local minima.
    """
    if reset_every!=None:
        rate = max(min_learning_rate, round(start * (decay ** ((iteration%reset_every)+1)),6))
        if verbose and iteration==reset_every: print("reset  ")
    else:
        rate = max(min_learning_rate, round(start * (decay ** iteration),6))
        if verbose: print(rate, end="\t")

    return rate














