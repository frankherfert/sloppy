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