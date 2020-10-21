
from data import extra_data_plotter as edp
import pandas as pd
from data import batch_plotter as bp


def main():

    df_dict = {}
    for pressure_i in (850, 500):
        for dt in (1800, 3600):
            for month in ('january', 'july'):
                df = pd.read_pickle(bp.PATH_DF+str(dt)+'_'+month+'_' +
                                    str(pressure_i)+'_df_results.pkl')
                df_dict[(dt, month, pressure_i)] = df

    edp.multiple_filter_plotter(df_dict, bp.PATH_PLOT+'january' +
                                '_results_test', 'january')
    edp.multiple_filter_plotter(df_dict, bp.PATH_PLOT+'july' +
                                '_results_test', 'july')


if __name__ == "__main__":
    main()
