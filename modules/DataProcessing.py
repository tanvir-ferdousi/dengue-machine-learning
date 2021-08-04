import os
import datetime
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import savgol_filter
from geopy.distance import distance
from sklearn import preprocessing


def ger_corr_coeff(comp_df):
    r, p = stats.pearsonr(comp_df.dropna()['cases_1'], comp_df.dropna()['cases_2'])
    return r,p

def get_sliding_window_corr(comp_df, r_window_size):
    df_interpolated = comp_df.interpolate()
    rolling_r = df_interpolated['cases_1'].rolling(window=r_window_size, center=True).corr(df_interpolated['cases_2'])
    rolling_mean = comp_df.rolling(window=r_window_size, center=True).mean()

    plt.figure(num=None, figsize=(20,4))
    plt.plot(comp_df['date'], rolling_mean)
    plt.xlabel('Time')
    plt.ylabel('Dengue cases')
    plt.title('Sliding window mean')

    plt.figure(num=None, figsize=(20,4))
    plt.plot(comp_df['date'], rolling_r)
    plt.xlabel('Time')
    plt.ylabel('Pearson r')
    plt.title('Sliding window correlation')

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        c = datax.corr(datay.shift(lag))
        if np.isnan(c):
            c = 0
        return c

def get_windowed_time_lagged_xcorr(comp_df, lag_weeks, no_splits, show_heatmap = False):
    samples_per_split = comp_df.shape[0]/no_splits
    rss=[]
    for t in range(0, no_splits):
        d1 = comp_df['cases_1'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
        d2 = comp_df['cases_2'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
        rs = [crosscorr(d1,d2, lag) for lag in range(-int(lag_weeks-1),int(lag_weeks))]
        rss.append(rs)
    rss = pd.DataFrame(rss)

    if show_heatmap:
        f,ax = plt.subplots(figsize=(19,8))
        sns.heatmap(rss,cmap='RdBu_r',ax=ax)
#         title=f'Windowed Time Lagged Cross Correlation' RdBu_r
        ax.set(title = '$IC_1$ leads < | > $IC_2$ leads     ', xlim=[0,lag_weeks*2], xlabel= r'Time shift ($\theta$), weeks',ylabel='Window epochs ($w_k$)')
        ax.set_xticklabels([int(np.ceil(item-lag_weeks)) for item in ax.get_xticks()]);
        plt.savefig('corr_heatmap.svg')
    return rss.transpose()

def get_var_windowed_time_lagged_xcorr(comp_df, win_markers, lag_weeks, show_heatmap = False):
    rss = []
    for win in win_markers:
        d1 = comp_df['cases_1'].loc[win[0]:win[1]]
        d2 = comp_df['cases_2'].loc[win[0]:win[1]]
        rs = [crosscorr(d1,d2, lag) for lag in range(-int(lag_weeks-1),int(lag_weeks))]
        rss.append(rs)
    rss = pd.DataFrame(rss)
    
    if show_heatmap:
        f,ax = plt.subplots(figsize=(19,8))
        sns.heatmap(rss,cmap='RdBu_r',ax=ax)
#         title=f'Windowed Time Lagged Cross Correlation' RdBu_r
        ax.set(title = '$IC_1$ leads < | > $IC_2$ leads     ', xlim=[0,lag_weeks*2], xlabel= r'Time shift ($\theta$), weeks',ylabel='Window epochs ($w_k$)')
        ax.set_xticklabels([int(np.ceil(item-lag_weeks)) for item in ax.get_xticks()]);
#        plt.savefig('corr_heatmap.svg')
    
    return rss.transpose()

def get_corr_strength(corr_matrix, sideLen):
    meanCorr = []
    max_corr_idx = corr_matrix.idxmax()
    for i in range(0,corr_matrix.shape[1]):
        lowerIdx = max_corr_idx[i]-sideLen
        upperIdx = max_corr_idx[i]+ sideLen + 1

        lowerIdx = max(0,lowerIdx)
        upperIdx = min(corr_matrix.shape[0], upperIdx)

        nIdx = upperIdx-lowerIdx

        xcorr = 0

        for j in range(lowerIdx,upperIdx):
            xcorr = xcorr + corr_matrix[i][j]

        meanCorr.append(xcorr/nIdx)
    return pd.DataFrame(meanCorr)

def get_leadability(corr_matrix, extended_phase):
    # true means signal 1 lags, signal 2 leads. Signal 2 can predict signal 1
    # false means signal 1 leads, signal 2 lags. Signal 2 cannot predict signal 1.
    midpoint_idx = int(np.ceil(corr_matrix.shape[0]/2)-1)
    max_corr_idx = corr_matrix.idxmax()
    m2_lead = max_corr_idx > (midpoint_idx - extended_phase)
    return m2_lead

def get_predictor_estimates(corr_matrix, sideLen, corr_min, extended_phase):
    lead_table = get_leadability(corr_matrix, extended_phase)
    strength_table = get_corr_strength(corr_matrix, sideLen)

    predictor_units = []
    for i in range(0,lead_table.shape[0]):
        if lead_table[i] and strength_table[0][i] > corr_min:
            predictor_units.append(strength_table[0][i])
        else:
            predictor_units.append(0)
    predictor_units = pd.DataFrame(predictor_units)
    return predictor_units

def compute_predictability(ic_id, inc_df, lag_weeks, no_of_windows, side_len, min_corr, extend_phase):
    IC_list = inc_df.columns[1:]
    PIC_list = IC_list[IC_list != ic_id]

    comp_df = pd.DataFrame()
    comp_df['date'] = inc_df['date']
    comp_df['cases_1'] = inc_df[ic_id]

    # for each ic, pic combination, find if pic can predict ic
    PIC_predictabilities = pd.DataFrame()

    for pic_id in PIC_list:
#         print(f'Estimating predictability of pic: {pic_id}')

        comp_df['cases_2'] = inc_df[pic_id]

        corr_matrix = get_windowed_time_lagged_xcorr(comp_df, lag_weeks, no_of_windows)

        pred_est = get_predictor_estimates(corr_matrix, side_len, min_corr, extend_phase)

        PIC_predictabilities[str(pic_id)] = pred_est[0]

    return PIC_predictabilities

def compute_var_window_predictability(ic_id, inc_df, lag_weeks, min_win_len, risk_min, side_len, min_corr, extend_phase):
    IC_list = inc_df.columns[1:]
    PIC_list = IC_list[IC_list != ic_id]
    
    comp_df = pd.DataFrame()
    comp_df['date'] = inc_df['date']
    comp_df['cases_1'] = inc_df[ic_id]
    
    win_markers = detect_outbreak_windows(comp_df['cases_1'].values, comp_df['date'], risk_min, min_win_len, False)
    
    # for each ic, pic combination, find if pic can predict ic
    PIC_predictabilities = pd.DataFrame()
    
    if len(win_markers) == 0:
        return PIC_predictabilities

    for pic_id in PIC_list:
        
        comp_df['cases_2'] = inc_df[pic_id]
        
        corr_matrix = get_var_windowed_time_lagged_xcorr(comp_df, win_markers, lag_weeks)
        
        pred_est = get_predictor_estimates(corr_matrix, side_len, min_corr, extend_phase)
        
        PIC_predictabilities[str(pic_id)] = pred_est[0]
    
    return PIC_predictabilities

def sort_predictors(pic_predictabilities, pop_df, ic_id, sort_options):
    # Three options: corr_weight (True, False), prev_weight (True, False), prev_type ('relative', 'absolute'), dist_weight (True, False)
    pic_df = pd.DataFrame()
    pic_df['loc_id'] = pic_predictabilities.mean().index
    pic_df['corr_weight'] = pic_predictabilities.mean().values

    inc_w = []
    dist_w = []

    ic_coords = pop_df.loc[ic_id].lat, pop_df.loc[ic_id].lon

    for index, row in pic_df.iterrows():
        if sort_options['prev_type'] == 'absolute':
            inc_w.append(pop_df.loc[int(row.loc_id)].case_tot)
        else:
            inc_w.append(pop_df.loc[int(row.loc_id)].case_frac)

        pic_coords = pop_df.loc[int(row.loc_id)].lat, pop_df.loc[int(row.loc_id)].lon
        dist_w.append(distance(ic_coords, pic_coords).km)

    pic_df['prev_weight'] = inc_w
    pic_df['distance_weight'] = dist_w

    pic_df['corr_weight'] = (pic_df['corr_weight'] - pic_df['corr_weight'].min())/(pic_df['corr_weight'].max()-pic_df['corr_weight'].min())
    pic_df['prev_weight'] = (pic_df['prev_weight'] - pic_df['prev_weight'].min())/(pic_df['prev_weight'].max()-pic_df['prev_weight'].min())
    pic_df['distance_weight'] = (pic_df['distance_weight'] - pic_df['distance_weight'].min())/(pic_df['distance_weight'].max()-pic_df['distance_weight'].min())
    pic_df['distance_weight'] = 1 - pic_df['distance_weight']

    if not sort_options['corr_weight']:
        pic_df['corr_weight'] = pic_df['corr_weight']*0 + 1

    if not sort_options['prev_weight']:
        pic_df['prev_weight'] = pic_df['prev_weight']*0 + 1

    if not sort_options['dist_weight']:
        pic_df['distance_weight'] = pic_df['distance_weight']*0 + 1

    pic_df['combined_weight'] = pic_df['corr_weight']*(pic_df['prev_weight'] + pic_df['distance_weight'])
    pic_df.set_index('loc_id', inplace=True)

    return pic_df.sort_values('combined_weight', ascending=False).index

def sort_predictors_for_ecoregion(pic_predictabilities, pop_df, sort_options):
    # Three options: corr_weight (True, False), prev_weight (True, False), prev_type ('relative', 'absolute')
    pic_df = pd.DataFrame()
    pic_df['loc_id'] = pic_predictabilities.mean().index
    pic_df['corr_weight'] = pic_predictabilities.mean().values

    inc_w = []
    for index, row in pic_df.iterrows():
        if sort_options['prev_type'] == 'absolute':
            inc_w.append(pop_df.loc[int(row.loc_id)].case_tot)
        else:
            inc_w.append(pop_df.loc[int(row.loc_id)].case_frac)

    pic_df['prev_weight'] = inc_w

    pic_df['corr_weight'] = (pic_df['corr_weight'] - pic_df['corr_weight'].min())/(pic_df['corr_weight'].max()-pic_df['corr_weight'].min())
    pic_df['prev_weight'] = (pic_df['prev_weight'] - pic_df['prev_weight'].min())/(pic_df['prev_weight'].max()-pic_df['prev_weight'].min())

    if not sort_options['corr_weight']:
        pic_df['corr_weight'] = pic_df['corr_weight']*0 + 1

    if not sort_options['prev_weight']:
        pic_df['prev_weight'] = pic_df['prev_weight']*0 + 1

    pic_df['combined_weight'] = pic_df['corr_weight']*pic_df['prev_weight']

    pic_df.set_index('loc_id', inplace=True)

    return pic_df.sort_values('combined_weight', ascending=False).index

def get_window_markers(outbreak_indices, min_win_len):
    mark_in = -100
    mark_out = -100
    last_val = -100
    time_wins = []
    for x in outbreak_indices:
        if x - last_val != 1:
            # New segment
            mark_out = last_val
            if mark_in != -100:
                win = [mark_in, mark_out]
                if win[1]-win[0] >= min_win_len:
                    time_wins.append(win)
            mark_in = x

        last_val = x
    
    if mark_out <= mark_in:
        mark_out = last_val
        win = [mark_in, mark_out]
        if win[1]-win[0] >= min_win_len:
            time_wins.append(win)

    return time_wins

def find_best_performance_by_win(win_error_dict, ic_list, metric_index):
    win_keys = list(win_error_dict.keys())
    win_opt_dict = {}
    
    for w_key in win_keys:
        win_opt_df = pd.DataFrame()

        linear_opt_error = []
        lstm_opt_error = []
        gru_opt_error = []

        linear_opt_pic = []
        lstm_opt_pic = []
        gru_opt_pic = []

        for target_loc in ic_list:
            test_errors = win_error_dict[w_key][target_loc]
            linear_errors = []
            lstm_errors = []
            gru_errors = []
            error_df = pd.DataFrame()

            for nPIC in feature_counts:
                linear_errors.append(test_errors[nPIC]['Linear'][metric_index])
                lstm_errors.append(test_errors[nPIC]['LSTM'][metric_index])
                gru_errors.append(test_errors[nPIC]['GRU'][metric_index])

            error_df['nFeat'] = feature_counts
            error_df['linear'] = linear_errors
            error_df['lstm'] = lstm_errors
            error_df['gru'] = gru_errors
            error_df = error_df.set_index('nFeat')

            opt_error = error_df.min()
            opt_pic = error_df.idxmin()

            linear_opt_error.append(opt_error.linear)
            lstm_opt_error.append(opt_error.lstm)
            gru_opt_error.append(opt_error.gru)

            linear_opt_pic.append(opt_pic.linear)
            lstm_opt_pic.append(opt_pic.lstm)
            gru_opt_pic.append(opt_pic.gru)

        win_opt_df['ic'] = ic_list
        win_opt_df = win_opt_df.set_index('ic')
        win_opt_df['linear_opt_error'] = linear_opt_error
        win_opt_df['lstm_opt_error'] = lstm_opt_error
        win_opt_df['gru_opt_error'] = gru_opt_error

        win_opt_df['linear_opt_pic'] = linear_opt_pic
        win_opt_df['lstm_opt_pic'] = lstm_opt_pic
        win_opt_df['gru_opt_pic'] = gru_opt_pic
        
        win_opt_dict[w_key] = win_opt_df
    
    
    return win_opt_dict

def detect_outbreak_windows(y, x, risk_threshold, min_win_len, plotEnable=False):    
    yn = (y-y.min())/(y.max()-y.min())
    yhat = savgol_filter(yn, 11, 3)
    
    risk = np.zeros(len(y))
    
    risk[yhat >= risk_threshold] = 1
    
    outbreak_idx = [idx for idx,val in enumerate(risk) if val >= 1]
    win_markers = get_window_markers(outbreak_idx, min_win_len)
    
    return win_markers