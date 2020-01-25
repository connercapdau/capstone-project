# install libraries used in multiple functions
import pandas as pd
import numpy as np


def read_pitch_datasets():
    """
    This function reads in the different pitch type datasets.
    """
    
    data_100_min_pitches = pd.read_csv('../Datasets/100_min_pitches.csv')
    data_4_seam_fb = pd.read_csv('../Datasets/4_seam_fb.csv')
    data_2_seam_fb = pd.read_csv('../Datasets/2_seam_fb.csv')
    data_changeup = pd.read_csv('../Datasets/changeup.csv')
    data_curveball = pd.read_csv('../Datasets/curveball.csv')
    data_cut_fb = pd.read_csv('../Datasets/cut_fb.csv')
    data_eephus = pd.read_csv('../Datasets/eephus.csv')
    data_forkball = pd.read_csv('../Datasets/forkball.csv')
    data_knuckle_curve = pd.read_csv('../Datasets/knuckle_curve.csv')
    data_knuckleball = pd.read_csv('../Datasets/knuckleball.csv')
    data_sinker = pd.read_csv('../Datasets/sinker.csv')
    data_slider = pd.read_csv('../Datasets/slider.csv')
    data_split_finger = pd.read_csv('../Datasets/split_finger.csv')
    movement_sinker = pd.read_csv('../Datasets/2_seam_fb_sinker_pitch_movement.csv') # includes sinker and 2-seam data
    movement_4_seam_fb = pd.read_csv('../Datasets/4_seam_fb_pitch_movement.csv')
    movement_changeup = pd.read_csv('../Datasets/changeup_pitch_movement.csv')
    movement_curveball = pd.read_csv('../Datasets/curveball_knuckle_curve_pitch_movement.csv') # includes curve and knuckle curve data
    movement_cut_fb = pd.read_csv('../Datasets/cut_fb_pitch_movement.csv')
    movement_slider = pd.read_csv('../Datasets/slider_pitch_movement.csv')
    movement_splitter = pd.read_csv('../Datasets/splitter_pitch_movement.csv')
    movement_all = pd.read_csv('../Datasets/pitch_movement.csv')

    all_datasets = [data_100_min_pitches, data_4_seam_fb, data_2_seam_fb, data_changeup,
                    data_curveball, data_cut_fb, data_eephus, data_forkball, data_knuckle_curve,
                    data_knuckleball, data_sinker, data_slider, data_split_finger]

    ind_pitch_datasets = [data_4_seam_fb, data_2_seam_fb, data_changeup,
                    data_curveball, data_cut_fb, data_eephus, data_forkball, data_knuckle_curve,
                    data_knuckleball, data_sinker, data_slider, data_split_finger]

    movement_datasets = [movement_4_seam_fb, movement_sinker, movement_changeup, movement_curveball,
                         movement_cut_fb, movement_slider, movement_splitter]

    # drop columns for individual pitch datasets. keep columns: 1, 2, 4, 9, 10, 16, 17, 18, 23
    # column names: player_id, player_name, pitch_percent, woba, xwoba, spin_rate, velocity, effective_speed, release_extension
    for i in range(len(ind_pitch_datasets)):
        ind_pitch_datasets[i].drop(ind_pitch_datasets[i].columns[[0,3,5,6,7,8,11,12,13,14,15,19,20,21,22,
                                                                  24,25,26,27,28,29,30]], axis=1, inplace=True)

    # drop same columns for data_100_min_pitches
    data_100_min_pitches.drop(data_100_min_pitches.columns[[0,3,4,5,6,7,8,11,12,13,14,15,19,20,21,22,
                                                            24,25,26,27,28,29,30]], axis=1, inplace=True)

    # drop year, last/first/abbrev name, avg_speed, total pitches, pitches/game, pitch type, pitch name (idx: 0,1,2,5,7,9,10,12,13)
    for i in range(len(movement_datasets)):
        movement_datasets[i].drop(movement_datasets[i].columns[[0,1,2,5,7,9,10,12,13]], axis=1, inplace=True)

    # rename columns so that pitch type is added to columns that are specific to pitch types
    pitch_type = ['4_seam_fb', '2_seam_fb', 'changeup', 'curveball', 'cut_fb', 'eephus', 'forkball', 'knuckle_curve',
                  'knuckleball', 'sinker', 'slider', 'split_finger']
    for i in range(len(ind_pitch_datasets)):
        col_names = ['player_id', 'player_name', '%s_pitch_percent' % pitch_type[i], '%s_woba' % pitch_type[i], '%s_xwoba' % pitch_type[i],
                     '%s_spin_rate' % pitch_type[i], '%s_velocity' % pitch_type[i], '%s_effective_speed' % pitch_type[i],
                     '%s_release_extension' % pitch_type[i]]
        ind_pitch_datasets[i].columns = col_names

    movement_pitch_type = ['4_seam_fb', 'sinker', 'changeup', 'curveball', 'cut_fb', 'slider', 'split_finger']
    for i in range(len(movement_datasets)):
        col_names = ['player_id', 'team_name', 'pitch_hand', '%s_pitches_thrown' % movement_pitch_type[i],
                     '%s_pitch_percentage' % movement_pitch_type[i], '%s_pitcher_break_z' % movement_pitch_type[i],
                     '%s_league_break_z' % movement_pitch_type[i], '%s_diff_z' % movement_pitch_type[i],
                     '%s_rise' % movement_pitch_type[i], '%s_pitcher_break_x' % movement_pitch_type[i],
                     '%s_league_break_x' % movement_pitch_type[i], '%s_diff_x' % movement_pitch_type[i],
                     '%s_tail' % movement_pitch_type[i], '%s_percent_rank_diff_z' % movement_pitch_type[i],
                     '%s_percent_rank_diff_x' % movement_pitch_type[i]]
        movement_datasets[i].columns = col_names

    return data_100_min_pitches, ind_pitch_datasets, movement_datasets


def join_datasets(data_100_min_pitches, ind_pitch_datasets, movement_datasets):
    """
    This function is used to combine all the datasets into a single dataframe on player_id and player_name.
    """

    for i in range(len(ind_pitch_datasets)):
        if 'df' not in globals():
            df = pd.merge(data_100_min_pitches, ind_pitch_datasets[i], how='left', left_on=['player_id', 'player_name'], right_on = ['player_id', 'player_name'])
        else:
            df = pd.merge(df, ind_pitch_datasets[i], how='left', left_on=['player_id', 'player_name'], right_on = ['player_id', 'player_name'])

    # add pitch movement datasets to df
    for i in range(len(movement_datasets)):
        if i == 0:
            df = pd.merge(df, movement_datasets[i], how='left', left_on=['player_id'], right_on=['player_id'])
        else:
            # merge on pitcher_id, team_name, pitch_hand;
            # add avg_velocity, pitches_thrown, pitcher_break_z, rise, break_x, tail
            df = pd.merge(df, movement_datasets[i], how='left', left_on=['player_id'], right_on=['player_id'], suffixes=('','_y'))
            df['team_name'].fillna(df['team_name_y'], inplace=True)
            df['pitch_hand'].fillna(df['pitch_hand_y'], inplace=True)
            del (df['team_name_y'], df['pitch_hand_y'])

    # move team_name and pitch_hand farther up on the list of columns
    cols = ['player_id', 'player_name', 'team_name', 'pitch_hand', 'woba', 'xwoba', 'spin_rate', 'effective_speed', 'release_extension',
            '4_seam_fb_woba', '4_seam_fb_xwoba', '4_seam_fb_pitch_percent', '4_seam_fb_spin_rate', '4_seam_fb_velocity', '4_seam_fb_effective_speed', '4_seam_fb_release_extension',
            '2_seam_fb_woba',  '2_seam_fb_xwoba', '2_seam_fb_pitch_percent', '2_seam_fb_spin_rate', '2_seam_fb_velocity', '2_seam_fb_effective_speed', '2_seam_fb_release_extension',
            'changeup_woba', 'changeup_xwoba', 'changeup_pitch_percent', 'changeup_spin_rate', 'changeup_velocity', 'changeup_effective_speed', 'changeup_release_extension',
            'curveball_woba', 'curveball_xwoba', 'curveball_pitch_percent', 'curveball_spin_rate', 'curveball_velocity', 'curveball_effective_speed', 'curveball_release_extension',
            'cut_fb_woba', 'cut_fb_xwoba', 'cut_fb_pitch_percent', 'cut_fb_spin_rate', 'cut_fb_velocity', 'cut_fb_effective_speed', 'cut_fb_release_extension', 'eephus_woba',
            'eephus_xwoba', 'eephus_pitch_percent', 'eephus_spin_rate', 'eephus_velocity', 'eephus_effective_speed', 'eephus_release_extension', 'forkball_woba', 'forkball_xwoba',
            'forkball_pitch_percent', 'forkball_spin_rate', 'forkball_velocity', 'forkball_effective_speed', 'forkball_release_extension', 'knuckle_curve_woba', 'knuckle_curve_xwoba',
            'knuckle_curve_pitch_percent', 'knuckle_curve_spin_rate', 'knuckle_curve_velocity', 'knuckle_curve_effective_speed', 'knuckle_curve_release_extension', 'knuckleball_woba',
            'knuckleball_xwoba', 'knuckleball_pitch_percent', 'knuckleball_spin_rate', 'knuckleball_velocity', 'knuckleball_effective_speed', 'knuckleball_release_extension', 'sinker_woba',
            'sinker_xwoba', 'sinker_pitch_percent', 'sinker_spin_rate', 'sinker_velocity', 'sinker_effective_speed', 'sinker_release_extension', 'slider_woba', 'slider_xwoba',
            'slider_pitch_percent', 'slider_spin_rate', 'slider_velocity', 'slider_effective_speed', 'slider_release_extension', 'split_finger_woba', 'split_finger_xwoba',
            'split_finger_pitch_percent', 'split_finger_spin_rate', 'split_finger_velocity', 'split_finger_effective_speed', 'split_finger_release_extension', '4_seam_fb_pitches_thrown',
            '4_seam_fb_pitch_percentage', '4_seam_fb_pitcher_break_z', '4_seam_fb_league_break_z', '4_seam_fb_diff_z', '4_seam_fb_rise',
            '4_seam_fb_pitcher_break_x', '4_seam_fb_league_break_x', '4_seam_fb_diff_x', '4_seam_fb_tail', '4_seam_fb_percent_rank_diff_z',
            '4_seam_fb_percent_rank_diff_x', 'sinker_pitches_thrown', 'sinker_pitch_percentage', 'sinker_pitcher_break_z',
            'sinker_league_break_z', 'sinker_diff_z', 'sinker_rise', 'sinker_pitcher_break_x', 'sinker_league_break_x', 'sinker_diff_x',
            'sinker_tail', 'sinker_percent_rank_diff_z', 'sinker_percent_rank_diff_x', 'changeup_pitches_thrown',
            'changeup_pitch_percentage', 'changeup_pitcher_break_z', 'changeup_league_break_z', 'changeup_diff_z', 'changeup_rise',
            'changeup_pitcher_break_x', 'changeup_league_break_x', 'changeup_diff_x', 'changeup_tail', 'changeup_percent_rank_diff_z',
            'changeup_percent_rank_diff_x', 'curveball_pitches_thrown', 'curveball_pitch_percentage',
            'curveball_pitcher_break_z', 'curveball_league_break_z', 'curveball_diff_z', 'curveball_rise', 'curveball_pitcher_break_x',
            'curveball_league_break_x', 'curveball_diff_x', 'curveball_tail', 'curveball_percent_rank_diff_z', 'curveball_percent_rank_diff_x',
            'cut_fb_pitches_thrown', 'cut_fb_pitch_percentage', 'cut_fb_pitcher_break_z', 'cut_fb_league_break_z',
            'cut_fb_diff_z', 'cut_fb_rise', 'cut_fb_pitcher_break_x', 'cut_fb_league_break_x', 'cut_fb_diff_x', 'cut_fb_tail',
            'cut_fb_percent_rank_diff_z', 'cut_fb_percent_rank_diff_x', 'slider_pitches_thrown', 'slider_pitch_percentage',
            'slider_pitcher_break_z', 'slider_league_break_z', 'slider_diff_z', 'slider_rise', 'slider_pitcher_break_x', 'slider_league_break_x',
            'slider_diff_x', 'slider_tail', 'slider_percent_rank_diff_z', 'slider_percent_rank_diff_x',
            'split_finger_pitches_thrown', 'split_finger_pitch_percentage', 'split_finger_pitcher_break_z', 'split_finger_league_break_z',
            'split_finger_diff_z', 'split_finger_rise', 'split_finger_pitcher_break_x', 'split_finger_league_break_x', 'split_finger_diff_x',
            'split_finger_tail', 'split_finger_percent_rank_diff_z', 'split_finger_percent_rank_diff_x']
    df = df[cols]
    # change pitch percent to decimal value
    pitch_type = ['4_seam_fb', 'sinker', 'cut_fb', 'changeup', 'curveball', 'slider', 'split_finger']
    for i in range(len(pitch_type)):
        df['%s_pitch_percent' % pitch_type[i]] = df['%s_pitch_percent' % pitch_type[i]] / 100

    return df


def remove_too_few_pitches(df):
    """
    After exploring the data, I discovered there aren't enough pitchers that throw knuckleballs, eephus or forkballs.
    This function removes all columns related to those pitch types, then reorganizes the data
    """
    
    # remove the knuckleballers and forkball pitcher
    df = df.drop(df.index[645])
    df = df.drop(df.index[684])
    df = df.drop(df.index[701])
    # remove eephus, knuckleball, and forkball
    df = df[df.columns.drop(list(df.filter(regex='knuckleball')))]
    df = df[df.columns.drop(list(df.filter(regex='forkball')))]
    df = df[df.columns.drop(list(df.filter(regex='eephus')))]

    # reduce pitch types and drop extra percentage column
    df = df.drop(columns=['4_seam_fb_pitch_percentage',
                          'sinker_pitch_percentage',
                          'changeup_pitch_percentage',
                          'slider_pitch_percentage',
                          'curveball_pitch_percentage',
                          'cut_fb_pitch_percentage',
                          'split_finger_pitch_percentage'])

    df['sinker_pitch_percent'].fillna(df['2_seam_fb_pitch_percent'], inplace=True)
    df['sinker_spin_rate'].fillna(df['2_seam_fb_spin_rate'], inplace=True)
    df['sinker_velocity'].fillna(df['2_seam_fb_velocity'], inplace=True)
    df['sinker_effective_speed'].fillna(df['2_seam_fb_effective_speed'], inplace=True)
    df['sinker_release_extension'].fillna(df['2_seam_fb_release_extension'], inplace=True)
    df['curveball_pitch_percent'].fillna(df['knuckle_curve_pitch_percent'], inplace=True)
    df['curveball_spin_rate'].fillna(df['knuckle_curve_spin_rate'], inplace=True)
    df['curveball_velocity'].fillna(df['knuckle_curve_velocity'], inplace=True)
    df['curveball_effective_speed'].fillna(df['knuckle_curve_effective_speed'], inplace=True)
    df['curveball_release_extension'].fillna(df['knuckle_curve_release_extension'], inplace=True)
    df = df.drop(columns=['2_seam_fb_pitch_percent', '2_seam_fb_spin_rate', '2_seam_fb_velocity', '2_seam_fb_effective_speed', '2_seam_fb_release_extension',
                                      'knuckle_curve_pitch_percent', 'knuckle_curve_spin_rate', 'knuckle_curve_velocity', 'knuckle_curve_effective_speed', 'knuckle_curve_release_extension'])

    # reorganize columns together by pitch type
    cols_df = ['player_id', 'player_name', 'team_name', 'pitch_hand', 'xwoba',
                              '4_seam_fb_pitch_percent', '4_seam_fb_spin_rate', '4_seam_fb_velocity', '4_seam_fb_effective_speed', '4_seam_fb_release_extension',
                              '4_seam_fb_pitcher_break_z', '4_seam_fb_rise', '4_seam_fb_pitcher_break_x', '4_seam_fb_tail',
                              'sinker_pitch_percent', 'sinker_spin_rate', 'sinker_velocity', 'sinker_effective_speed', 'sinker_release_extension',
                              'sinker_pitcher_break_z', 'sinker_rise', 'sinker_pitcher_break_x', 'sinker_tail',
                              'cut_fb_pitch_percent', 'cut_fb_spin_rate', 'cut_fb_velocity', 'cut_fb_effective_speed', 'cut_fb_release_extension',
                              'cut_fb_pitcher_break_z', 'cut_fb_rise', 'cut_fb_pitcher_break_x', 'cut_fb_tail',
                              'changeup_pitch_percent', 'changeup_spin_rate', 'changeup_velocity', 'changeup_effective_speed', 'changeup_release_extension',
                              'changeup_pitcher_break_z', 'changeup_rise', 'changeup_pitcher_break_x', 'changeup_tail',
                              'curveball_pitch_percent', 'curveball_spin_rate', 'curveball_velocity', 'curveball_effective_speed', 'curveball_release_extension',
                              'curveball_pitcher_break_z', 'curveball_rise', 'curveball_pitcher_break_x', 'curveball_tail',
                              'slider_pitch_percent', 'slider_spin_rate', 'slider_velocity', 'slider_effective_speed', 'slider_release_extension',
                              'slider_pitcher_break_z', 'slider_rise', 'slider_pitcher_break_x', 'slider_tail',
                              'split_finger_pitch_percent', 'split_finger_spin_rate', 'split_finger_velocity', 'split_finger_effective_speed', 'split_finger_release_extension',
                              'split_finger_pitcher_break_z', 'split_finger_rise', 'split_finger_pitcher_break_x', 'split_finger_tail']
    df = df[cols_df]

    return df


def evaluate(model, test_x, test_y):
    """
    This function is for evaluating a model
    """

    predictions = model.predict(test_x)
    # predictions = predictions.reshape(len(predictions), 1)
    errors = abs(predictions - test_y)
    mape = 100 * np.mean(errors / test_y)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error = %.4f points' % np.mean(errors.values))
    print('Accuracy = %0.2f' % accuracy + '%')
    return accuracy


def simple_df(df):
    """
    Not all samples have movement data (481 out of 707 pitchers have movement data). This function outputs simple_df
    which doesn't include movement data
    """

    df = df[['player_id', 'player_name', 'team_name', 'pitch_hand', 'xwoba',
             '4_seam_fb_pitch_percent', '4_seam_fb_spin_rate', '4_seam_fb_effective_speed',
             'sinker_pitch_percent', 'sinker_spin_rate', 'sinker_effective_speed',
             'cut_fb_pitch_percent', 'cut_fb_spin_rate', 'cut_fb_effective_speed',
             'changeup_pitch_percent', 'changeup_spin_rate', 'changeup_effective_speed',
             'curveball_pitch_percent', 'curveball_spin_rate', 'curveball_effective_speed',
             'slider_pitch_percent', 'slider_spin_rate', 'slider_effective_speed',
             'split_finger_pitch_percent', 'split_finger_spin_rate',
             'split_finger_effective_speed']].copy()

    # one hot encode pitch_hand
    df['pitch_hand'] = pd.get_dummies(df['pitch_hand'])

    # replace nan with 0
    df = df.fillna(0)

    return df


def split_simple_data(df):
    """
    This function is used for splitting the simple dataframe into train and test sets
    """

    from sklearn.model_selection import train_test_split

    train_simple_df, test_simple_df = train_test_split(simple_df, test_size=0.33, random_state=1)

    train_x = train_simple_df[
        ['pitch_hand', '4_seam_fb_pitch_percent', '4_seam_fb_spin_rate', '4_seam_fb_effective_speed',
         'sinker_pitch_percent', 'sinker_spin_rate', 'sinker_effective_speed',
         'cut_fb_pitch_percent', 'cut_fb_spin_rate', 'cut_fb_effective_speed',
         'changeup_pitch_percent', 'changeup_spin_rate', 'changeup_effective_speed',
         'curveball_pitch_percent', 'curveball_spin_rate', 'curveball_effective_speed',
         'slider_pitch_percent', 'slider_spin_rate', 'slider_effective_speed',
         'split_finger_pitch_percent', 'split_finger_spin_rate', 'split_finger_effective_speed']]

    test_x = test_simple_df[
        ['pitch_hand', '4_seam_fb_pitch_percent', '4_seam_fb_spin_rate', '4_seam_fb_effective_speed',
         'sinker_pitch_percent', 'sinker_spin_rate', 'sinker_effective_speed',
         'cut_fb_pitch_percent', 'cut_fb_spin_rate', 'cut_fb_effective_speed',
         'changeup_pitch_percent', 'changeup_spin_rate', 'changeup_effective_speed',
         'curveball_pitch_percent', 'curveball_spin_rate', 'curveball_effective_speed',
         'slider_pitch_percent', 'slider_spin_rate', 'slider_effective_speed',
         'split_finger_pitch_percent', 'split_finger_spin_rate', 'split_finger_effective_speed']]

    train_y = train_simple_df[['xwoba']]
    test_y = test_simple_df[['xwoba']]

    return train_x, test_x, train_y, test_y


def complex_df(df):
    """
    This function is used to take only the pitchers that have movement data. This excludes about 1/3 of the pitchers
    """

    df = df[['player_id', 'player_name', 'team_name', 'pitch_hand', 'xwoba',
             '4_seam_fb_pitch_percent', '4_seam_fb_spin_rate', '4_seam_fb_velocity',
             '4_seam_fb_effective_speed', '4_seam_fb_release_extension',
             '4_seam_fb_pitcher_break_z', '4_seam_fb_rise', '4_seam_fb_pitcher_break_x', '4_seam_fb_tail',
             'sinker_pitcher_break_z', 'sinker_rise', 'sinker_pitcher_break_x', 'sinker_tail',
             'sinker_pitch_percent', 'sinker_spin_rate', 'sinker_velocity', 'sinker_effective_speed',
             'sinker_release_extension',
             'cut_fb_pitch_percent', 'cut_fb_spin_rate', 'cut_fb_velocity', 'cut_fb_effective_speed',
             'cut_fb_release_extension',
             'cut_fb_pitcher_break_z', 'cut_fb_rise', 'cut_fb_pitcher_break_x', 'cut_fb_tail',
             'changeup_pitch_percent', 'changeup_spin_rate', 'changeup_velocity',
             'changeup_effective_speed', 'changeup_release_extension',
             'changeup_pitcher_break_z', 'changeup_rise', 'changeup_pitcher_break_x', 'changeup_tail',
             'curveball_pitch_percent', 'curveball_spin_rate', 'curveball_velocity',
             'curveball_effective_speed', 'curveball_release_extension',
             'curveball_pitcher_break_z', 'curveball_rise', 'curveball_pitcher_break_x', 'curveball_tail',
             'slider_pitch_percent', 'slider_spin_rate', 'slider_velocity', 'slider_effective_speed',
             'slider_release_extension',
             'slider_pitcher_break_z', 'slider_rise', 'slider_pitcher_break_x', 'slider_tail',
             'split_finger_pitch_percent', 'split_finger_spin_rate', 'split_finger_velocity',
             'split_finger_effective_speed', 'split_finger_release_extension',
             'split_finger_pitcher_break_z', 'split_finger_rise', 'split_finger_pitcher_break_x',
             'split_finger_tail']].copy()

    # not all rows have pitch movement data. rows that don't have team_name don't have movement data so I'll remove them
    df = df[pd.notna(df['team_name']) == True]
    # 481 players left

    # one hot encode pitch_hand
    df['pitch_hand'] = pd.get_dummies(df['pitch_hand'])

    # replace nan with 0
    df = df.fillna(0)

    return df


def split_complex_data(df):
    """
    This function splits the complex data into train and test groups
    """

    from sklearn.model_selection import train_test_split
    train_model_features, test_model_features = train_test_split(df, test_size=0.33, random_state=1)

    train_x = train_model_features[['pitch_hand',
                                    '4_seam_fb_pitch_percent', '4_seam_fb_spin_rate', '4_seam_fb_velocity',
                                    '4_seam_fb_effective_speed', '4_seam_fb_release_extension',
                                    '4_seam_fb_pitcher_break_z', '4_seam_fb_rise', '4_seam_fb_pitcher_break_x',
                                    '4_seam_fb_tail',
                                    'sinker_pitcher_break_z', 'sinker_rise', 'sinker_pitcher_break_x', 'sinker_tail',
                                    'sinker_pitch_percent', 'sinker_spin_rate', 'sinker_velocity',
                                    'sinker_effective_speed', 'sinker_release_extension',
                                    'cut_fb_pitch_percent', 'cut_fb_spin_rate', 'cut_fb_velocity',
                                    'cut_fb_effective_speed', 'cut_fb_release_extension',
                                    'cut_fb_pitcher_break_z', 'cut_fb_rise', 'cut_fb_pitcher_break_x', 'cut_fb_tail',
                                    'changeup_pitch_percent', 'changeup_spin_rate', 'changeup_velocity',
                                    'changeup_effective_speed', 'changeup_release_extension',
                                    'changeup_pitcher_break_z', 'changeup_rise', 'changeup_pitcher_break_x',
                                    'changeup_tail',
                                    'curveball_pitch_percent', 'curveball_spin_rate', 'curveball_velocity',
                                    'curveball_effective_speed', 'curveball_release_extension',
                                    'curveball_pitcher_break_z', 'curveball_rise', 'curveball_pitcher_break_x',
                                    'curveball_tail',
                                    'slider_pitch_percent', 'slider_spin_rate', 'slider_velocity',
                                    'slider_effective_speed', 'slider_release_extension',
                                    'slider_pitcher_break_z', 'slider_rise', 'slider_pitcher_break_x', 'slider_tail',
                                    'split_finger_pitch_percent', 'split_finger_spin_rate', 'split_finger_velocity',
                                    'split_finger_effective_speed', 'split_finger_release_extension',
                                    'split_finger_pitcher_break_z', 'split_finger_rise', 'split_finger_pitcher_break_x',
                                    'split_finger_tail']]

    test_x = test_model_features[['pitch_hand',
                                  '4_seam_fb_pitch_percent', '4_seam_fb_spin_rate', '4_seam_fb_velocity',
                                  '4_seam_fb_effective_speed', '4_seam_fb_release_extension',
                                  '4_seam_fb_pitcher_break_z', '4_seam_fb_rise', '4_seam_fb_pitcher_break_x',
                                  '4_seam_fb_tail',
                                  'sinker_pitcher_break_z', 'sinker_rise', 'sinker_pitcher_break_x', 'sinker_tail',
                                  'sinker_pitch_percent', 'sinker_spin_rate', 'sinker_velocity',
                                  'sinker_effective_speed', 'sinker_release_extension',
                                  'cut_fb_pitch_percent', 'cut_fb_spin_rate', 'cut_fb_velocity',
                                  'cut_fb_effective_speed', 'cut_fb_release_extension',
                                  'cut_fb_pitcher_break_z', 'cut_fb_rise', 'cut_fb_pitcher_break_x', 'cut_fb_tail',
                                  'changeup_pitch_percent', 'changeup_spin_rate', 'changeup_velocity',
                                  'changeup_effective_speed', 'changeup_release_extension',
                                  'changeup_pitcher_break_z', 'changeup_rise', 'changeup_pitcher_break_x',
                                  'changeup_tail',
                                  'curveball_pitch_percent', 'curveball_spin_rate', 'curveball_velocity',
                                  'curveball_effective_speed', 'curveball_release_extension',
                                  'curveball_pitcher_break_z', 'curveball_rise', 'curveball_pitcher_break_x',
                                  'curveball_tail',
                                  'slider_pitch_percent', 'slider_spin_rate', 'slider_velocity',
                                  'slider_effective_speed', 'slider_release_extension',
                                  'slider_pitcher_break_z', 'slider_rise', 'slider_pitcher_break_x', 'slider_tail',
                                  'split_finger_pitch_percent', 'split_finger_spin_rate', 'split_finger_velocity',
                                  'split_finger_effective_speed', 'split_finger_release_extension',
                                  'split_finger_pitcher_break_z', 'split_finger_rise', 'split_finger_pitcher_break_x',
                                  'split_finger_tail']]

    train_y = train_model_features['xwoba']
    test_y = test_model_features['xwoba']

    return train_x, test_x, train_y, test_y


def create_baseline(test_y):
    """
    Establish a baseline prediction which is based on average of predicted value
    """

    base_pred = test_y.mean()
    base_errors = abs(base_pred - test_y)
    base_mape = 100 * np.mean(base_errors / test_y)
    base_accuracy = 100 - base_mape
    print('Baseline Error: %.4f points.' % np.mean(base_errors.values))
    print('Baseline Accuracy = %0.2f.' % base_accuracy)

    return base_accuracy, base_errors


def rr_model(train_x, test_x, train_y, test_y):
    """
    This function is used to create a Random Forest Regression model
    """

    from sklearn.ensemble import RandomForestRegressor

    regr = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=1)
    regr.fit(train_x, train_y.values.ravel())

    regr_accuracy = evaluate(regr, test_x, test_y)

    return regr, regr_accuracy


def standardized_complex_df(df):
    """
    This function standardizes the values from the output dataframe using complex_df function.
    """
    df = complex_df(df)

    complex_pitch_types = ['4_seam_fb', 'sinker', 'cut_fb', 'changeup', 'curveball', 'slider', 'split_finger']

    from sklearn.preprocessing import StandardScaler
    for i in range(len(complex_pitch_types)):
        df[['%s_spin_rate' % complex_pitch_types[i], '%s_effective_speed' % complex_pitch_types[i],
            '%s_pitcher_break_z' % complex_pitch_types[i], '%s_pitcher_break_x' % complex_pitch_types[i]]
        ] = StandardScaler().fit_transform(df[
                                               ['%s_spin_rate' % complex_pitch_types[i],
                                                '%s_effective_speed' % complex_pitch_types[i],
                                                '%s_pitcher_break_z' % complex_pitch_types[i],
                                                '%s_pitcher_break_x' % complex_pitch_types[i]
                                                ]])

    # effective_speed is a calculation based on velocity and release_extension so the latter two can be removed from model dataset
    for i in range(len(complex_pitch_types)):
        df = df.drop(columns=['%s_velocity' % complex_pitch_types[i], '%s_release_extension' % complex_pitch_types[i]])

    # break_z and rise are similar; break_x and tail are similar; break seems to be slightly more correlated with xwoba
    # so removing rise and tail
    for i in range(len(complex_pitch_types)):
        df = df.drop(columns=['%s_rise' % complex_pitch_types[i], '%s_tail' % complex_pitch_types[i]])

    return df


def pred_xwoba(df, model):
    # calculate pred_xwoba for all players
    all_samples_model = df[['total_pitch_score_w_pitch_percent']]
    df['pred_xwoba'] = model.predict(df[['total_pitch_score_w_pitch_percent']])

    # add column to df_model showing pred_xwoba - xwoba
    df['diff_xwoba'] = df['pred_xwoba'] - df['xwoba']

    # round diff_xwoba and pred_xwoba
    df[['diff_xwoba', 'pred_xwoba']] = df[['diff_xwoba', 'pred_xwoba']].round(3)

    return df


def find_overperformers(df, model, n):
    """
    df is the data
    model is the model used for predicting xwoba
    n is the number of largest overperformers to select
    """

    df1 = pred_xwoba(df, model)
    overperformed = df1[['player_name', 'pred_xwoba', 'xwoba', 'diff_xwoba']].nlargest(n, ['diff_xwoba'])

    return overperformed


def find_underperformers(df, model, n):
    """
    df is the data
    model is the model used for predicting xwoba 
    n is the number of largest underperformers to select
    """

    df1 = pred_xwoba(df, model)
    underperformed = df1[['player_name', 'pred_xwoba', 'xwoba', 'diff_xwoba']].nsmallest(n, ['diff_xwoba'])

    return underperformed
