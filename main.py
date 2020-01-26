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
    regr.predict()
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

    df = pred_xwoba(df, model)
    overperformed = df[['player_name', 'pred_xwoba', 'xwoba', 'diff_xwoba']].nlargest(n, ['diff_xwoba'])

    return overperformed


def find_underperformers(df, model, n):
    """
    df is the data
    model is the model used for predicting xwoba
    n is the number of largest underperformers to select
    """

    df = pred_xwoba(df, model)
    underperformed = df[['player_name', 'pred_xwoba', 'xwoba', 'diff_xwoba']].nsmallest(n, ['diff_xwoba'])

    return underperformed


if __name__ == '__main__':

    # I first reviewed the data and created visualizations to use in my paper. I had decided to use data within the
    # first 2 standard deviations. The visualization code I used is in the text block below; I didn't feel like it
    # needed to be ran in this portfolio.
    """
    # get data for 2 and 3 st dev
    from scipy import stats
    df_2sd = df[(np.abs(stats.zscore(df.xwoba)) < 2)]
    # df_3sd = df[(np.abs(stats.zscore(df.xwoba)) < 3)]
    
    corr_df_2sd = df_2sd.corr()
    # corr_df_3sd = df_3sd.corr()
    
    import seaborn as sns
    sns.set(style='ticks', color_codes=True)
    sns.distplot(df.xwoba)
    sns.plt.show()
    
    sns.distplot(df_3sd.xwoba)
    sns.plt.show()
    
    sns.distplot(df_2sd.xwoba)
    sns.plt.show()
    
    sns.pairplot(df, dropna=True, vars=['xwoba', 'spin_rate', 'effective_speed'], kind='reg')
    sns.plt.show()
    
    # drop pitch specific woba and xwoba columns and woba
    df_2sd = df_2sd[df_2sd.columns.drop(list(df_2sd.filter(regex='_xwoba')))]
    df_2sd = df_2sd[df_2sd.columns.drop(list(df_2sd.filter(regex='_woba')))]
    # df_2sd = df_2sd.drop(columns=['woba'])
    
    g = sns.PairGrid(df, y_vars=['xwoba'], x_vars=['spin_rate', 'effective_speed'])
    g.map(sns.regplot)
    g = sns.PairGrid(df_1_2sd, y_vars=['xwoba'], x_vars=['4_seam_fb_spin_rate', '4_seam_fb_effective_speed'])
    g.map(sns.regplot)
    g = sns.PairGrid(df_1_2sd, y_vars=['xwoba'], x_vars=['sinker_spin_rate', 'sinker_effective_speed'])
    g.map(sns.regplot)
    g = sns.PairGrid(df_1_2sd, y_vars=['xwoba'], x_vars=['cut_fb_spin_rate', 'cut_fb_effective_speed'])
    g.map(sns.regplot)
    g = sns.PairGrid(df_1_2sd, y_vars=['xwoba'], x_vars=['changeup_spin_rate', 'changeup_effective_speed'])
    g.map(sns.regplot)
    g = sns.PairGrid(df_1_2sd, y_vars=['xwoba'], x_vars=['curveball_spin_rate', 'curveball_effective_speed'])
    g.map(sns.regplot)
    g = sns.PairGrid(df_1_2sd, y_vars=['xwoba'], x_vars=['slider_spin_rate', 'slider_effective_speed'])
    g.map(sns.regplot)
    g = sns.PairGrid(df_1_2sd, y_vars=['xwoba'], x_vars=['split_finger_spin_rate', 'split_finger_effective_speed'])
    g.map(sns.regplot)
    
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    sns.regplot(x='4_seam_fb_spin_rate', y='xwoba', data=df_1_2sd, ax=ax)
    sns.regplot(x='4_seam_fb_effective_speed', y='xwoba', data=df_1_2sd, ax=ax2)
    sns.show()
    
    
    sns.regplot(x='sinker_effective_speed', y='xwoba', data=df_1_2sd, ax=axs[0])
    sns.regplot(x='cut_fb_spin_rate', y='xwoba', data=df_1_2sd, ax=axs[0])
    sns.regplot(x='cut_fb_effective_speed', y='xwoba', data=df_1_2sd, ax=axs[0])
    sns.regplot(x='changeup_fb_spin_rate', y='xwoba', data=df_1_2sd, ax=axs[0])
    sns.regplot(x='changeup_effective_speed', y='xwoba', data=df_1_2sd, ax=axs[0])
    sns.regplot(x='curveball_spin_rate', y='xwoba', data=df_1_2sd, ax=axs[0])
    sns.regplot(x='curveball_effective_speed', y='xwoba', data=df_1_2sd, ax=axs[0])
    """


    # read in all data
    data_1, data_2, data_3 = read_pitch_datasets()

    # combine datasets together
    df = join_datasets(data_1, data_2, data_3)

    # remove some of the pitches
    df = remove_too_few_pitches(df)


    # Approach 1
    # Random Forest Regression on data from simple_df()

    df_1 = simple_df(df)

    # split data
    train_x_1, test_x_1, train_y_1, test_y_1 = split_simple_data(df_1)
    # create baseline
    base_accuracy_1, base_error_1 = create_baseline(test_y_1)

    # create random forest regression model
    regr_1, regr_accuracy_1 = rr_model(train_x_1, test_x_1, train_y_1, test_y_1)
    # 
    # 90.18 accuracy; 0.0313 error


    # I might be able to improve model by hyperparameter tuning model

    # see current forest parameters
    regr_1.get_params()

    # use random hyperparameter grid
    from sklearn.model_selection import RandomizedSearchCV
    from pprint import pprint
    # num of trees
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=20)]
    # num of features to consider at every split
    max_features = ['auto', 'sqrt']
    # max num of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num= 11)]
    max_depth.append(None)
    # min num of samples required to split a node
    min_samples_split = [2, 5, 10]
    # min num of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # create random grid
    random_grid_1 = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    pprint(random_grid_1)

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    regr_random_1 = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=1, n_jobs=-1)
    regr_random_1.fit(train_x_1, train_y_1.values.ravel())

    # view best parameters
    regr_random_1.best_params_

    random_best_accuracy_1 = evaluate(regr_random_1, test_x_1, test_y_1)


    # use GridSearchCV to explicitly specify combinations to try based on results from regr_random.best_params_
    from sklearn.model_selection import GridSearchCV

    # create parameter grid based on results of random search
    param_grid_1 = {'max_depth': [60, 70, 80, 90, 100],
                  'max_features': ['auto'],
                  'min_samples_leaf': [2, 3, 4, 5, 6],
                  'min_samples_split': [3, 4, 5, 6, 7],
                  'n_estimators': [200, 300, 400]}
    # create a based model
    rf_1 = RandomForestRegressor()
    grid_search_1 = GridSearchCV(estimator=rf_1, param_grid=param_grid_1, cv=5, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search_1.fit(train_x_1, train_y_1.values.ravel())
    grid_search_1.best_params_

    best_grid_1 = grid_search_1.best_estimator_
    grid_accuracy_1 = evaluate(best_grid_1, test_x_1, test_y_1)




    # Approach 2
    # Random Forest Regression on data using complex_df()

    df_2 = simple_df(df)

    # split data
    train_x_2, test_x_2, train_y_2, test_y_2 = split_simple_data(df_2)
    # create baseline
    base_accuracy_2, base_error_2 = create_baseline(test_y_2)

    # create random forest regression model
    regr_2, regr_accuracy_2 = rr_model(train_x_2, test_x_2, train_y_2, test_y_2)
    # 91.48 accuracy; 0.0263 error
    
    # I might be able to improve model by hyperparameter tuning model
    # use random hyperparameter grid
    
    # num of trees
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
    # num of features to consider at every split
    max_features = ['auto', 'sqrt']
    # max num of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num= 11)]
    max_depth.append(None)
    # min num of samples required to split a node
    min_samples_split = [2, 5, 10]
    # min num of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # create random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    pprint(random_grid)
    
    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    regr_random_2 = RandomizedSearchCV(estimator=regr_2, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=1, n_jobs=-1)
    regr_random_2.fit(train_x_2, train_y_2.values.ravel())
    
    # view best parameters
    regr_random_2.best_params_

    random_best_accuracy_2 = evaluate(regr_random_2, test_x_2, test_y_2)
    
    # use GridSearchCV to explicitly specify combinations to try based on results from regr_random.best_params_
    
    # create parameter grid based on results of random search
    param_grid = {'max_depth': [60, 70, 80, 90],
                  'max_features': ['auto'],
                  'min_samples_leaf': [3, 4, 5],
                  'min_samples_split': [2, 3, 4],
                  'n_estimators': [850, 950, 1050]}

    grid_search_complex_2 = GridSearchCV(estimator=regr_2, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    
    # Fit the grid search to the data
    grid_search_complex_2.fit(train_x_2, train_y_2.values.ravel())
    grid_search_complex_2.best_params_
    
    best_grid_complex_2 = grid_search_complex_2.best_estimator_
    grid_accuracy_2 = evaluate(best_grid_complex_2, test_x_2, test_y_2)
    # 91.66% accuracy, 0.0258 points error



    
    # Approach 3
    # involved combining features per pitch type to create a "pitch score" as the input features for Linear Regression.
    # this involved using data from complex_df function

    df_3 = complex_df(df)
    
    # standardize variables
    df_3 = standardized_complex_df(df_3)
    
    # check correlations of df from standardized_complex_df()
    corr_model = df_3.corr()
    
    # need to create a single feature per pitch type
    # each pitch type will be some variation of:
    # pitch_score = pitch_percent * (spin_rate * x1 + effective_speed * x2 + break_z * x3 + break_x * x4)
    # x values are based on correlation of variable with xwoba per pitch type (summation of abs(x) values equals 1)
    corr_xwoba = np.array([[-0.26420, -0.24806, 0.22962, -0.02282],
                  [-0.23614, -0.26535, 0.01287, -0.13379],
                  [-0.10610, -0.17348, 0.10354, -0.19249],
                  [-0.15349, -0.13444, 0.06987, -0.09544],
                  [-0.18391, -0.14302, 0.03514, -0.10084],
                  [-0.20793, -0.04600, -0.00349, -0.18731],
                  [-0.15859, 0.21755, -0.11235, 0.30157]])
    
    # get percent of xwoba corr per pitch type
    x1 = [corr_xwoba[0][0] / sum(np.abs(corr_xwoba[0])), corr_xwoba[0][1] / sum(np.abs(corr_xwoba[0])), corr_xwoba[0][2] / sum(np.abs(corr_xwoba[0])), corr_xwoba[0][3] / sum(np.abs(corr_xwoba[0]))]
    x2 = [corr_xwoba[1][0] / sum(np.abs(corr_xwoba[1])), corr_xwoba[1][1] / sum(np.abs(corr_xwoba[1])), corr_xwoba[1][2] / sum(np.abs(corr_xwoba[1])), corr_xwoba[1][3] / sum(np.abs(corr_xwoba[1]))]
    x3 = [corr_xwoba[2][0] / sum(np.abs(corr_xwoba[2])), corr_xwoba[2][1] / sum(np.abs(corr_xwoba[2])), corr_xwoba[2][2] / sum(np.abs(corr_xwoba[2])), corr_xwoba[2][3] / sum(np.abs(corr_xwoba[2]))]
    x4 = [corr_xwoba[3][0] / sum(np.abs(corr_xwoba[3])), corr_xwoba[3][1] / sum(np.abs(corr_xwoba[3])), corr_xwoba[3][2] / sum(np.abs(corr_xwoba[3])), corr_xwoba[3][3] / sum(np.abs(corr_xwoba[3]))]
    x5 = [corr_xwoba[4][0] / sum(np.abs(corr_xwoba[4])), corr_xwoba[4][1] / sum(np.abs(corr_xwoba[4])), corr_xwoba[4][2] / sum(np.abs(corr_xwoba[4])), corr_xwoba[4][3] / sum(np.abs(corr_xwoba[4]))]
    x6 = [corr_xwoba[5][0] / sum(np.abs(corr_xwoba[5])), corr_xwoba[5][1] / sum(np.abs(corr_xwoba[5])), corr_xwoba[5][2] / sum(np.abs(corr_xwoba[5])), corr_xwoba[5][3] / sum(np.abs(corr_xwoba[5]))]
    x7 = [corr_xwoba[6][0] / sum(np.abs(corr_xwoba[6])), corr_xwoba[6][1] / sum(np.abs(corr_xwoba[6])), corr_xwoba[6][2] / sum(np.abs(corr_xwoba[6])), corr_xwoba[6][3] / sum(np.abs(corr_xwoba[6]))]
    x = [x1, x2, x3, x4, x5, x6, x7]
    
    for i in range(len(complex_pitch_types)):
        df_3['%s_pitch_score' % complex_pitch_types[i]] = (df_model['%s_spin_rate' % complex_pitch_types[i]] * x[i][0] +
                                                               df_model['%s_effective_speed' % complex_pitch_types[i]] * x[i][1] +
                                                               df_model['%s_pitcher_break_z' % complex_pitch_types[i]] * x[i][2] +
                                                               df_model['%s_pitcher_break_x' % complex_pitch_types[i]] * x[i][3])
    
    # add pitch scores for total score
    # first replace nan with 0
    df_3 = df_3.fillna(0)
    
    # a negative score is currently better so multiply by -1 for ease of understanding
    for i in range(len(complex_pitch_types)):
        df_3['%s_pitch_score' % complex_pitch_types[i]] = (df_model['%s_pitch_score' % complex_pitch_types[i]] * (-1))
    
    
    # calculate total pitch score by multiplying each pitch score by it's thrown percentage
    df_3['total_pitch_score_w_pitch_percent'] = (df_3['4_seam_fb_pitch_score'] * df_3['4_seam_fb_pitch_percent'] +
                                     df_3['sinker_pitch_score'] * df_3['sinker_pitch_percent'] +
                                     df_3['cut_fb_pitch_score'] * df_3['cut_fb_pitch_percent'] +
                                     df_3['changeup_pitch_score'] * df_3['changeup_pitch_percent'] +
                                     df_3['curveball_pitch_score'] * df_3['curveball_pitch_percent'] +
                                     df_3['slider_pitch_score'] * df_3['slider_pitch_percent'] +
                                     df_3['split_finger_pitch_score'] * df_3['split_finger_pitch_percent'])
    
    df_3['total_pitch_score'] = (df_3['4_seam_fb_pitch_score'] + df_3['sinker_pitch_score'] +
                                 df_3['cut_fb_pitch_score'] + df_3['changeup_pitch_score'] +
                                 df_3['curveball_pitch_score'] + df_3['slider_pitch_score'] +
                                 df_3['split_finger_pitch_score'])
    
    # recheck correlation
    corr_model = df_3.corr()
    
    # visualization added to paper
    """
    sns.pairplot(df_3, vars=['xwoba', 'total_pitch_score', 'total_pitch_score_w_pitch_percent'])
    sns.show()
    """
     
    # time to train a model
    train_model_features_3, test_model_features_3 = train_test_split(df_3, test_size=0.33, random_state=1)
    train_x_3 = train_model_features_3[['pitch_hand', 'total_pitch_score_w_pitch_percent']]
    test_x_3 = test_model_features_3[['pitch_hand', 'total_pitch_score_w_pitch_percent']]
    train_y_3 = train_model_features_3[['xwoba']]
    test_y_3 = test_model_features_3[['xwoba']]
    
    
    # linear regression model
    from sklearn.linear_model import LinearRegression
    lin_reg_model_3 = LinearRegression()
    lin_reg_model_3.fit(train_x_3, train_y_3)
    #y_pred = lin_reg_model.predict(test_x)
    # check results
    lin_reg_accuracy_3 = evaluate(lin_reg_model_3, test_x_3, test_y_3)
    # 91.80 accuracy
    # currently best model

        
    # calculate pred_xwoba for all players
    # find 10 largest overperformers and underperformers based on the linear regression model
    find_overperformers(df_3, lin_reg_model_3, 10)
    find_underperformers(df_3, lin_reg_model_3, 10)
    
    # view min, max, and mean predicted xwoba
    df_3.pred_xwoba.min()
    df_3.pred_xwoba.max()
    df_3.pred_xwoba.mean()
