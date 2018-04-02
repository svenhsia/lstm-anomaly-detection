import os
import pandas as pd
import numpy as np


def eventGeneration(path, state_events):
    """ 
        Combine files of each state event into one file. 
        Extract 4 useful attributes. 
    """
    cols = ['timestamp', 'nodeNumber', 'eventType', 'value']
    for event in state_events:        # one event file containing several recording files
        df_event = pd.DataFrame(columns=cols)     # create a dataframe for one event
        for filename in os.listdir(path + '2015/singleTypes-all-chunks2/' + event + '/'):       # one recording file in this event file
            df = pd.read_csv(path + '2015/singleTypes-all-chunks2/' + event + '/' + filename)
            df_event = pd.concat([df_event, df.loc[:,cols]])        # append information in this recording file to the dataframe for this event
        print (event, df_event.shape)
        df_event.to_csv(path + 'combined events/' + event + ".csv", index=False)


def eventCombination(path):
    """ 
        Combine all state event into one big file. 
    """
    cols = ['timestamp', 'nodeNumber', 'eventType', 'value']
    df_all = pd.DataFrame(columns=cols)         # create a dataframe for all events
    for file in os.listdir(path + 'combined events/'):
            df_event = pd.read_csv(path + 'combined events/' + file)        # load one event file
            df_all= pd.concat([df_all, df_event])         # append to the dataframe
    print ('Feature table\'s shape ', df_all.shape)
    df_all = df_all.sort_values(by=['timestamp', 'nodeNumber'])         # sort by timestamp and nodeNumber
    df_all.loc[:, 'timestamp'] = (df_all.loc[:, 'timestamp'] - 1427515215000) / float(15000)        # transform timestamp
    df_all['timestamp'] = df_all['timestamp'].astype(int)
    df_all.to_csv(path + 'all.csv', index=False)


def featureNameGeneration(path):
    """ 
        Determine features' names from the 
        combinations of nodeNumber and eventType. 
    """
    df_all = pd.read_csv(path + 'all.csv')
    sample = df_all[df_all.timestamp==0]
    sample = sample.sort_values(['nodeNumber', 'eventType'])
    feature_names = ['timestamp']
    for i in range(sample.shape[0]):
        node = sample.iloc[i,1]
        event = sample.iloc[i,2]
        feature = 'node_' + str(node) + '_' + event         # feature is the combination of node and event
        feature_names.append(feature)
    with open(path + "feature_names.txt", "w") as output:
        output.write(str(feature_names))
    assert len(feature_names) == 301


def featureDistribution(path):
    """ 
        Count the number of features for each timestamp. 
        Calculate the counts distributions. 
    """
    df_all = pd.read_csv(path + 'all.csv')
    ts_counts = pd.DataFrame(df_all['timestamp'].value_counts())
    ts_counts.columns = ['count']
    print ('\nDistribution of feature number each timestamp')
    print (ts_counts['count'].value_counts(ascending=False))


def timestampFiltration(path):
    """ 
        Remove timestamps with missing features. 
    """
    df_all = pd.read_csv(path + 'all.csv')
    ts_counts = pd.DataFrame(df_all['timestamp'].value_counts())
    ts_counts.columns = ['count']
    ts_to_delete = ts_counts.loc[ts_counts['count']<300].index.tolist()
    df300 = df_all.copy().set_index('timestamp')
    df300.drop(ts_to_delete, inplace=True)
    df300.to_csv(path + 'all_300.csv',index=True)


def timestampDistribution(path):
    """
        The timestamps are not continuous. 
        Calculate the distribution of timestamps.
        Find the continuous periods.
    """
    df_all = pd.read_csv(path + 'all_300.csv')
    ts = list(set(df_all.loc[:, 'timestamp']))
    t_starts = [0]
    t_ends = []
    t_prev = 0
    t_start = 0
    t_end = -1
    for t in ts[1:]:
        if t != t_prev + 1:
            t_end = t_prev
            t_ends.append(t_end)
            t_start = t
            t_starts.append(t_start)
        t_prev = t
    t_ends.append(ts[len(ts)-1])
    assert len(t_starts) == len(t_ends)
    print ('\nPeriods info')
    periods = []
    period_length = []
    for i in range(len(t_starts)):
        print ('consecutive period (%d, %d) with length %d' % (t_starts[i],t_ends[i],t_ends[i]-t_starts[i]))
        periods.append((int(t_starts[i]), int(t_ends[i])))
        period_length.append(t_ends[i]-t_starts[i])
    for elt in np.sort(list(set(period_length))):
        cnt = period_length.count(elt)
        print ('there are %d periods with length %d'%(cnt,elt))


def featureTableCreation(path):
    """
        Create an empty feature table indexed by timestamp 
        and using features as columns.
    """
    print ('Creation starts !')
    df_all = pd.read_csv(path + 'all_300.csv')
    with open(path + "feature_names.txt", "r") as input:
        feature_names = input.read()
    ft = pd.DataFrame(columns=feature_names)
    ts = list(set(df_all.timestamp))      # choose all timestamps (unique)
    for i in range(len(ts)):
        ft.loc[i] = [np.NaN for i in range(301)]        # fill in with NaN rows
        ft.iloc[i,0] = ts[i]        # fill in the timestamp
    ft.to_csv(path + 'feature_table.csv', index=False)
    print ('Creation finished !')
    print ('Empty feature table shape ', ft.shape)


def featureTableFilling(path):
    """
        Fill in the empty feature table created before. 
        Transform information into timestamp vectors.
    """
    print ('Transformation starts !')
    chunks = pd.read_csv(path + 'all_300.csv', iterator=True, chunksize=20000)
    for df in chunks:
        ft = pd.read_csv(path + 'feature_table.csv')
        ft.set_index('timestamp', inplace=True)
        for i in range(df.shape[0]):
            timestamp = df.iloc[i,0]
            node = df.iloc[i,1]
            event = df.iloc[i,2]
            value = df.iloc[i,3]
            feature_name = 'node_' + str(node) + '_' + event
            ft.loc[timestamp, feature_name] = value
        ft.to_csv(path + 'feature_table.csv', index=True)
    print ('Transformation finished !')


def missingVectorPrediction(path):
    """
        The timestamps are not continuous.
        Predict the missing vectors using their neighbors.
    """
    ft = pd.read_csv(path + 'feature_table.csv')
    ft.set_index('timestamp', inplace=True)
    ts = ft.index
    for i in ts:
        if i+1 not in ts:
            if i+2 in ts:
                ft.loc[i+1, :] = (ft.loc[i, :] + ft.loc[i+2, :]) / 2
            elif i+3 in ts:
                ft.loc[i+1, :] = (ft.loc[i, :] + ft.loc[i+3, :]) * 0.33
                ft.loc[i+2, :] = (ft.loc[i, :] + ft.loc[i+3, :]) * 0.67
    ft.sort_index(inplace=True)
    ft.to_csv(path + 'final_feature_table.csv', index=True)


def periodInfo(path):
    """
        Periods' information after the missing vector prediction.
    """
    ft = pd.read_csv(path + 'final_feature_table.csv')
    ft.set_index('timestamp', inplace=True)
    periods = []
    broken = True
    print ('\nperiods info')
    for i in ft.index:
        if broken:
            start = i
            broken = False
        else:
            end = i
        if i+1 not in ft.index:
            periods.append((start, end))
            broken = True
            print("continuous between ({}, {}) for {} timesteps".format(start, end, end-start+1))


def abnormalTreatment(path):
    """
        Timestamp treatement for abnormal file.
    """
    abnormal = pd.read_csv(path + 'abnormal.csv')
    abnormal['abnormalStart'] = abnormal['abnormalStart'].apply(lambda x : int((x - 1427515215000) / float(15000) + 1))
    abnormal['abnormalEnd'] = abnormal['abnormalEnd'].apply(lambda x : int((x - 1427515215000) / float(15000)))
    abnormal.to_csv('../data/abnormal_treated.csv', index=False)





# Set the path for data. 
# The extracted 2015 data file should be under this path.
# A directory called 'combined events' should be created under this path before running this scripy.
# The file of ground truth information should also be under this path.
path = '../data/'
# Choose state events used in our project.
state_events = ['load_one_event', 'mem_total_event', 'swap_free_event', 'cpu_speed_event', 'mem_buffers_event', 'cpu_user_event', 'proc_total_event', 'part_max_used_event', 'mem_shared_event', 'balance','cpu_idle_event', 'mem_free_event', 'disk_free_event', 'swap_total_event', 'balance-standarddeviation', 'load_fifteen_event','balance-numeric', 'proc_run_event', 'bytes_in_event', 'load_five_event', 'pkts_out_event', 'pkts_in_event', 'mem_cached_event', 'disk_total_event', 'cpu_system_event', 'cpu_aidle_event', 'bytes_out_event', 'cpu_num_event', 'cpu_nice_event', 'cpu_wio_event']
# With all these 11 steps, we will obtain a feature table and a treated file for ground truth.
eventGeneration(path, state_events)
eventCombination(path)
featureNameGeneration(path)
featureDistribution(path)
timestampFiltration(path)
timestampDistribution(path)
featureTableCreation(path)
featureTableFilling(path)
missingVectorPrediction(path)
periodInfo(path)
abnormalTreatment(path)

