

'''
This script exctracts training variables from all logs from 
tensorflow event files ("event*"), writes them to Pandas 
and finally stores in long-format to a CSV-file including
all (readable) runs of the logging directory.
The magic "5" infers there are only the following v.tags:
[lr, loss, acc, val_loss, val_acc]
'''

import tensorflow as tf
import glob
import os
import pandas as pd
from pathlib import Path

# Get all event* runs from logging_dir subdirectories
logging_dir = Path('./models') / 'simple_push/b/run1/logs/agent0/losses/'
event_paths = glob.glob(os.path.join(logging_dir,'*',"event*"))

# Extraction function
def sum_log(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    try:
        for e in tf.train.summary_iterator(path):
            for v in e.summary.value:
                r = {'metric': v.tag, 'value':v.simple_value}
                runlog = runlog.append(r, ignore_index=True)
    
    # Dirty catch of DataLossError
    except:
        print('Event file possibly corrupt: {}'.format(path))
        return None

    #runlog['epoch'] = [item for sublist in [[i]*5 for i in range(0, len(runlog)//5)] for item in sublist]
    
    return runlog

# Call & append
all_log = pd.DataFrame()


log = sum_log(event_paths[0])    # Change here to select stored data files
if log is not None:
    if all_log.shape[0] == 0:
        all_log = log
    else:
        all_log = all_log.append(log)

'''
for path in event_paths:
    log = sum_log(path)
    if log is not None:
        if all_log.shape[0] == 0:
            all_log = log
        else:
            all_log = all_log.append(log)
'''

# Inspect
print(all_log.shape)
all_log.head()    
            
# Store
all_log.to_csv(Path('./models') / 'CSV_Results/For_test.csv', index=None)




