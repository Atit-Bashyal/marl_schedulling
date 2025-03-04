import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import utilities
from scipy.interpolate import CubicSpline


price_signal = [107.21, 101.1, 96.73, 82.34, 81.93, 84.03, 97.25, 106.51, 
                120.7, 116.86,112.85, 113.37, 115, 109.78, 119.71, 125.83, 
                139.57, 155.93, 163.5,146.74, 139.38, 131.8, 129.94,121.3,121.3]


utilities.plot_value_vs_step(
    '/Users/sakshisharma/Desktop/rl_scheduling/single_agent_PPO_Training_Experiment_PPO_SingleAgentEnvironment.csv',
    '/Users/sakshisharma/Desktop/rl_scheduling/env1_PPO_Training_Experiment_PPO_MRLEnvironment.csv',
    '/Users/sakshisharma/Desktop/rl_scheduling/attention_PPO_Training_Experiment_PPO_MRLEnvironment.csv')


utilities.plot_comparative_cost(['/Users/sakshisharma/Desktop/rl_scheduling/total_cost_singleenv.csv',
                                 '/Users/sakshisharma/Desktop/rl_scheduling/total_cost_env1.csv',
                                 'total_cost_attention.csv'],price_signal[:-1],
                                 ['PPO','MAPPO','Proposed algorithm'])





operating_points_single = utilities.convert_csv_to_list2('/Users/sakshisharma/Desktop/rl_scheduling/actions_test.csv')
operating_points_marl = utilities.convert_csv_to_list('/Users/sakshisharma/Desktop/rl_scheduling/actions_test_multiagent.csv')
operating_points_prop = utilities.convert_csv_to_list('/Users/sakshisharma/Desktop/rl_scheduling/actions_test_attention.csv')



dict={}
for i in range(len(operating_points_single)):
    dict[i] = {'single': operating_points_single[i],
               'milti': operating_points_marl[i],
               'proposed':operating_points_prop[i] }

# Convert to DataFrame
df = pd.DataFrame.from_dict(dict, orient='index')

# Expand each list into subcolumns
df_expanded = pd.concat(
    [df[col].apply(pd.Series).add_prefix(f"{col}_") for col in df.columns], 
    axis=1
)
df_expanded.to_excel('actions.xlsx')


