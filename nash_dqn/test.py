import torch
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from NashRL import *
from model import *
from NashAgent_lib import *
from textwrap import wrap
from visualization import *

# Initialize a dummy agent
nash_agent = NashNN(input_dim=2 + num_players, output_dim=4, nump=num_players, t=15, t_cost=.1, term_cost=.1,
                    num_moms=5)

# Load saved network parameters from file
net_file_name = "Action_Net"
nash_agent.action_net.load_state_dict(torch.load(net_file_name))
nash_agent.action_net.eval()

# Output Heatmap when other agent's average inventory is low
heatmap_old(net=nash_agent, t_step=15, q_step=50, p_step=5, t_range=[0, 14],
            q_range=[-25, 25], p_range=[6, 14], nump=num_players, other_agent_inv=-20)

# Output Heatmap when other agent's average inventory is zero
heatmap_old(net=nash_agent, t_step=15, q_step=50, p_step=5, t_range=[0, 14],
            q_range=[-25, 25], p_range=[6, 14], nump=num_players, other_agent_inv=0)

# Output Heatmap when other agent's average inventory is zero
heatmap_old(net=nash_agent, t_step=15, q_step=50, p_step=5, t_range=[0, 14],
            q_range=[-25, 25], p_range=[6, 14], nump=num_players, other_agent_inv=20)

# Generates fixed paths (fixing starting inventory levels horizontally or fixing price paths vertically)
seed = 33333
np.random.seed(seed)
fixed_sample_paths(nash_agent, 9, num_players, 15, sim_dict, seed)
