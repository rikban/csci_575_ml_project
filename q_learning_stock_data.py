import numpy as np
import glob
import os
import pprint as pp
import matplotlib.pyplot as plt

def main():
    filename = "data.csv"
    test_path = "/home/hcr-student/hcr/human_centered_robotics/project3/dataset/test"
    key_list = ["Date", "Price", "Open", "High", "Low", "Volume", "Change"]
    processed_data = dict.fromkeys(key_list)
    Q_table = {}
    with open(filename, "r") as f:
      dow_data = np.genfromtxt(f, dtype='str', delimiter=',', invalid_raise=False, usecols=np.arange(0, 7))
      dataPreprocessing(dow_data, processed_data)
      print("Dow Data shape: ", dow_data.shape)
      print("Date shape: ", processed_data["Date"].shape)
     #pp.pprint(processed_data)
      print("Num rows: ", processed_data["Date"].shape[0])
    for i in range(processed_data["Date"].shape[0]):
     state = stateEstimator(processed_data, i)



def stateEstimator(processed_data, i):
    # Use Price and daily Change of the Dow to define states
    if (processed_data["Price"][i] - processed_data["Price"][i-1]) > 0:
        state_pl = 2
    else:
        state_pl = 1

    if (processed_data["High"][i] - processed_data["Low"][i]) > 120:
       state_daily_swing = 2
    else:
       state_daily_swing = 1
    if processed_data["Change"][i] > 0:
        state_change = 2
    else:
        state_change = 1
    if processed_data["Volume"][i] > 200:
        state_volume = 2
    else:
        state_volume = 1
    cumulative_state = str(1000*state_pl + 100*state_daily_swing + 10*state_change + 1*state_volume)
    return cumulative_state


def calculateReward(state):
    if state[0] == '2':
        reward = 0
    else:
        reward = -1

    return reward

def updateQTable(Q_table, state, action, previous_state):
    # Define parameters
    gamma = 0.8
    reward = calculateReward(state)
    #global episode_number
    global accumulated_rewards
    alpha = 0.3

    if state not in Q_table:
        print("New state observed, appending ", state, " to Q table")
        Q_table[state] = [0.0, 0.0, 0.0]  # Buy, Sell, Hold

    previous_q_table_entry = Q_table[previous_state][action]
    Q_table[previous_state][action] = Q_table[previous_state][action] + alpha * (
                reward + (gamma * max(Q_table[state])) - Q_table[previous_state][action])
    print("Training Residual: ", (Q_table[previous_state][action] - previous_q_table_entry))
    accumulated_rewards = accumulated_rewards + reward


def dataPreprocessing(dow_data, processed_data):
    # Convert ndarray of strings to a dictionary of floats
    end_index = -503 # 503 days from Jan 8th 2009. Avoids the volatility of the 2008 crisis
    processed_data["Date"] = dow_data[1:end_index, 0].astype(str)
    processed_data["Price"] = dow_data[1:end_index, 1].astype(float)
    processed_data["Open"] = dow_data[1:end_index, 2].astype(float)
    processed_data["High"] = dow_data[1:end_index, 3].astype(float)
    processed_data["Low"] = dow_data[1:end_index, 4].astype(float)
    processed_data["Volume"] = dow_data[1:end_index, 5].astype(float)
    processed_data["Change"] = dow_data[1:end_index, 6].astype(float)

    # From 2010 to 2019
    pass

def getState(dow_data):
    pass


if __name__=="__main__":
    main()
