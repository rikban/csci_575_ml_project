import numpy as np
import glob
import os
import pprint as pp
import matplotlib.pyplot as plt
import collections

class Portfolio:
    def __init__(self, starting_cash, starting_units_of_dow, current_price):
        self.cash = starting_cash
        self.units_of_dow = starting_units_of_dow
        self.previous_networth = self.cash + (self.units_of_dow*current_price)
        self.portfolio_balance = self.previous_networth

    def reComputePortfolioBalance(self, current_price):
        self.portfolio_balance = self.cash + (self.units_of_dow * current_price)
        print("I now have ",self.units_of_dow, " stocks, and ", self.portfolio_balance, " in cash")

    def Buy(self, units, current_price):
    # Need to check for availability of enough cash
        if self.cash >= units*current_price:
            self.cash -= units*current_price
            self.units_of_dow += units
            self.reComputePortfolioBalance(current_price)

    def Sell(self, units, current_price):
    # Need to check for availability of enough units to sell
        if self.units_of_dow > units:
            self.cash += units * current_price
            self.units_of_dow -= units
            self.reComputePortfolioBalance(current_price)


def main():
    global episode_number
    global iterations

    while iterations < processed_data["Date"].shape[0]:
        previous_state = stateEstimator(processed_data, iterations)
        if portfolio.portfolio_balance <= 0 or portfolio.units_of_dow <= 0:
            # Report progress made and reset to start of data file
            episode_number += 1
            iteration_deque.appendleft(iterations)
            iterations = 0

            if len(iteration_deque) > 10:
                iteration_deque.pop()

            print("State Abort - Resetting")
            print("Iteration: ", iterations)
        print("The last 10 sims have run for ", iteration_deque)
        print("Episode number: ", episode_number)

        action = selectActionUsingEpsilonGreedyPolicy(Q_table=Q_table,
                                                      state=previous_state)
        print("Iterations: ", iterations)
        reward = executeActionAndCalculateReward(action)
        state = stateEstimator(processed_data=processed_data, iterations=iterations)
        updateQTable(reward=reward, state=state, action=action, previous_state=previous_state)
        portfolio.previous_networth = portfolio.portfolio_balance
        pp.pprint(Q_table)
        iterations = iterations + 1
    print("End of file")


def executeActionAndCalculateReward(action):
    global iterations

    # Execute action
    if action == 0:  # buy
        success = portfolio.Buy(units=1, current_price=processed_data['Price'][iterations])
    elif action == 1:  # sell
        success = portfolio.Sell(units=1, current_price=processed_data['Price'][iterations])
    elif action == 2:  # hold
        pass

    # Calculate reward
    if portfolio.previous_networth > portfolio.portfolio_balance:
        reward = -1

    elif portfolio.previous_networth == portfolio.portfolio_balance:
        reward = 0

    else:
        reward = 1

    return reward


def selectActionUsingEpsilonGreedyPolicy(Q_table, state):
    # Define parameters
    epsilon_0 = 0.9
    d = 0.985

    if state not in Q_table:
        print("New state observed, appending ", state, " to Q table")
        Q_table[state] = [0.0, 0.0, 0.0]  # Buy, Sell, and Hold

    # Generate random number
    r = np.random.uniform(low=0.0, high=1.0)
    epsilon = epsilon_0 * pow(d, episode_number)
    print("Epsilon: ", epsilon)

    if r > epsilon:
        print("Selecting Q Table action")
        action = actionSelectionUsingQTable(state=state)
    else:
        print("Selecting random action")
        action = selectRandomAction()

    return action

def selectRandomAction():
    # Generate random number
    r = np.random.uniform(low=0.0, high=1.0)
    action = ''

    if r <= (1.0 / 3.0):
        print("Buying")
        action = 0

    elif r > 1.0 / 3.0 and r <= 2.0 / 3.0:
        print("Selling")
        action = 1

    elif r > 2.0 / 3.0 and r <= 3.0 / 3.0:
        print("Holding")
        action = 2

    return action


def actionSelectionUsingQTable(state):
    action_row = Q_table[state]
    action = action_row.index(max(action_row))
    if action == 0:
        print("Buying")
        action = 0

    elif action == 1:
        print("Selling")
        action = 1

    elif action == 2:
        print("Holding")
        action = 2

    return action


def stateEstimator(processed_data, iterations):
    # Use Price and daily Change of the Dow to define states

    if (processed_data["Price"][iterations] - processed_data["Price"][iterations - 1]) > 0:
        state_pl = 2
    else:
        state_pl = 1

    if (processed_data["High"][iterations] - processed_data["Low"][iterations]) > 120:
        state_daily_swing = 2
    else:
        state_daily_swing = 1

    if processed_data["Change"][iterations] > 0:
        state_change = 2
    else:
        state_change = 1

    if processed_data["Volume"][iterations] > 200:
        state_volume = 2
    else:
        state_volume = 1

    cumulative_state = str(1000*state_pl + 100*state_daily_swing + 10*state_change + 1*state_volume)
    return cumulative_state


def updateQTable(reward, state, action, previous_state):
    # Define parameters
    gamma = 0.8
    alpha = 0.3

    if state not in Q_table:
        print("New state observed, appending ", state, " to Q table")
        Q_table[state] = [0.0, 0.0, 0.0]  # Buy, Sell, Hold

    previous_q_table_entry = Q_table[previous_state][action]
    Q_table[previous_state][action] = Q_table[previous_state][action] + alpha * (
                reward + (gamma * max(Q_table[state])) - Q_table[previous_state][action])
    print("Training Residual: ", (Q_table[previous_state][action] - previous_q_table_entry))



def dataPreprocessing():
    # Convert ndarray of strings to a dictionary of floats
    end_index = -503  # 503 days from Jan 8th 2009. Avoids the volatility of the 2008 crisis
    processed_data["Date"] = dow_data[1:end_index, 0].astype(str)
    processed_data["Price"] = dow_data[1:end_index, 1].astype(float)
    processed_data["Open"] = dow_data[1:end_index, 2].astype(float)
    processed_data["High"] = dow_data[1:end_index, 3].astype(float)
    processed_data["Low"] = dow_data[1:end_index, 4].astype(float)
    processed_data["Volume"] = dow_data[1:end_index, 5].astype(float)
    processed_data["Change"] = dow_data[1:end_index, 6].astype(float)


if __name__=="__main__":

    key_list = ["Date", "Price", "Open", "High", "Low", "Volume", "Change"]
    processed_data = dict.fromkeys(key_list)
    filename = "data.csv"

    with open(filename, "r") as f:
        dow_data = np.genfromtxt(f, dtype='str', delimiter=',', invalid_raise=False, usecols=np.arange(0, 7))
        dataPreprocessing()
        print("Dow Data shape: ", dow_data.shape)
        print("Date shape: ", processed_data["Date"].shape)

        print("Num rows: ", processed_data["Date"].shape[0])
    portfolio = Portfolio(starting_cash=(processed_data["Price"][0]), starting_units_of_dow=1,
                          current_price=processed_data["Price"][0])
    Q_table = {}
    iteration_deque = collections.deque()
    episode_number = 0
    iterations = 0
    main()
