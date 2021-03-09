import pandas as pd
import numpy as np
import random

class Env:
	def __init__(self, data_path):

		self.fee = 0.001

		self.dataframe = pd.DataFrame()
		self.dataframe = pd.read_csv(data_path)

		self.prices = self.dataframe["close"].values
		self.opens = self.dataframe["open"].values
		self.lows = self.dataframe["low"].values
		self.highs = self.dataframe["high"].values
		self.volumes = self.dataframe["volume"].values

		self.state_size_param = 4
		self.action_size_param = 3
		self.window_size_param = 120

		self.step_param = 0

	def random_act(self):

		return np.random.randint(0,3)

	def reset(self):

		self.step_param = 0

		state = self.prices[self.step_param:self.window_size_param+self.step_param]

		self.sell_btc_price = state[-1]

		next_price_state = self.prices[self.step_param:self.window_size_param+self.step_param]
		next_volume_state = self.volumes[self.step_param:self.window_size_param+self.step_param]
		next_low_state = self.lows[self.step_param:self.window_size_param+self.step_param]
		next_high_state = self.highs[self.step_param:self.window_size_param+self.step_param]

		next_price_state = (next_price_state - np.amin(next_price_state)) / (np.amax(next_price_state) - np.amin(next_price_state))
		next_volume_state = (next_volume_state - np.amin(next_volume_state)) / (np.amax(next_volume_state) - np.amin(next_volume_state))
		next_low_state = (next_low_state - np.amin(next_low_state)) / (np.amax(next_high_state) - np.amin(next_low_state))
		next_high_state = (next_high_state - np.amin(next_low_state)) / (np.amax(next_high_state) - np.amin(next_low_state))
		
		next_state = [next_price_state,next_volume_state,next_low_state,next_high_state]

		return np.array(next_state)


	def buy_reward(self,now_price):

		p_mean = self.prices[self.window_size_param+self.step_param:self.window_size_param+self.step_param+6]

		mean = np.mean(p_mean)

		if (now_price*(1+self.fee)) < mean:
			return 1
		else:
			return -1

	def sell_reward(self,now_price):

		p_mean = self.prices[self.window_size_param+self.step_param:self.window_size_param+self.step_param+6]

		mean = np.mean(p_mean)

		if (now_price*(1-self.fee)) > mean:
			return 1
		else:
			return -1

	def step(self, action):
		price = self.prices[self.window_size_param+self.step_param]
		last_price = self.prices[self.window_size_param+self.step_param - 1]
		
		rew = 0

		done = False

		if action == 0:
			rew = self.buy_reward(price)
		elif action == 1:
			rew = self.sell_reward(price)
		else:
			rew = -0.1
		

		self.step_param += 1

		if (self.window_size_param+self.step_param+7) >= len(self.prices):
			done = True

		next_price_state = self.prices[self.step_param:self.window_size_param+self.step_param]
		next_volume_state = self.volumes[self.step_param:self.window_size_param+self.step_param]
		next_low_state = self.lows[self.step_param:self.window_size_param+self.step_param]
		next_high_state = self.highs[self.step_param:self.window_size_param+self.step_param]

		next_price_state = (next_price_state - np.amin(next_price_state)) / (np.amax(next_price_state) - np.amin(next_price_state))
		next_volume_state = (next_volume_state - np.amin(next_volume_state)) / (np.amax(next_volume_state) - np.amin(next_volume_state))
		next_low_state = (next_low_state - np.amin(next_low_state)) / (np.amax(next_high_state) - np.amin(next_low_state))
		next_high_state = (next_high_state - np.amin(next_low_state)) / (np.amax(next_high_state) - np.amin(next_low_state))
		
		next_state = [next_price_state,next_volume_state,next_low_state,next_high_state]
		

		return np.array(next_state), rew, done




