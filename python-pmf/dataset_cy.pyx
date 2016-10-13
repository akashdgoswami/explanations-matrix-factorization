from __future__ import print_function # silly cython

import numpy as np
cimport numpy as np
import pandas as pd
import matplotlib as mpl
import argparse
import random

import pmf_cy as pmf
cimport pmf_cy as pmf

class Dataset(object):
	'''

	'''
	def __init__(self, dataset):
		self.name = dataset

		if(dataset == "movielens100k"):
			#import raw dataset in format : user id | item id | rating | timestamp
			self.original_data = pd.read_table(
				"../ml-100k/u.data",
				header=None,
				names=["user_id", "item_id", "rating", "timestamp"],
				engine='python'
				)
			#remove timestamp, which is not required
			self.original_data = self.original_data.drop('timestamp', axis=1)

			#get a numpy version
			self.ratings = self.original_data.as_matrix()

			#hardcode basic statistics
			self.num_users = 943
			self.num_items = 1682
			self.num_ratings = 100000
			self.genre_list = ["unknown", 
				"Action", "Adventure", 
				"Children", "Comedy", "Crime", 
				"Documentary", "Drama", 
				"Fantasy", "Film-Noir", 
				"Horror", 
				"Musical", "Mystery", 
				"Romance", 
				"Sci-Fi", 
				"Thriller", 
				"War", "Western"
				]

			#for item data column names
			item_data_names = ["id", "title", "release_date", "video_release_date", "IMDB_URL"]
			item_data_names.extend(self.genre_list)
			#index_col = False BIG HEADACHE
			self.item_data = pd.read_table(
				"../ml-100k/u.item",
				sep='|',
				header=None,
				names=item_data_names,
				index_col=False,
				engine='python'
				)
			#remove useless columns
			self.item_data = self.item_data.drop(["video_release_date", "IMDB_URL"], axis=1)
			
			#make a dict which maps movie names to item ids
			self.movie_to_item = dict(zip(self.item_data["title"], self.item_data["id"]))
			self.item_to_movie = dict(zip(self.item_data["id"], self.item_data["title"]))

			#make a dict which maps item id to its index in the numpy ndarray for fast access.
			#Each item will map tp several indices. Since each item is rated by several users
			#EXTREMELY BIG
			#make a dict which maps (user id, item id) pairs to its index in the numpy ndarray for fast access. Singleton values.
			self.item_to_index = {}
			self.user_item_to_index = {}
			for i in range(0,self.ratings.shape[0]):
				self.item_to_index.setdefault(self.ratings[i,1], []).append(i)
				self.user_item_to_index[(self.ratings[i,0], self.ratings[i,1])] = i

	'''
	Returns the ratings ndarray
	'''
	def get_ratings(self):
		return self.ratings

	'''
	The following functions return item id (movie name) corresponding to the movie name (item_id)
	'''
	def movie_name(self, item):
		if item > self.num_items:
			raise IndexError ("item out of range")
		#return self.item_data.loc[item-1]["title"]
		return self.item_to_movie[item]

	def item_number(self, movie):
		return self.movie_to_item[movie]

	def user_item_rating(self, user, item):
		return self.ratings[self.user_item_to_index[user, item], 2]

	def user_movie_rating(self, user, movie):
		return self.user_item_rating(user, self.item_number(movie))

	'''
	The following functions randomize the ratings matrix given (user id and) item id 
	'''
	def randomInterveneItem(self, item):
		new_ratings = self.ratings
		#cdef int i
		for i in range(0,self.ratings.shape[0]):
			if self.ratings[i,1] == item:
				new_ratings[i,2] = np.random.randint(1,6)
		print("Wave")
		return new_ratings

	def randomInterveneUserItem(self, user, item):
		new_ratings = self.ratings
		#if (user, item) in self.user_item_to_index:
		new_ratings[self.user_item_to_index[(user, item)], 2] = np.random.randint(1,6)
		return new_ratings

	'''
	The following functions randomize the ratings matrix given (user_id and) movie name.
	Simply get the item id corresponding to the movie from the dict and call the above functions.
	'''
	def randomInterveneMovie(self, movie):
		item = self.movie_to_item[movie]
		return self.randomInterveneItem(item)

	def randomInterveneUserMovie(self, user, movie):
		item = self.movie_to_item[movie]
		return self.randomInterveneUserItem(user, item)

	'''
	The following functions remove given item from the ratings matrix given user_id (movie name).
	'''
	def removeInterveneItem(self, item):
		new_ratings = self.ratings
		new_ratings = np.delete(new_ratings, self.item_to_index[item], axis=0)
		return new_ratings


	def removeInterveneMovie(self, movie):
		item = self.movie_to_item[movie]
		return self.removeInterveneItem(item)

	def printer(self):
		print(self.movie_to_item)

'''
Single (user,movie) Influence Measure
'''
def cell_influence(user, movie, target_user, target_movie):
	import pmf_cy as pmf

	dataset_object = Dataset("movielens100k")
	original_ratings = dataset_object.get_ratings()
	fit_type = ('mini-valid', 50, 50)

	pmf1 = pmf.ProbabilisticMatrixFactorization(original_ratings, latent_d=5, fit_type=fit_type)
	pmf1.do_fit()

	target_item = dataset_object.item_number(target_movie)
	pred1 = pmf1.prediction_for(target_user, target_item)

	iters = 4
	influence = 0
	for i in range(iters):
		random_ratings = dataset_object.randomInterveneUserMovie(user, movie)
		pmf2 = pmf.ProbabilisticMatrixFactorization(random_ratings, latent_d=5, fit_type=fit_type)
		pmf2.do_fit()
		pred2 = pmf2.prediction_for(target_user, target_item)
		influence = influence + pred2
	influence = influence/iters
	influence = abs(influence - pred1)

	print("Influence of user %d, movie %s on the rating of user %d, movie %s = %f" %(user, movie, target_user, target_movie, influence))

'''
Single movie Influence Measure
'''
def item_influence(movie, target_user, target_movie):
	import pmf_cy as pmf

	dataset_object = Dataset("movielens100k")
	original_ratings = dataset_object.get_ratings()
	fit_type = ('mini-valid', 10000, 10000)

	pmf1 = pmf.ProbabilisticMatrixFactorization(original_ratings, latent_d=5, fit_type=fit_type)
	pmf1.do_fit()

	target_item = dataset_object.item_number(target_movie)
	pred1 = pmf1.prediction_for(target_user, target_item)
	print("No randomization prediction: %f" %(pred1))

	iters = 4
	influence = 0
	for i in range(iters):
		random_ratings = dataset_object.randomInterveneMovie(movie)
		pmf2 = pmf.ProbabilisticMatrixFactorization(random_ratings, latent_d=5, fit_type=fit_type)
		pmf2.do_fit()
		pred2 = pmf2.prediction_for(target_user, target_item)
		print(pred2)
		iter_influence = abs(pred1 - pred2)
		influence = influence + iter_influence

	influence = influence/iters

	print("Influence of movie %s on the rating of user %d, movie %s = %f" %(movie, target_user, target_movie, influence))


def plotter(items, target_user, target_movie):
	import pmf_cy as pmf
	import matplotlib.pyplot as plt

	dataset_object = Dataset("movielens100k")
	original_ratings = dataset_object.get_ratings()
	fit_type = ('mini-valid', 10000, 10000)

	target_item = dataset_object.item_number(target_movie)

	pmf1 = pmf.ProbabilisticMatrixFactorization(original_ratings, latent_d=5, fit_type=fit_type)
	pmf1.do_fit()
	pred1 = pmf1.prediction_for(target_user, target_item)
	print("For movie %s" %(target_movie))
	print("Original prediction: %f" %(pred1))

	inf_series = {}
	#cdef int iters = 10
	#cdef float influence
	#cdef float iter_influence
	iters = 10
	for item in items:
		movie = dataset_object.movie_name(item)
		print("Randomizing %d: %s" %(item, movie))
		print("Randomized predictions:")
		influence = 0

		for i in range(iters):
			random.seed()
			np.random.seed()			
			random_ratings = dataset_object.randomInterveneItem(item)
			pmf2 = pmf.ProbabilisticMatrixFactorization(random_ratings, latent_d=5, fit_type=fit_type)
			pmf2.do_fit()
			pred2 = pmf2.prediction_for(target_user, target_item)

			print("Prediction: %f" %(pred2))

			iter_influence = abs(pred1 - pred2)
			influence = influence + iter_influence

		influence = influence/iters
		inf_series[movie] = influence
		print("Influence: %f" %(influence))

	plt.figure(figsize=(10,8))
	influence_series = pd.Series(inf_series, index=inf_series.keys())
	influence_series.plot(kind="bar")
	plt.xticks(rotation=45, ha='right', size='small')
	plt.xlabel('Movie')
	plt.ylabel('Influence')
	plt.tight_layout()
	plt.show()

def converge(movie, target_user, target_movie, threshold=0.0):
	dataset_object = Dataset("movielens100k")
	original_ratings = dataset_object.get_ratings()
	fit_type = ('mini-valid', 10000, 10000)

	target_item = dataset_object.item_number(target_movie)

	pmf1 = pmf.ProbabilisticMatrixFactorization(original_ratings, latent_d=5, fit_type=fit_type)
	pmf1.do_fit()
	pred1 = pmf1.prediction_for(target_user, target_item)
	print("For movie %s" %(target_movie))
	print("Original prediction: %f" %(pred1))
	print("Randomizing")
	print(".")

	iters = 0
	prev = 0
	diff = float('inf')
	while diff > threshold :
		iters = iters + 1
		random.seed()
		np.random.seed()
		random_ratings = dataset_object.randomInterveneMovie(movie)
		pmf2 = pmf.ProbabilisticMatrixFactorization(random_ratings, latent_d=5, fit_type=fit_type)
		pmf2.do_fit()
		pred2 = pmf2.prediction_for(target_user, target_item)
		print("Prediction: %f" %(pred2))

		iter_influence = abs(pred1 - pred2)
		print("%d Influence: %f" %(iters, iter_influence))

		influence = (prev*(iters-1) + iter_influence)/iters
		diff = abs(influence - prev)
		prev = influence

		print("Current: %f" %(iter_influence))
		print("Average: %f" %(influence))
		print("Difference: %f" %(diff))
		print(".")

	print(iters)


def sampler(all_items, samplesize):
	np.random.seed(seed=13)
	sampled_items = np.random.choice(all_items, size=samplesize, replace=False)
	return sampled_items

'''
Main starts here
'''
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--samplesize', default=500, type=int, help='Size of sample of movies')
	args = parser.parse_args()

	all_items = list(range(1,1683))
	sampled_items = sampler(all_items, args.samplesize)
	plotter(sampled_items, 20, "Batman Forever (1995)")

	#converge("Toy Story (1995)", 20, "Batman Forever (1995)")


if __name__ == '__main__':
	main()






