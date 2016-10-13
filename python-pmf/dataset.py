import numpy as np
import pandas as pd
import matplotlib as mpl
import pmf_cy as pmf

class Dataset(object):
	def __init__(self, dataset):
		self.name = dataset

		if(dataset == movielens100k):
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
				"../ml-100k/u.data",
				header=None,
				names=item_data_names,
				index_col=False,
				engine='python'
				)
			#remove useless columns
			self.item_data = self.item_data.drop(["video_release_date", "IMDB_URL"], axis=1)
			
			#make a dict which maps movie names to item ids
			self.item_name_index = dict(zip(self.item_data["title"], self.item_data["id"]))

	def movie_name(item):
		if item > self.num_items:
			raise IndexError ("item out of range")
		return self.item_data.loc[item-1]["title"]

	def item_number(movie):
		return self.item_name_index[movie]

	def randomInterveneItemArray(item):
		new_ratings = self.ratings
		for i in range(0,ratings.shape[0]):
			if ratings[i,1] == item:
				new_ratings[i,2] = np.random.randint(1,6)
		return new_ratings

	def randomInterveneMovieArray(movie):
		item = self.item_name_index[movie]
		return randomInterveneItemArray(item)

	def get_ratings():
		return self.ratings

def main():
	movies = Dataset(movielens100k)
	original_ratings = movies.get_ratings()
	random_ratings = movies.randomInterveneItemArray(9)

	fit_type=('mini-valid', 50, 50)

	pmf1 = pmf.ProbabilisticMatrixFactorization(original_ratings, latent_d=5, fit_type=fit_type)
	pmf2 = pmf.ProbabilisticMatrixFactorization(random_ratings, latent_d=5, fit_type=fit_type)

	pmf1.do_fit()
	pmf2.do_fit()

    pred1 = pmf1.prediction_for(5,5)
    pred2 = pmf2.prediction_for(5,5)

    print(pred1)
    print(pred2)
    print(pred1-pred2)

if __name__ == '__main__':
	main()






