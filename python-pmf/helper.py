import itertools
import random

import numpy as np

#works on matrix

def getPMFFormat(ratings_matrix):
	new_ratings = []
	for i in range(0,ratings_matrix.shape[0]):
		for j in range(0,ratings_matrix.shape[1]):
			if ratings_matrix[i,j] > 0 :
				new_ratings.append((i,j,ratings_matrix[i,j]))
	new_ratings = np.array(new_ratings)
	return new_ratings

def randomInterveneItemMatrix(ratings_matrix, item):
    #if item >= ratings_matrix.shape[1]:
    #    raise IndexError("item out of bounds")
	new_matrix = ratings_matrix
	new_item = np.random.randint(1,6,size=ratings_matrix.shape[0])
	new_matrix[:,item] = new_item
	return new_matrix

def shayakInterveneItemMatrix(ratings_matrix, item):
	#if item>=ratings_matrix.shape[1]:
	#	raise IndexError("item out of range")
	new_matrix = ratings_matrix
	old_item = ratings_matrix[:,item]
	new_item = np.zeros(ratings_matrix.shape[0])
	for i in range(0,ratings_matrix.shape[0]):
		if old_item[i]!=0:
			new_item[i]=np.random.randint(1,6)
	np.random.shuffle(new_item)
	#new_item = np.array(new_item)
	new_matrix[:,item] = new_item
	return new_matrix

def removeInterveneItemMatrix(ratings_matrix, item):
	#if item>=ratings_matrix.shape[1]:
	#	raise IndexError("item out of range")
	new_matrix = np.delete(ratings_matrix, item, axis=1)
	return new_matrix



#works on PMF array

def randomInterveneItemArray(ratings, item):
    new_ratings = ratings
    for i in range(0,ratings.shape[0]):
        if ratings[i,1] == item:
            new_ratings[i,2] = np.random.randint(1,6)
    return new_ratings

def getMatrixFormat(ratings):
    #if ratings.shape[1] != 3:
    #    raise TypeError("invalid rating tuple length")
    num_users = len(set(ratings[:,0]))
    num_items = len(set(ratings[:,1]))
    ratings_matrix = np.zeros(num_users, num_items)
    for i in range(0,ratings.shape[0]):
    	user = ratings[i,0]
    	item = ratings[i,1]
    	rate = ratings[i,2]
    	ratings_matrix[user,item] = rate
    return ratings_matrix