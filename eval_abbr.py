import sys, os, time
import logging
import argparse
import glob
import fnmatch
import nltk
import regex
import concepts
import string
import pickle
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim as gs

from sklearn.manifold import TSNE
from sklearn import (manifold, datasets, decomposition, ensemble,
					 discriminant_analysis, random_projection)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

class MyArgParser(argparse.ArgumentParser):

	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

def most_similar(vecs, word, n):

	normword = np.linalg.norm(vecs[word])
	out = []
	for cand_word, cand_vec in vecs.items():
		#x = np.dot(vecs[word], cand_vec) / (normword * np.linalg.norm(cand_vec))
		x = 1.0 - cosine(vecs[word], cand_vec)
		out.append([word, cand_word, x])
	sorted_most_sim = sorted(out, key = lambda x: x[2])[::-1]
	
	final_out = []
	for x in sorted_most_sim[1:n]:
		final_out.append(x[1])
	return final_out
'''
def top_most_similar(vecs):

	query_word = input ("What is the word you want to explore [Enter DONE if you are done exploring]? ")
	if query_word == "DONE": exit()
	ps = nltk.stem.PorterStemmer()
	query_word = ps.stem(query_word)
	try:
		print ("Most similar terms with their cosine similarity are given below:")
		most_sim = most_similar(query_word.strip().lower(), vecs)
	except:
		print ("The word ", query_word, " is not in the vocabulary")
		return
	r = 1
	for x in most_sim[1:11]:
		print ('\n') 
		print (r, ':', x[1], x[2])
		r+=1
'''

def top_most_similar(word,model,n):
	ps = nltk.stem.PorterStemmer()
	query_word = ps.stem(word)
	try:
		most_sim = most_similar(model, query_word.strip().lower(), n)
	except:
		print ("The word ", query_word, " is not in the vocabulary")
		return
	return most_sim

def load_abbr_list(abbr_path):
	list_abbr = []
	with open(abbr_path, "r") as f:
		for line in f:
			if line.startswith("#"): continue
			list_abbr.append(line.strip().split(" "))
	return list_abbr

def get_machine_score(simi_words, full_word):
	ps = nltk.stem.PorterStemmer()
	rank = 1
	for sw in simi_words:
		if sw[0] == ps.stem(full_word):
			return sw[1] #* (1.0/rank)
		rank += 1
	return 0

def get_correlations(list_abbr, model):
	c=1
	score_list_human = []
	score_list_machine = []
	for x in list_abbr:
		print (c, ". Finding the terms in the vocab (size = ", len(model.wv.vocab),") that are most similar to the term ", x[0])
		simi_words = top_most_similar(x[0],model,100000)
		if simi_words is None: continue
		machine_score = get_machine_score(simi_words, x[1])
		score_list_human.append(int(x[2]))
		score_list_machine.append(machine_score)
		c+=1
	print ("Peasrson score: ", np.corrcoef(score_list_human, score_list_machine)[0, 1])
	print ("Spearman score: ", scipy.stats.spearmanr(score_list_human, score_list_machine))


def get_pat1(list_abbr, model):
	num = 0
	ps = nltk.stem.PorterStemmer()
	for x in list_abbr:
		simi_words = top_most_similar(x[0], model, 2)
		if simi_words is None: continue
		if simi_words[0][0] == ps.stem(x[1]): num+=1

	print ("Precision at rank 1:", num)
def get_pat5(list_abbr, model):
	num = 0
	ps = nltk.stem.PorterStemmer()
	for x in list_abbr:
		simi_words = top_most_similar(x[0], model, 5)
		if simi_words is None: continue
		for i in range(0,5):
			if simi_words[i][0] == ps.stem(x[1]): 
				num+=1
				continue

	print ("Precision at rank 5: ", num)
def get_pat10(list_abbr, model):
	num = 0
	ps = nltk.stem.PorterStemmer()
	for x in list_abbr:
		simi_words = top_most_similar(x[0], model, 10)
		if simi_words is None: continue
		for i in range(0,10):
			if simi_words[i][0] == ps.stem(x[1]): 
				num+=1
				continue

	print ("Precision at rank 10:" ,num)

def get_precisions(list_abbr, model):
	get_pat1(list_abbr, model)
	get_pat5(list_abbr, model)
	get_pat10(list_abbr, model)

def main(argv):

	# start up message
	startup_message = ("\n\n",
					   "****************************************************************\n",
					   "Welcome to the MiST (Mining Software Toolkit)\n",
					   "This tool extracts knowledge from software repository\n",
					   "People working on this tool: Shayan Ali Akbar (sakbar@purdue.edu)\n",
					   "****************************************************************\n\n")
	print (''.join(startup_message))

	# Parse the arguements
	parser = MyArgParser()
	parser.add_argument('-vecspath', '--vecspath', type=str, metavar= 'vecs_path',
						help='which directory contains vectors', required=True)
	parser.add_argument('-abbrpath', '--abbrpath', type=str, metavar= 'abbr_path',
						help='which file contains abbreviations', required=True)

	args = parser.parse_args()
	
	print ("Loading model from disk ", args.vecspath+'/input.vectors')
	with open(args.vecspath + "/input.vectors", "rb") as f: model = pickle.load(f)
	print ("Loaded Model ",)
	print (model.keys())
	print ("Loading abbreviation list from disk ", args.abbrpath)
	list_abbr = load_abbr_list(args.abbrpath)
	print ("Loaded abbreviation list")
	
	# Scoring abbrs
	#get_correlations(list_abbr, model)

	# Get precisions
	get_precisions(list_abbr, model)

if __name__ == "__main__":
	main(sys.argv)
