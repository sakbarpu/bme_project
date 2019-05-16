import sys, os, time
import argparse
import nltk
import regex
import pickle
import numpy as np

from scipy.spatial.distance import cosine

class MyArgParser(argparse.ArgumentParser):

	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

def most_similar(word, vecs):

	normword = np.linalg.norm(vecs[word])
	out = []
	for cand_word, cand_vec in vecs.items():
		#x = np.dot(vecs[word], cand_vec) / (normword * np.linalg.norm(cand_vec))
		x = 1.0 - cosine(vecs[word], cand_vec)
		out.append([word, cand_word, x])
	return sorted(out, key = lambda x: x[2])[::-1]

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
def main(argv):

	# Parse the arguements
	parser = MyArgParser()
	parser.add_argument('-vecspath', '--vecspath', type=str, metavar= 'vecs_path',
						help='which directory contains vectors', required=True)
	args = parser.parse_args()

	print ("Do you want to explore (a) input vectors [IN ---> PROJ] or (b) output vectors [PROJ ---> OUT]")
	whichvecs = input ("ENTER your choice [a,b]: ")

	print ("Loading model from disk ", args.vecspath+'/model.output')
	if whichvecs == 'a': vecs = pickle.load(open(args.vecspath+'/input.vectors','rb'))
	elif whichvecs == 'b': vecs = pickle.load(open(args.vecspath+'/output.vectors','rb'))
	print ("Loaded Model:")
	print ("Vocabulary size is :", len(vecs))
	print ("Vector dimenstion is :", len(vecs[list(vecs.keys())[0]])) 

	print ( "\n\n\n",
		"		       Input							Vectors\n",
		"                 ----------------                                       ----------------------\n",
		"                 -              -                                       |0.4|0.2|  ..... |2.1|\n",
		"		  -  HUGE CORPUS -   --------->          ---------->     |3.4|1.2|  ..... |5.6|\n",
		"		  -              -   ---------> word2vec ---------->     |4.5|1.4|  ..... |1.8|\n",
		"		  -              -   --------->          ---------->     | . | . |        | . |\n",
		"		  -      	 -					 | . | . |        | . |\n",
		"                 ----------------                                       ----------------------\n\n")

	while True:
		top_most_similar(vecs)		
	
if __name__ == "__main__":
	main(sys.argv)
