'''
This is the implementation of word2vec algorithm.

It takes as input the corpus file and the word2vec 
parameters file, and returns word embeddings saved as 
numpy file to an specified output path.

TODO: 

implement CBOW NS
implement SG HS 
implement CBOW HS

'''

import sys, os, time
import logging
import argparse
import fnmatch
import nltk
import regex
import string
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import multiprocessing

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from scipy.special import expit
from math import ceil

class MyArgParser(argparse.ArgumentParser):

	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

class LineSentences:

	def __init__(self, inpath):
		
		self.infile = open(inpath, encoding="iso-8859-15")

	def __iter__(self):

		self.infile.seek(0)
		for line in itertools.islice(self.infile, None):
			yield line
	
	def __len__(self):
		self.infile.seek(0)
		c = 0
		for line in itertools.islice(self.infile, None):
			c+=1
		return c

	def __getitem__(self, key):
		self.infile.seek(0)
		if type(key) is int:
			return next(itertools.islice(self.infile, key, key+1))
			#yield itertools.islice(self.infile, key, key+1)
		elif type(key) is tuple:
			tmp = []
			for x in itertools.islice(self.infile, key[0], key[1]):
				tmp.append(x)
			return tmp

class Reader:
	'''
	This class is used to read files from the disk
	Also it can list all the files that are there in the data
	'''

	def __init__(self):
		self.data_path = None  # path for the repo
		self.pattern = None  # extension pattern for regex
		self.files = None  # list of all files in the repo

	def get_file_list(self):
		'''
		Get a list of all files from the data dir
		ret1: list of filenames
		'''

		filenames = []
		counter_files = 0
		with open("etc/filenames.txt", "w") as f:
			for root, dirs, files in os.walk(self.data_path):
				for basename in files:
					if fnmatch.fnmatch(basename, self.pattern):
						filename = os.path.join(root, basename)
						
						f.write(str(counter_files) + "," + filename + "\n")
						counter_files += 1
						filenames.append(filename)
		self.files = filenames
		return filenames

	def read_file(self, file_path):
		'''
		Read contents of a single file
		arg1: the path to the file
		ret1: content of the file
		'''

		with open(file_path, encoding='iso-8859-15') as f:
			file_content = f.read()
		return file_content

	def read_sentences(self, file_path):
		'''
		Read list of sentences from file
		arg1: file path to the corpus (a single file with entire corpus)
		ret1: object of list of sentences
		'''
		return LineSentences(file_path)

class Preprocessor:
	'''
	This class implements the functions for the preprocessing of content of file.
	The pipeline that we follow for preprocessing is as follows:
		(1)remove_punctuations
		(2)perform_camel_case_splitting
		(3)perform_lower_casing
		(4)remove_stopwords_using_file
		(5)perform_stemming
	'''

	def __init__(self):
		self.raw_content = None
		self.stopwords_file = None
		self.list_stopwords = None
		self.punctuation_removed_content = None
		self.camel_case_split_content = None
		self.lowercased_content = None
		self.stopword_removed_content = None
		self.stemmed_content = None
		self.current_content = None
		self.processed_content = None
		self.tokenized_content = None

	def read_stopwords(self):
		list_stopwords = []
		with open(self.stopwords_file) as f:
			for line in f:
				list_stopwords.append(line.strip())
		self.list_stopwords = list_stopwords
		return list_stopwords

	def perform_stemming(self):
		'''
		This function does the porter stemming using nltk
		ret1: the processed/stemmed content
		'''

		porter_stemmer = nltk.stem.PorterStemmer()
		# wn_lemmatizer = nltk.wordnet.WordNetLemmatizer()
		self.tokenized_content = [porter_stemmer.stem(i) for i in nltk.tokenize.word_tokenize(self.current_content)]
		# self.tokenized_content = [wn_lemmatizer.lemmatize(i) for i in nltk.tokenize.word_tokenize(self.current_content)]
		self.current_content = " ".join(self.tokenized_content)
		self.processed_content = self.current_content
		self.stemmed_content = self.current_content
		return self.stemmed_content

	def remove_stopwords_using_file(self):
		'''
		Remove all stopwords from the content
		ret1: the processed content
		'''

		content = self.current_content

		for stopword in self.list_stopwords:
			pattern = " " + stopword + " "
			content = regex.sub(pattern, " ", content)

		content = ''.join([i for i in content if not i.isdigit()])
		self.stopword_removed_content = content
		self.current_content = self.stopword_removed_content
		return self.stopword_removed_content

	def perform_lower_casing(self):
		'''
		Convert content to lower case
		ret1: processed lower cased content
		'''

		content = self.current_content
		self.lowercased_content = self.current_content.lower()
		self.current_content = self.lowercased_content
		return self.lowercased_content

	def perform_camel_case_splitting(self):
		'''
		Convert all camelcase terms into individual terms
		ret1: processed content without any camelcase terms
		'''

		content = self.current_content
		# self.camel_case_split_content = regex.sub(r'([a-z]*)([A-Z])', r'\1 \2', content)
		matches = regex.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', content)
		self.camel_case_split_content = " ".join([m.group(0) for m in matches])
		self.current_content = self.camel_case_split_content
		return self.camel_case_split_content

	def remove_punctuations(self):
		'''
		Remove all punctuations from the contents
		ret1: The processed content
		'''

		content = self.raw_content
		self.punctuation_removed_content = "".join(l if l not in string.punctuation else " " for l in content)
		self.current_content = self.punctuation_removed_content
		return self.punctuation_removed_content

	def perform_preprocessing(self):
		self.current_content = self.raw_content
		self.punctuation_removed_content = self.remove_punctuations()
		self.camel_case_split_content = self.perform_camel_case_splitting()
		self.lowerecased_content = self.perform_lower_casing()
		self.stopword_removed_content = self.remove_stopwords_using_file()
		self.stemmed_content = self.perform_stemming()
		self.processed_content = self.stemmed_content
		return self.processed_content

class Word2Vec:

	def __init__(self, contentpath=None, sentences=None, size=100, alpha=0.025, window=5, 
				min_count=5, sample=0.001, 
				workers=3, min_alpha=0.0001, sg=1, hs=0, negative=5, 
				ns_exponent=0.75, cbow_mean=1, iters=5, 
				batch_words=10000, compute_loss=False):
		
		self.contentpath = contentpath
		self.sentences = sentences
		self.size = size
		self.alpha = alpha
		self.window = window
		self.min_count = min_count
		self.sample = sample
		self.workers = workers
		self.min_alpha = min_alpha
		self.sg = sg
		self.hs = hs
		self.negative = negative
		self.ns_exponent = ns_exponent
		self.cbow_mean = cbow_mean
		self.iters = iters
		self.batch_words = batch_words
		self.compute_loss = compute_loss

		self.EXP_TABLE_SIZE = 1000
		self.MAX_STRING = 100
		self.MAX_EXP = 6
		self.MAX_SENTENCE_LEN = 1000
		self.MAX_CODE_LENGTH = 40
		
		self.vocab = {}
		self.unigram_table = None
		self.sorted_vocab_words = []
		self.sorted_vocab_words_counts = []
		self.current_num_iters = 0
		self.current_num_words_processed = 0
		self.start_alpha = alpha
		self.total_words_in_corpus = 0
		if contentpath: self.total_sents_in_corpus = len(LineSentences(contentpath))
		elif sentences: self.total_sents_in_corpus = len(sentences)
		#self.exp_table = self.compute_exp_table()
		self.W = {}
		self.Z = {}

	'''
	def compute_exp_table(self):

		exp_table = []
		for i in range(0,self.EXP_TABLE_SIZE):
			tmp = np.exp((i / float(self.EXP_TABLE_SIZE) * 2 - 1) * self.MAX_EXP)
			exp_table.append(tmp / (tmp+1))
		return exp_table
	'''
	def trim_vocab(self):
		'''
		For removing less frequent words from the vocab.
		If count of a certain word is < min_count it is removed from vocab
		and not considered while training.
		'''
		for k in list(self.vocab.keys()):
			if self.vocab[k] < self.min_count:
				del self.vocab[k]

	def build_sorted_vocab(self):
		'''
		form two lists. one of sorted vocab words. the other of its count corresponding.
		'''
		
		self.sorted_vocab_words = sorted(self.vocab.keys())
		self.sorted_vocab_words_counts = [self.vocab[x] for x in self.sorted_vocab_words]


	def build_unigram_table(self, domain=2**31 - 1):
		'''
		We build a unigram table here so that we can call it when getting a random word in negative sampling
		Following word2vec folks we raise the power of the count to 3/4
		
		The unigram table is constructed as follows: 
		For each word x in the vocab we find its count count(x) in the corpus
		We raise its count to power 3/4 count(x)^(3/4)
		Then we divide it to the normalization factor Z = sum_x count(x)^(3/4)			
		Save all the results in a list
		
		'''

		vals_raised_power = np.power(self.sorted_vocab_words_counts, 0.75)
		sum_all_powered_vals = np.sum(vals_raised_power)
		
		self.unigram_table = np.zeros(len(self.sorted_vocab_words_counts), dtype=np.uint32)
		cum = 0.0
		for a in range(0,len(self.sorted_vocab_words_counts)):
			cum += np.power(self.sorted_vocab_words_counts[a], 0.75)
			self.unigram_table[a] = round(cum / sum_all_powered_vals * domain)

		if len(self.unigram_table) > 0:
			assert self.unigram_table[-1] == domain

	def learn_vocab(self):
		'''
		This is where the corpus is scanned and relevant information are extracted
		This function is called at the start of the modeling.
		Learning the vocab means populating the self.vocab dictionary with the words
		found in the corpus along with their total count in the corpus.
		
		So if there are 4 words in the corpus "dog", "cat", "is", and "am" with 
		frequencies/count 4, 3, 10, and 1, respectively, then we will end up with a 
		vocab {"cat":3, "dog":4, "am":1, "is":10}.
		
		We also trim the vocab removing words that are very rare in the corpus.
		This is controlled by the parameter "min_count".
		If min_count is 2, the "am" will be removed in the above example.
		And we will end up with the vocab {"cat":3, "dog":4, "is":10}.

		This function also calls the build unigram table function which basically
		builds a cumulative distribution table. This table is used to get a random
		word out from the unigram to the power 3/4 distribution.
		'''
		self.sentences = LineSentences(self.contentpath)
		for sent in self.sentences:
			for w in sent.split(" "):
				self.total_words_in_corpus += 1
				w = w.strip()
				if w in self.vocab:
					self.vocab[w] += 1
				else:
					self.vocab[w] = 1
		print ("\nThere are total ", self.total_words_in_corpus, " words in corpus")
		print ("Out of which ", len(self.vocab), "are distinct words")
		self.trim_vocab()
		self.build_sorted_vocab()
		self.build_unigram_table()
		print ("After triming vocab we are left with ", len(self.vocab), "distinct words")

	def init_model(self):
		'''
		initializing model 
		weight/parameters from input to proj layer W init random
		W in MxN. M is the inherent dims of vectors and N is the total number of words in the vocab
		Z is also MxN weight/parameter matrix from proj to output.
		Z is init as all 0s.
		We use dicts data structure for storing W and Z
				_						_
		W =  |word1		 word2 ... wordN|
			1|0.1		0.2			0.4 |
			.| .		 .			 .	|
			.| .		 .			 .	|
			.| .		 .			 .	|
			M|0.2		0.1			0.8 |
			 |_						 _|
		
		Z =  |word1		 word2 ... wordN|
			1|0.0		0.0			0.0 |
			.| .		 .			 .	|
			.| .		 .			 .	|
			.| .		 .			 .	|
			M|0.0		0.0			0.0 |
			 |_						 _|

		'''
		
		for word in self.vocab:
			self.W[word] = np.random.rand(self.size)
			self.Z[word] = np.random.rand(self.size)

	def get_random_train_sample_from_a_sent(self, sent):
		'''
                Given a sentence sent get a random sample out of it
                Random sample means a random word with surrounding words
		'''

		rand_pos_in_sent = np.random.randint(0, len(sent))

		if rand_pos_in_sent - self.window < 0: return #check for boundary
		if rand_pos_in_sent + self.window > len(sent): return

		return [sent[rand_pos_in_sent-self.window:rand_pos_in_sent] + sent[rand_pos_in_sent+1:rand_pos_in_sent+self.window],
			sent[rand_pos_in_sent]]		
			
	def call_a_sgwithns_thread(self, local_sents, start, stop, counter_worker):
		'''
                This is a function that implements a single thread functionality and should be called from the multiprocessing Process
                This implements SG model with NS heuristic routine
                arg1: local_sents is the local copy of LineSentences instance for this thread
                arg2: start is the starting number of sentence for the chunk of sentences which this thread will train on
                arg3: stop is the ending number of sentence for the chunk of sentences which this thread will train on
                arg4: counter_worker is the number of this worker thread
		'''

		local_num_words_processed = 0
		local_batch_count = 0
		
		for local_iter in range(self.iters): #loop over this worker for iter number of times
			start_time = time.time()
			for sent in local_sents[start,stop]: #the data chunk for this worker is [start,stop] so loop over that chunk
				sent = sent.strip().split(" ") #tokenize the current sentence
				if len(sent) < self.window: continue #no need to process this small sentence

				for counter_terms in range(len(sent)): #loop over the sentence the length of sentence times
					
					if local_num_words_processed % self.batch_words == 0:
						local_batch_count += 1
						#update current alpha 
						self.alpha = self.start_alpha * (1.0 - local_num_words_processed / float(self.iters * self.total_words_in_corpus + 1))
						if self.alpha < self.start_alpha * 0.0001: self.alpha = self.start_alpha * 0.0001
						
					train_sample = self.get_random_train_sample_from_a_sent(sent) #get a train sample
					if train_sample is None: continue #if the sample is no good we got a None
					if train_sample[1] not in self.W: continue #no point training for this term as it is not in W (i.e. trimmed vocab)

					for context_word in train_sample[0]:
						if context_word == train_sample[1]: continue #no point predicting from same input same output
						if context_word not in self.W: continue #no point training with this context term as it is not in vocab

						local_num_words_processed += 1
						
						neu = np.zeros((self.size),dtype=float)
						for d in range(0, self.negative):
							if d == 0: 
								label = 1
							else:
								context_word = self.sorted_vocab_words[
											self.unigram_table.searchsorted(np.random.randint(self.unigram_table[-1]))]
								label = 0

							f = np.dot(self.W[train_sample[1]],self.Z[context_word]) #propagate proj to output

							if (f > self.MAX_EXP): g = (label - 1) * self.alpha
							elif (f < -self.MAX_EXP): g = (label - 0) * self.alpha
							else: g = (label - expit(f)) * self.alpha # gradient calculation
							
							neu = neu + (g * self.Z[context_word]) # error sum calculation
							
							self.Z[context_word] = self.Z[context_word] + (g * self.W[train_sample[1]]) #learn proj to output
						
						self.W[train_sample[1]] = self.W[train_sample[1]] + neu # learn input to proj 
			print ("WORKER:", counter_worker, "ITER:", local_iter, "Took time:", time.time()-start_time)	
		print ("WORKER:", counter_worker, "DONE Training,", "Trained", local_num_words_processed, "examples")
		
	def train_sg_model_with_ns(self):
		
		m = ceil(self.total_sents_in_corpus / self.workers)
		print ("\nWorking with ", self.workers, " workers, with each worker working on", m, "sentences")
		
		processes = []
		#lock = multiprocessing.Lock()
		counter_worker = 0
		for counter in range(0,self.total_sents_in_corpus,m):
			print ("WORKER: ", counter_worker, ", will work on sentences from :", counter, "to: ", min(self.total_sents_in_corpus, counter+m), "\n")
			local_sents = LineSentences(self.contentpath)
			processes.append(multiprocessing.Process(target=self.call_a_sgwithns_thread, 
								  args=(local_sents,counter,min(self.total_sents_in_corpus,counter+m),counter_worker,)))
			counter_worker += 1
		for process in processes: process.start()
		for process in processes: process.join()

	def build_sg_model(self):
		'''
		This function builds the skipgram model

		'''
		print ("Building skipgram model\n")
		
		if self.negative > 0:
			self.train_sg_model_with_ns()
			print ("DONE WITH TRAINING SG model with Negative Sampling")
		else: 
			#TODO: NOT IMPLEMENTED YET THE SG MODELING WITH HS
			pass 

	def build_cbow_model(self):

		print ("Building cbow model")
		
		if self.negative > 0:
			#TODO: NOT IMPLEMENTED YET THE CBOW MODELING WITH NS
			pass
		else: 
			#TODO: NOT IMPLEMENTED YET THE CBOW MODELING WITH HS
			pass

	def build_model(self):
		
		self.learn_vocab()
		print ("Learned vocabulary from corpus\n")
		
		self.init_model()
		print ("initialized model parameters W (input vectors) and Z (output vectors)\n")

		if self.sg == 1: self.build_sg_model()
		else: self.build_cbow_model()

	def save(self, out_path):
		with open(out_path+'/input.vectors', 'wb') as f: pickle.dump(self.W, f, pickle.HIGHEST_PROTOCOL)		
		with open(out_path+'/output.vectors', 'wb') as f: pickle.dump(self.Z, f, pickle.HIGHEST_PROTOCOL)		


def main(argv):

	# start up message
	startup_message = ("\n\n",
						 "****************************************************************\n",
						 "Welcome to the MiST (Mining Software Toolkit)\n",
						 "This tool extracts knowledge from software repository\n",
						 "People working on this tool: Shayan Ali Akbar (sakbar@purdue.edu)\n",
						 "****************************************************************\n\n")
	print (''.join(startup_message))
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	# Parse the arguements
	parser = MyArgParser()
	parser.add_argument('-corpuspath', '--corpuspath', type=str, metavar= 'corpus_dir', action='store',
						help='which directory is containing corpus (e.g. eclipse path)',
						required=False)
	parser.add_argument('-processedfilepath', '--processedfilepath', type=str, metavar= 'processed_file_path', action='store',
						help='which directory is containing the preprocessed file (each document in a separate line)',
						required=False)
	parser.add_argument('-output', '--output', type=str, metavar= 'output_dir', action='store',
						help='which directory is output',
						required=True)
	args = parser.parse_args()
	if args.corpuspath is not None: corpuspath = args.corpuspath
	if args.processedfilepath is not None: processedfilepath = args.processedfilepath
	output = args.output
	
	# Get the sentences from input corpus or preprocessed file
	if args.corpuspath:
		# Read the files and preprocess them before giving to the word2vec algorithm
		reader = Reader()
		reader.data_path = args.corpuspath
		reader.pattern = "*.java"
		files = reader.get_file_list()
		print ("\nThere are ", len(files), " in the repo(s) at ", reader.data_path)

		preprocessor = Preprocessor()
		preprocessor.stopwords_file = 'etc/stopword-list.txt'
		preprocessor.read_stopwords()
		c = 0
		with open(args.output + "/content.txt", "w") as f:
			for file in files:
				if not os.path.isfile(file): continue
				if c%100==0: print ('file number:', c, '/', len(files))
				content = reader.read_file(file)
				preprocessor.raw_content = content
				f.write(preprocessor.perform_preprocessing() + "\n")
				c+=1
		print ("Done preprocessing... Now writing to file ")

		print ("Getting sentences from the repo(s)")
		reader = Reader()
		sentences = reader.read_sentences(args.output + "/content.txt")

	elif args.processedfilepath:
		reader = Reader()
		sentences = reader.read_sentences(args.processedfilepath)

	else:
		print ("You need to either set the corpuspath or processedfilepath to provide input")
		exit()
	
	print ("Got sentences list ")
	
	print ("\nNow training with word2vec algorithm")
	t0 = time.time()
	model = Word2Vec(contentpath=processedfilepath, min_count=0, size=500, sg=1, hs=0, negative=15, iters=1,
					 window=8, compute_loss=True, workers=16)
	model.build_model()
	t1 = time.time()
	print ("Done with training. Took time: ", t1-t0, "secs")

	print ("Saving model to disk ", args.output)
	model.save(args.output)
	print ("Saved")

if __name__ == "__main__":
	main(sys.argv)
