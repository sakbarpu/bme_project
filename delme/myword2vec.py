import sys, os, time
import logging
import argparse
import fnmatch
import nltk #only used for stemming (a preprocessing step for text data)
import regex
import string
import random
import matplotlib.pyplot as plt
import itertools
import multiprocessing
import math
import struct
import warnings
import pickle
import numpy as np

from multiprocessing import Pool, Value, Array
from math import ceil
from scipy.special import expit

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

		self.EXP_TABLE_SIZE = 1e8
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
		#self.ops = (add, sub)
		global W, Z 
		W = {}
		Z = {}

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

		self.vocab['<unk>'] = 0
		for k in list(self.vocab.keys()):
			if self.vocab[k] < self.min_count:
				self.vocab['<unk>'] += 1
				del self.vocab[k]
		if '<unk>' not in self.vocab: self.vocab['<unk>'] = 0

	def build_sorted_vocab(self):
		'''
		form two lists. one of sorted vocab words. the other of its count corresponding.
		'''
		
		self.sorted_vocab_words = sorted(self.vocab.keys())
		#self.sorted_vocab_words_counts = [self.vocab[x] for x in self.sorted_vocab_words]
		self.sorted_vocab_words_counts = sorted(self.vocab.values(), reverse=True)

	def build_unigram_table(self, domain=2**31 - 1):
		'''
		We build a unigram table here so that we can call it when getting a random word in negative sampling
		Following word2vec folks we raise the power of the count to 3/4
		
		The unigram table is constructed as follows: 
		For each word x in the vocab we find its count count(x) in the corpus
		We raise its count to power 3/4 count(x)^(3/4)
		Then we divide it by the normalization factor Z = sum_x count(x)^(3/4) where x loops over vocab
		Save all the results in a list
		
		'''

		sum_all_powered_vals = np.sum([np.power(self.sorted_vocab_words_counts, 0.75)])
		self.unigram_table = np.zeros(int(self.EXP_TABLE_SIZE), dtype=np.uint32)

		cum = 0.0
		i = 0
		print (self.sorted_vocab_words_counts)
		for j, count in enumerate(self.sorted_vocab_words_counts):
			cum += float(math.pow(count, 0.75)) / sum_all_powered_vals
			while i < self.EXP_TABLE_SIZE and float(i) / self.EXP_TABLE_SIZE < cum:
				self.unigram_table[i] = j
				i += 1

		#self.unigram_table = np.array([np.count_nonzero(self.unigram_table == x) for x in range(len(self.sorted_vocab_words_counts))])
		
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
		'''

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
		num_sents = len(self.sentences)
		count_sent = 0
		self.vocab['<start>'] = 0
		self.vocab['<end>'] = 0
		self.vocab['<unk>'] = 0
		print (self.vocab)
		for sent in self.sentences:
			self.vocab['<start>'] += 1
			self.vocab['<end>'] += 1
			percent_sent_done = (count_sent/num_sents)*100 
			if percent_sent_done % 5==0: print ("PROGRESS:", percent_sent_done, "%, Working on sentence #",
								 count_sent, "out of total", num_sents, "sentences")
			count_sent += 1
			for w in sent.split(" "):
				self.total_words_in_corpus += 1
				w = w.strip()
				if w in self.vocab:
					self.vocab[w] += 1
				else:
					self.vocab[w] = 1
		print ("\nThere are total ", self.total_words_in_corpus, " words in corpus of size", num_sents)
		print ("Out of which ", len(self.vocab), "are distinct words")
		self.trim_vocab()
		print (self.vocab['<unk>'])
		for k,v in self.vocab.items(): print (k,v)
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
		global W, Z
		tmp = np.random.uniform(low=-0.5/self.size, high=0.5/self.size, size=(len(self.vocab), self.size))
		c = 0
		for word in self.vocab:
			W[word] = tmp[c,:]
			Z[word] = np.zeros(self.size)
			c += 1

	def get_random_train_sample_from_a_sent(self, sent, pos_term):
		s = max(0, pos_term - self.window)
		e = min(len(sent), pos_term + int(self.window * random.random()))
		return [sent[s : e], sent[pos_term]]

	def get_random_train_sample_from_a_sent1(self, sent):
		'''
                Given a sentence sent get a random sample out of it
                Random sample means a random word with a context word chosen from surrounding words
		'''

		#rand_pos_in_sent = np.random.randint(0, len(sent))
		rand_pos_in_sent = int(len(sent) * random.random())
		#rand_pos_in_sent = fastrand.pcg32bounded(len(sent))		
		#rand_pos_in_sent = 2

		if rand_pos_in_sent - self.window < 0: return #check for boundary
		if rand_pos_in_sent + self.window > len(sent): return

		#op = random.choice(self.ops)
		#return [sent[rand_pos_in_sent - self.window +  int(self.window*random.random())], sent[rand_pos_in_sent]]
		#return [sent[op(rand_pos_in_sent,int(self.window*random.random()))], sent[rand_pos_in_sent]]
		return [sent[rand_pos_in_sent - self.window : rand_pos_in_sent + int(self.window*random.random())], sent[rand_pos_in_sent]]
		#return [sent[op(rand_pos_in_sent,fastrand.pcg32bounded(self.window))], sent[rand_pos_in_sent]]
		#return [sent[rand_pos_in_sent - self.window + fastrand.pcg32bounded(self.window * 2)], sent[rand_pos_in_sent]]
		#return [sent[1], sent[2]]



	def call_a_sgwithns_thread1(self, local_sents, start, stop, counter_worker):
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
		labels = np.zeros(self.negative+1)
		labels[0] = 1.0	
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
						
						print ("WORKER:", counter_worker, ", Iter:", local_iter, ", Batch:", local_batch_count)
					train_sample = self.get_random_train_sample_from_a_sent(sent) #get a train sample
					if train_sample is None: continue #if the sample is no good we got a None
					if train_sample[1] not in self.W: continue #no point training for this term as it is not in W (i.e. trimmed vocab)

					context_word = train_sample[0]
					focus_word = train_sample[1]
					if context_word == focus_word: continue #no point predicting from same input same output
					if context_word not in self.W: continue #no point training with this context term as it is not in vocab
					v_w = self.W[focus_word]
					err_sum = np.zeros(self.size)
					local_num_words_processed += 1
					
					words = [context_word]
					for d in range(0, self.negative):
						words.append(self.sorted_vocab_words[self.unigram_table.searchsorted(int(self.unigram_table[-1])*random.random())])
					v_p_ws = np.array([self.Z[word] for word in words]).T
					f = np.dot(v_w, v_p_ws) #propagate proj to output
					output = expit(f)
					g = (labels - output) * self.alpha
					tmp = np.outer(g,v_w).T
					for c in range(len(words)): self.Z[words[c]] += tmp[:,c]
					err_sum += np.dot(g, v_p_ws.T)
					self.W[focus_word] += err_sum
			print ("WORKER:", counter_worker, "ITER:", local_iter, "Took time:", time.time()-start_time)	
		print ("WORKER:", counter_worker, "DONE Training,", "Trained", local_num_words_processed, "examples")
	




	
	def call_a_sgwithns_thread(self, local_sents, start, stop, counter_worker):
		'''
                This is a function that implements a single thread functionality and should be called from the multiprocessing Process
                This implements SG model with NS heuristic routine
                arg1: local_sents is the local copy of LineSentences instance for this thread
                arg2: start is the starting number of sentence for the chunk of sentences which this thread will train on
                arg3: stop is the ending number of sentence for the chunk of sentences which this thread will train on
                arg4: counter_worker is the number of this worker thread
		'''
		global W, Z

		local_num_words_processed = 0
		local_batch_count = 0
		#print ("***", counter_worker, start, stop)
		
		for local_iter in range(self.iters): #loop over this worker for iter number of times
			start_time = time.time()
			sent_count = 0
			for sent in local_sents[start,stop]: #the data chunk for this worker is [start,stop] so loop over that chunk
				sent_count += 1
				#print ("===", counter_worker, local_iter, sent)
				sent = sent.strip().split(" ") #tokenize the current sentence
				if len(sent) < self.window: continue #no need to process this small sentence

				for counter_terms in range(len(sent)): #loop over the sentence the length of sentence times
					#print (counter_worker, counter_terms)
					if local_num_words_processed % self.batch_words == 0:
						local_batch_count += 1
						#update current alpha 
						self.alpha = self.start_alpha * (1.0 - local_num_words_processed / float(self.iters * self.total_words_in_corpus + 1))
						if self.alpha < self.start_alpha * 0.0001: self.alpha = self.start_alpha * 0.0001
						
						print ("WORKER:", counter_worker, ", Iter:", local_iter, ", Batch:", local_batch_count)
					train_sample = self.get_random_train_sample_from_a_sent(sent, counter_terms) #get a train sample
					if train_sample is None: continue #if the sample is no good we got a None
					if train_sample[1] not in W: continue #no point training for this term as it is not in W (i.e. trimmed vocab)
					
					print ("...", counter_worker, train_sample[0], train_sample[1])	
					return
					#context_word = train_sample[0]
					for context_word in train_sample[0]:	
						focus_word = train_sample[1]
						if context_word == focus_word: continue #no point predicting from same input same output
						if context_word not in W: continue #no point training with this context term as it is not in vocab
						
						#print ("worker:", counter_worker, "true:", context_word, focus_word)
						local_num_words_processed += 1
						
						neu = np.zeros((self.size),dtype=float)
						#for d in range(0, self.negative):
						d = 0
						while d < self.negative + 1:
							if d == 0: 
								label = 1.0
								context_word1 = focus_word
							else:
								#context_word = self.sorted_vocab_words[
								#			self.unigram_table.searchsorted(np.random.randint(self.unigram_table[-1]))]
								context_word1 = self.sorted_vocab_words[
											self.unigram_table.searchsorted(int(self.unigram_table[-1]*random.random()))]
								#context_word = self.sorted_vocab_words[
								#			self.unigram_table.searchsorted(fastrand.pcg32bounded(self.unigram_table[-1]))]
								#context_word = self.sorted_vocab_words[self.unigram_table.searchsorted(self.unigram_table[-1])]
								if context_word1 == focus_word: continue
								label = 0.0

							#print ("\nd", d, context_word, focus_word)	
							#print (self.W[focus_word].shape, self.Z[context_word].shape)
							f = np.dot(W[context_word],Z[context_word1]) #propagate proj to output
							#print ("f", f, "sigf", expit(f), "label", label)
							if (f > self.MAX_EXP): g = (label - 1.0) * self.alpha
							elif (f < -self.MAX_EXP): g = (label - 0.0) * self.alpha
							else: g = (label - expit(f)) * self.alpha # gradient calculation
							#print ("iter", local_iter, "sent_count", sent_count, "term_count", counter_terms, "d", d, 
							#       "focus", focus_word, "context", context_word, "f", f, "sigf", expit(f), "label", label, "g", g)
							neu = neu + (g * Z[context_word1]) # error sum calculation
							
							Z[context_word1] = Z[context_word1] + (g * W[context_word]) #learn proj to output
							d += 1
						#print (neu)
						W[context_word] = W[context_word] + neu # learn input to proj
						#return
			print ("WORKER:", counter_worker, "ITER:", local_iter, "Took time:", time.time()-start_time)	
		print ("WORKER:", counter_worker, "DONE Training,", "Trained", local_num_words_processed, "examples")
		
	def train_sg_model_with_ns(self):
		
		m = ceil(self.total_sents_in_corpus / self.workers)
		print ("\nWorking with ", self.workers, " workers, with each worker working on", m, "sentences")
		
		processes = []
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
		print ("Building skipgram model...\n")
		
		if self.negative > 0:
			self.train_sg_model_with_ns()
			print ("\n\nDONE WITH TRAINING SG model with Negative Sampling")
		else: 
			#TODO: NOT IMPLEMENTED YET THE SG MODELING WITH HS
			pass 

	def build_cbow_model(self):

		print ("Building cbow model...\n")
		
		if self.negative > 0:
			#TODO: NOT IMPLEMENTED YET THE CBOW MODELING WITH NS
			pass
		else: 
			#TODO: NOT IMPLEMENTED YET THE CBOW MODELING WITH HS
			pass

	def build_model(self):
		
		print ("Now learning the vocab of the corpus...\n")
		self.learn_vocab()
		print ("Learned vocabulary from corpus\n")
		
		for k,v in self.vocab.items(): print (k, v)
		self.init_model()
		print ("initialized model parameters W (input vectors) and Z (output vectors)\n")

		if self.sg == 1: self.build_sg_model()
		else: self.build_cbow_model()

	def save(self, out_path):
		global W, Z
		with open(out_path+'/input.vectors', 'wb') as f: pickle.dump(W, f, pickle.HIGHEST_PROTOCOL)		
		with open(out_path+'/output.vectors', 'wb') as f: pickle.dump(Z, f, pickle.HIGHEST_PROTOCOL)		






class VocabItem:
	def __init__(self, word):
		self.word = word
		self.count = 0
		self.path = None # Path (list of indices) from the root to the word (leaf)
		self.code = None # Huffman encoding

class Vocab:
	def __init__(self, fi, min_count):
		vocab_items = []
		vocab_hash = {}
		word_count = 0
		fi = open(fi, 'r')

		# Add special tokens <bol> (beginning of line) and <eol> (end of line)
		for token in ['<bol>', '<eol>']:
			vocab_hash[token] = len(vocab_items)
			vocab_items.append(VocabItem(token))

		for line in fi:
			tokens = line.split()
			for token in tokens:
				if token not in vocab_hash:
					vocab_hash[token] = len(vocab_items)
					vocab_items.append(VocabItem(token))
					
				#assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
				vocab_items[vocab_hash[token]].count += 1
				word_count += 1
			
				if word_count % 10000 == 0:
					sys.stdout.write("\rReading word %d" % word_count)
					sys.stdout.flush()

			# Add special tokens <bol> (beginning of line) and <eol> (end of line)
			vocab_items[vocab_hash['<bol>']].count += 1
			vocab_items[vocab_hash['<eol>']].count += 1
			word_count += 2

		self.bytes = fi.tell()
		self.vocab_items = vocab_items		                 # List of VocabItem objects
		self.vocab_hash = vocab_hash			         # Mapping from each token to its index in vocab
		self.word_count = word_count			         # Total number of words in train file

		# Add special token <unk> (unknown),
		# merge words occurring less than min_count into <unk>, and
		# sort vocab in descending order by frequency in train file
		self.__sort(min_count)

		#assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
		print ('Total words in training file: %d' % self.word_count)
		print ('Total bytes in training file: %d' % self.bytes)
		print( 'Vocab size: %d' % len(self))

	def __getitem__(self, i):
		return self.vocab_items[i]

	def __len__(self):
		return len(self.vocab_items)

	def __iter__(self):
		return iter(self.vocab_items)

	def __contains__(self, key):
		return key in self.vocab_hash

	def __sort(self, min_count):
		tmp = []
		tmp.append(VocabItem('<unk>'))
		unk_hash = 0
		
		count_unk = 0
		for token in self.vocab_items:
			if token.count < min_count:
				count_unk += 1
				tmp[unk_hash].count += token.count
			else:
				tmp.append(token)

		tmp.sort(key=lambda token : token.count, reverse=True)

		# Update vocab_hash
		vocab_hash = {}
		for i, token in enumerate(tmp):
			vocab_hash[token.word] = i

		self.vocab_items = tmp
		self.vocab_hash = vocab_hash

		print (" ")
		print ('Unknown vocab size:', count_unk)

	def indices(self, tokens):
		return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

	def encode_huffman(self):
		# Build a Huffman tree
		vocab_size = len(self)
		count = [t.count for t in self] + [1e15] * (vocab_size - 1)
		parent = [0] * (2 * vocab_size - 2)
		binary = [0] * (2 * vocab_size - 2)
		
		pos1 = vocab_size - 1
		pos2 = vocab_size

		for i in xrange(vocab_size - 1):
			# Find min1
			if pos1 >= 0:
				if count[pos1] < count[pos2]:
					min1 = pos1
					pos1 -= 1
				else:
					min1 = pos2
					pos2 += 1
			else:
				min1 = pos2
				pos2 += 1

			# Find min2
			if pos1 >= 0:
				if count[pos1] < count[pos2]:
					min2 = pos1
					pos1 -= 1
				else:
					min2 = pos2
					pos2 += 1
			else:
				min2 = pos2
				pos2 += 1

			count[vocab_size + i] = count[min1] + count[min2]
			parent[min1] = vocab_size + i
			parent[min2] = vocab_size + i
			binary[min2] = 1

		# Assign binary code and path pointers to each vocab word
		root_idx = 2 * vocab_size - 2
		for i, token in enumerate(self):
			path = [] # List of indices from the leaf to the root
			code = [] # Binary Huffman encoding from the leaf to the root

			node_idx = i
			while node_idx < root_idx:
				if node_idx >= vocab_size: path.append(node_idx)
				code.append(binary[node_idx])
				node_idx = parent[node_idx]
			path.append(root_idx)

			# These are path and code from the root to the leaf
			token.path = [j - vocab_size for j in path[::-1]]
			token.code = code[::-1]

class UnigramTable:
	"""
	A list of indices of tokens in the vocab following a power law distribution,
	used to draw negative samples.
	"""
	def __init__(self, vocab):
		vocab_size = len(vocab)
		power = 0.75
		norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constant
		print (norm)

		table_size = 1e8 # Length of the unigram table
		table = np.zeros(int(table_size), dtype=np.uint32)

		print ('Filling unigram table')
		p = 0 # Cumulative probability
		i = 0
		for x in vocab: print (x.word, x.count)
		for j, unigram in enumerate(vocab):
			p += float(math.pow(unigram.count, power))/norm
			while i < table_size and float(i) / table_size < p:
				table[i] = j
				i += 1
		self.table = table

	def sample(self, count):
		indices = np.random.randint(low=0, high=len(self.table), size=count)
		return [self.table[i] for i in indices]

def sigmoid(z):
	if z > 6:
		return 1.0
	elif z < -6:
		return 0.0
	else:
		return 1 / (1 + math.exp(-z))

def init_net(dim, vocab_size):
	# Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
	tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
	syn0 = np.ctypeslib.as_ctypes(tmp)
	syn0 = Array(syn0._type_, syn0, lock=False)

	# Init syn1 with zeros
	tmp = np.zeros(shape=(vocab_size, dim))
	syn1 = np.ctypeslib.as_ctypes(tmp)
	syn1 = Array(syn1._type_, syn1, lock=False)

	return (syn0, syn1)

def train_process(pid):
	# Set fi to point to the right chunk of training file
	start = vocab.bytes / num_processes * pid
	end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
	fi.seek(start)
	#print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)
	
	print (pid, start, end, fi.seek(start))


	alpha = starting_alpha

	word_count = 0
	last_word_count = 0

	while fi.tell() < end:
		line = fi.readline().strip()
		# Skip blank lines
		if not line:
			continue
		print (pid, line)
		# Init sent, a list of indices of words in line
		sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])
		print (pid, sent)
		
		for sent_pos, token in enumerate(sent):
			if word_count % 10000 == 0:
				global_word_count.value += (word_count - last_word_count)
				last_word_count = word_count

				# Recalculate alpha
				alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
				if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

				# Print progress info
				#sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
				#				 (alpha, global_word_count.value, vocab.word_count,
				#				                float(global_word_count.value) / vocab.word_count * 100))
				#sys.stdout.flush()

			# Randomize window size, where win is the max window size
			current_win = np.random.randint(low=1, high=win+1)
			context_start = max(sent_pos - current_win, 0)
			context_end = min(sent_pos + current_win + 1, len(sent))
			context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?
			print ("...", pid, vocab[token], context)
			return
			# CBOW
			if cbow:
				# Compute neu1
				neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
				assert len(neu1) == dim, 'neu1 and dim do not agree'

				# Init neu1e with zeros
				neu1e = np.zeros(dim)

				# Compute neu1e and update syn1
				if neg > 0:
					classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
				else:
					classifiers = zip(vocab[token].path, vocab[token].code)
				for target, label in classifiers:
					z = np.dot(neu1, syn1[target])
					p = sigmoid(z)
					g = alpha * (label - p)
					neu1e += g * syn1[target] # Error to backpropagate to syn0
					syn1[target] += g * neu1	        # Update syn1

				# Update syn0
				for context_word in context:
					syn0[context_word] += neu1e

			# Skip-gram
			else:
				for context_word in context:
					# Init neu1e with zeros
					neu1e = np.zeros(dim)

					# Compute neu1e and update syn1
					if neg > 0:
						classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
					else:
						classifiers = zip(vocab[token].path, vocab[token].code)
					for target, label in classifiers:
						z = np.dot(syn0[context_word], syn1[target])
						p = sigmoid(z)
						g = alpha * (label - p)
						neu1e += g * syn1[target]				         # Error to backpropagate to syn0
						syn1[target] += g * syn0[context_word] # Update syn1

					# Update syn0
					syn0[context_word] += neu1e

			word_count += 1

	# Print progress info
	global_word_count.value += (word_count - last_word_count)
	sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
					 (alpha, global_word_count.value, vocab.word_count,
					                float(global_word_count.value)/vocab.word_count * 100))
	sys.stdout.flush()
	fi.close()

def save(vocab, syn0, fo, binary):
	print ('Saving model to', fo)
	dim = len(syn0[0])
	if binary:
		fo = open(fo, 'wb')
		fo.write('%d %d\n' % (len(syn0), dim))
		fo.write('\n')
		for token, vector in zip(vocab, syn0):
			fo.write('%s ' % token.word)
			for s in vector:
				fo.write(struct.pack('f', s))
			fo.write('\n')
	else:
		invec = {}
		for token, vector in zip(vocab, syn0):
			invec[token.word] = np.array([x for x in vector])
		with open(fo, "wb") as f: pickle.dump(invec, f, pickle.HIGHEST_PROTOCOL)

def __init_process(*args):
	global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
	global win, num_processes, global_word_count, fi
	
	vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
	fi = open(args[-1], 'r')
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		syn0 = np.ctypeslib.as_array(syn0_tmp)
		syn1 = np.ctypeslib.as_array(syn1_tmp)

def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary):
	# Read train file to init vocab
	vocab = Vocab(fi, min_count)
	for x in vocab: print (x.word, x.count)

	# Init net
	syn0, syn1 = init_net(dim, len(vocab))
	print (type(syn0))
	print (type(syn1))
	
	global_word_count = Value('i', 0)
	table = None
	if neg > 0:
		print ('Initializing unigram table')
		table = UnigramTable(vocab)
		#print (table.table)
		#print (table.table.shape)
		#print ([np.count_nonzero(table.table == x) for x in range(len(vocab))])
		#exit()
	else:
		print ('Initializing Huffman tree')
		vocab.encode_huffman()

	# Begin training using num_processes workers
	t0 = time.time()
	pool = Pool(processes=num_processes, initializer=__init_process,
				initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
						                win, num_processes, global_word_count, fi))
	pool.map(train_process, range(num_processes))
	t1 = time.time()
	print ("")
	print ('Completed training. Training took', (t1 - t0) / 60, 'minutes')

	# Save model to file
	save(vocab, syn0, fo, binary)

def main(argv):

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
	
	print ("Got sentences list \n")

	print ("Now training with word2vec algorithm\n\n")
	t0 = time.time()
	model = Word2Vec(contentpath=processedfilepath, min_count=3, size=1000, sg=1, hs=0, negative=30, iters=20,
					 window=8, compute_loss=True, workers=3, alpha=0.025, batch_words=10000)
	model.build_model()
	t1 = time.time()
	print ("Done. Took time: ", t1-t0, "secs\n\n")

	print ("Saving model to disk ", args.output)
	model.save(args.output)
	print ("Saved\n\n")


if __name__ == '__main__':
	x = 0 
	if x == 1: main(sys.argv)
	else:
		parser = MyArgParser()
		parser.add_argument('--processedfilepath', help='Training file', dest='fi', required=True)
		parser.add_argument('--output', help='Output model file', dest='fo', required=True)
		parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=0, type=int)
		parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=30, type=int)
		parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=1000, type=int)
		parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
		parser.add_argument('-window', help='Max window length', dest='win', default=8, type=int) 
		parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=3, type=int)
		parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=3, type=int)
		parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
		#TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
		args = parser.parse_args()

		train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.win,
					args.min_count, args.num_processes, bool(args.binary))



