import sys, os, time
import argparse
import fnmatch
import nltk #only used for stemming (a preprocessing step for text data)
import regex
import string
import random
import itertools
import multiprocessing as mp
import math
import warnings
import pickle
import numpy as np

from scipy.special import expit

class MyArgParser(argparse.ArgumentParser):

	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

class LineSentences:
	'''
	The purpose of this class is to read sentences/documents from a single file.
	The format of the file is such that each document is on a separate line.
	Notice that the file is never really stored in working memory. 
	Rather, sentences are read from the file on the fly.

	Basically, a pointer to the file is initialized and is returned as an object
	to the function that instantiated the object of this class. Then, whenever a
	new sentence is required the calling function just needs to either do __getitem__
	or sentences[key], or can also iterate through the sentences using __iter__ method
	simply using a for loop.

	'''
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


def get_neg_samples(unigram_table, count):

	indices = np.random.randint(low=0, high=len(unigram_table), size=count)
	return [unigram_table[i] for i in indices]

def get_neg_samples_1(unigram_table, count):

	return [unigram_table.searchsorted(np.random.randint(unigram_table[-1])) for x in range(count)]
	#return indices

def get_random_train_sample_from_a_sent(sent, pos_term):

	rand_win = math.ceil(window * random.random())
	s = max(0, pos_term - rand_win)
	e = min(len(sent), pos_term + rand_win + 1)
	return [sent[s : pos_term] + sent[pos_term + 1 : e], sent[pos_term]]

def sigmoid(f):
	if f > 6: return 1.0
	elif f < -6: return 0.0
	else: return 1 / (1 + math.exp(-f))

def save_model(vocab_map,W,Z,out_path):
	invec = {}
	for k in vocab_map.keys():
		invec[k] = np.array([x for x in W[vocab_map[k]]])
	with open(out_path + "/input.vectors", "wb") as f: pickle.dump(invec, f, pickle.HIGHEST_PROTOCOL)

def callp(worker):
	'''
	This is the meat of the code where neural network is trained for each process
	'''
	
	total_sents_in_corpus = len(content_file)	
	m = math.ceil(total_sents_in_corpus / workers)
	start = m * worker 
	end = min(total_sents_in_corpus, start+m)
	
	size = W.shape[1]
	
	local_alpha = alpha
	local_num_words_processed = 0
	last_local_num_words_processed = 0

	print ("\n\nWe are printing progress information below everytime a batch finishes inside a process\n")
	#loop over all the iterations (that is how many corpus is scanned by word2vec)	
	for local_iter in range(iters):
		for sent in content_file[start,end]: #the data chunk for this worker/process is [start,end] so loop over that chunk
			sent = sent.strip().split(" ") #tokenize the current sentence
			if len(sent) < window: continue #no need to process this small sentence whose size is less than window
			
			sent = ['<start>'] + sent + ['<end>'] #add two special terms at the start and end of sentence
			sent = [vocab_map[word] if word in vocab_map else vocab_map['<unk>'] for word in sent] #replace all the terms not in vocab as <unk> special word
				
			for counter_terms in range(len(sent)): #loop over the sentence the length of sentence times (i.e for each word in sentence)
				if (local_num_words_processed + 1) % batch_words == 0: #one batch finishes for this worker/process (print info and update alpha)
					# Update global number of words processed
					curr_num_words_processed.value += (local_num_words_processed - last_local_num_words_processed)
					last_local_num_words_processed = local_num_words_processed
					
					# Update alpha
					local_alpha = alpha * (1 - float(curr_num_words_processed.value) / float(iters * total_words_in_corpus + 1))
					if local_alpha < alpha * 0.0001: local_alpha = alpha * 0.0001
					
					# Print progress info
					sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
								 (local_alpha, curr_num_words_processed.value, total_words_in_corpus,
								                float(curr_num_words_processed.value) / total_words_in_corpus * 100))
					sys.stdout.flush()

				train_sample = get_random_train_sample_from_a_sent(sent, counter_terms) #get a train sample ([context1, context2, ...], focus) pair
				if train_sample is None: continue #if the sample is no good we got a None
				
				context = train_sample[0]
				focus = train_sample[1]
				
				for context_word in context: #for each context word in the window around focus word
					neu1e = np.zeros(size) #initialize a zero array for errors collected for each dimension of vector (size=size of word vecs)

					#Gather labels for focus and negative sample terms
					#focus term gets a label 1, while negative sample terms get labels 0
					#negative sample terms are randomly drawn using the unigram distribution 
					if negative > 0: clfs = [(focus, 1.0)] + [(neg_sample, 0.0) for neg_sample in get_neg_samples(unigram_table, negative)]
					
					# Below we assume the input vector for context word W[context_word] is v_c
					# And we assume the output vector for target word Z[target] is v'_t
					for target, label in clfs:
						f = np.dot(W[context_word], Z[target]) #get net score a single scalar, f = v_c . v'_t
						p = sigmoid(f) #get probability estimate a scalar value, p = sigmoid(u) = 1 / (e^-u + 1)
						g = local_alpha * (label - p) #get gradient for this example a scalar value, g = alpha * (label - p)
						neu1e += g * Z[target] #multiply g with v'_t vector, add this error to neu1e error for all targets (focus + negsamples)
						Z[target] += g * W[context_word] #update output vector, g * v_c 
#						if label == 0.0: loss.value -= math.log(expit(-z)) #loss calculation for negative samples
#						else: loss.value -= math.log( expit(z) ) #loss calculation for positive sample

					W[context_word] += neu1e #update input vector, simply add error 
				local_num_words_processed += 1
	
	curr_num_words_processed.value += (local_num_words_processed - last_local_num_words_processed)
	sys.stdout.write ("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
								 (local_alpha, curr_num_words_processed.value, total_words_in_corpus,
								                float(curr_num_words_processed.value) / total_words_in_corpus * 100))
	sys.stdout.flush()

def init_process(cp, vm, tw, vws, vwscs, w, z, ut, neg, a, ma, win, bws, its, works, cnwp, l):

        #define a lot of global variables to share across processes
        #notice how the local variables we get as arguments are different in names than global ones
	global content_file, vocab_map, sorted_vocab_words, sorted_vocab_words_counts, W, Z, unigram_table, negative
	global alpha, min_alpha, window, batch_words, iters, workers, curr_num_words_processed, total_words_in_corpus, loss

	contentpath, vocab_map, total_words_in_corpus, sorted_vocab_words, sorted_vocab_words_counts, Wt, Zt, unigram_table, negative = cp, vm, tw, vws, vwscs, w, z, ut, neg 
	alpha, min_alpha, window, batch_words, iters, workers, curr_num_words_processed, loss =  a, ma, win, bws, its, works, cnwp, l

	content_file = LineSentences(cp) #open the input sentences file to read

	with warnings.catch_warnings():#if there is no avaialbility of ctypeslib then revert to the np array
		warnings.simplefilter('ignore', RuntimeWarning)
		W = np.ctypeslib.as_array(Wt)
		Z = np.ctypeslib.as_array(Zt)

def train_sg_model_with_ns(contentpath, vocab_map, total_words_in_corpus, sorted_vocab_words, sorted_vocab_words_counts, W, Z, unigram_table, negative, 
				alpha, min_alpha, window, batch_words, iters, workers):

        #how many sentences is each worker/process going to work on is decided as follows
	total_sents_in_corpus = len(LineSentences(contentpath))
	m = math.ceil(total_sents_in_corpus / workers)
	print ("\nWorking with ", workers, " workers, with each worker working on", m, "sentences")

        #we define some global mp values. defining them this way lets us use them in every process globally
	curr_num_words_processed = mp.Value('i', 0)	
	loss = mp.Value('d', 0.0)

        #now we call the processes using mp library
	t = time.time()
	pool = mp.Pool(processes=workers, initializer=init_process,
			initargs=(contentpath, vocab_map, total_words_in_corpus, sorted_vocab_words, sorted_vocab_words_counts, W, Z, unigram_table, negative, 
					alpha, min_alpha, window, batch_words, iters, workers, curr_num_words_processed, loss))
	pool.map(callp, range(workers))
	print ("\nTook time to train only:", time.time() - t, "sec") 
	
        #finally we save the model
	save_model(vocab_map, W, Z, "result/")

def init_model(vocab, size):
	'''
	initializing model 
	weight/parameters from input to proj layer W init random
	W in MxN. N is the inherent dims of vectors and M is the total number of words in the vocab
	Z is also MxN weight/parameter matrix from proj to output.
	Z is init as all 0s.
		  _	 					_
	W = 	 |1		 2 	... 		N	|
		1|0.1		0.2			0.4 	|
		.| .		 .			 .	|
		.| .		 .			 .	|
		.| .		 .			 .	|
		M|0.2		0.1			0.8 	|
		 |_						|
	
	Z =  	 |1		 2 		... 	N	|
		1|0.0		0.0			0.0 	|
		.| .		 .			 .	|
		.| .		 .			 .	|
		.| .		 .			 .	|
		M|0.0		0.0			0.0 	|
		 |_						|

	'''
       
	#Create and return ctypes objects from a numpy arrays of W and Z
	#This is done for fast processing
	#ctypes is an advanced Foreign Function Interface package
	W = np.ctypeslib.as_ctypes(np.random.uniform(low=-0.5/size, high=0.5/size, size=(len(vocab), size)))	
	Z = np.ctypeslib.as_ctypes(np.zeros(shape=(len(vocab),size)))

	#Initialize multiprocessing Array using the ctypes based array
	#Return ctypes arrays allocated from shared memory using mp Array call
	#Notice that we have not placed any locks in the arrays
	#That is different processes can simultaneously access and write to arrays
	#In other words it is not process safe
	#The SGD therefore that happens to these arrays is what is called 
	#Asynchronous SGD where model parameters W and Z are adjusted by each
	#process simultaneously. 
	#Why? It just works!
	W = mp.Array(W._type_, W, lock=False)
	Z = mp.Array(Z._type_, Z, lock=False)

	return (W, Z)

def build_vocab_map_word2int(sorted_vocab_words):
	'''
	Each word in the vocab is mapped to an integer so that we can acess its 
	vectors from the matrices W and Z using just indexes
	'''
	vocab_map = {}
	c = 0
	for word in sorted_vocab_words:
		vocab_map[word] = c
		c += 1
	return vocab_map

def trim_vocab(vocab, min_count):
	'''
	For removing less frequent words from the vocab.
	If count of a certain word is < min_count it is removed from vocab
	and not considered while training.

	'''

	vocab['<unk>'] = 0
	words2del = []
	for k in list(vocab.keys()):
		if vocab[k] < min_count:
			words2del.append(k)
			vocab['<unk>'] += 1
	for k in words2del: del vocab[k]
	if '<unk>' not in vocab: vocab['<unk>'] = 0
	return vocab

def build_sorted_vocab(vocab):
	'''
	form two lists. one of sorted vocab words. the other of its count.
	'''
	
	sorted_vocab_words = []
	sorted_vocab_words_counts = []
	for w in sorted(vocab, key=vocab.get, reverse=True):
		sorted_vocab_words.append(w)
		sorted_vocab_words_counts.append(vocab[w])
	return sorted_vocab_words, sorted_vocab_words_counts

	
def build_unigram_table(sorted_vocab_words_counts, ns_exponent):
	'''
	If U(w)	is the unigram distribution (i.e. sorted_vocab_words_counts)
	Then Z = sum_w(U(w) ^ 0.75)
	
	We define a new unigram distribution as U(w)^0.75 / Z

	'''
	
	#Some useful variables
	domain = 2**31 - 1
	vocab_size = len(sorted_vocab_words_counts)
	
	#Initialize unigram table to zeros
	unigram_table = np.zeros(vocab_size, dtype=np.uint32)

	#Finding Z	
	sum_all_powered_vals = 0.0
	for idx in range(vocab_size):
		sum_all_powered_vals += sorted_vocab_words_counts[idx] ** ns_exponent
	
	#Finding U(w)^0.75 / Z
	p = 0.0
	for idx in range(vocab_size):
		p += sorted_vocab_words_counts[idx] ** ns_exponent
		unigram_table[idx] = round(p / sum_all_powered_vals * domain)
	if len(unigram_table) > 0: 
		assert unigram_table[-1] == domain

	#Normalize the unigram (basically by dividing by the max value in unigram)	
	tmp = [int(x) for x in unigram_table / unigram_table[0]]
	
	#If the normalized unigram distribution is like:
	#[6, 3, 2]
	#Get the list as follows
	#[[1,1,1,1,1,1],[2,2,2],[3,3]]
	#Based on the unigram distribution create a list of lists of repeated values of 
	#the index of the words in the vocab
	tmp = [list(itertools.repeat(x,y)) for x,y in zip(range(vocab_size), tmp)]
	
	#flatten the list and return
	return [item for sublist in tmp for item in sublist]
	
def learn_vocab(contentpath):
	'''
	This is where the corpus is scanned and relevant information is extracted.
	This function is called at the start of the modeling.
	Learning the vocab means populating the self.vocab dictionary with the words
	found in the corpus along with their total count in the corpus.
	
	So if there are 4 words in the corpus "dog", "cat", "is", and "am" with 
	frequencies/counts 4, 3, 10, and 1, respectively, then we will end up with a 
	vocab {"cat":3, "dog":4, "am":1, "is":10}.
	
	This function also calls the build unigram table function which basically
	builds a cumulative distribution table. This table is used to get a random
	word out from the unigram to the power 3/4 distribution.
	'''

	sentences = LineSentences(contentpath) #initialize sentence object
	num_sents = len(sentences)
	count_sent = 0
	#initialize a vocab of dict type and start the counts for some special words
	vocab = {}
	vocab['<start>'] = 0 
	vocab['<end>'] = 0
	vocab['<unk>'] = 0
	total_words_in_corpus = 0
	#loop over all the sentences (notice these sentences are not in memory, rather, read from file)
	for sent in sentences:

		#increase count for <start> and <end> words for each new sentence
		vocab['<start>'] += 1 
		vocab['<end>'] += 1
		total_words_in_corpus += 2
		
		#print a progress report as to how many sentences are done
		percent_sent_done = (count_sent/num_sents)*100 
		if percent_sent_done % 5==0: print ("PROGRESS:", percent_sent_done, "%, Working on sentence #",
							 count_sent, "out of total", num_sents, "sentences")
		count_sent += 1
		
		#loop over each word in the sentence and increase its count in vocab
		for w in sent.split(" "):
			total_words_in_corpus += 1
			w = w.strip()
			if w in vocab:
				vocab[w] += 1
			else:
				vocab[w] = 1

	print ("\nThere are total ", total_words_in_corpus, " words in corpus of size", num_sents)
	print ("Out of which ", len(vocab), "are distinct words")
	return vocab, total_words_in_corpus

def build_sg_model(contentpath, vocab_map, total_words_in_corpus, sorted_vocab_words, sorted_vocab_words_counts, W, Z, negative, 
			alpha, min_alpha, window, batch_words, iters, workers, ns_exponent):
	'''
	This function builds the skipgram model

	'''
	print ("\n\n\nBuilding skipgram model...\n")
	
	print ("But first we need to build the unigram table")
	print ("For this we need sorted_vocab_words_counts and negative sampling exponent = 0.75 (taken from paper)")
	unigram_table = build_unigram_table(sorted_vocab_words_counts, ns_exponent)
	print ("Done with building table\n")
	print ("If the normalized unigram distribution is like:")
	print ("[6, 3, 2]")
	print ("Get the list as follows")
	print ("[[1,1,1,1,1,1],[2,2,2],[3,3]]")
	print ("Based on the unigram distribution create a list of lists of repeated values of ")
	print ("the index of the words in the vocab")

	if negative > 0:
		train_sg_model_with_ns(contentpath, vocab_map, total_words_in_corpus, sorted_vocab_words, sorted_vocab_words_counts, W, Z, unigram_table, negative, 
					alpha, min_alpha, window, batch_words, iters, workers)
		print ("\n\nDONE WITH TRAINING SG model with Negative Sampling")

def word2vec(contentpath=None, sentences=None, size=100, alpha=0.025, window=5, 
			min_count=5, sample=0.001, 
			workers=3, min_alpha=0.0001, sg=1, negative=5, 
			ns_exponent=0.75, cbow_mean=1, iters=5, 
			batch_words=10000, compute_loss=False):

	#Just getting the total number of sentences in the corpus	
	if contentpath: total_sents_in_corpus = len(LineSentences(contentpath))
	elif sentences: total_sents_in_corpus = len(sentences)

	print ("Now learning the vocab of the corpus...\n")
	vocab, total_words_in_corpus = learn_vocab(contentpath)
	print ("Learned vocabulary from corpus\n")
	print ("If there are 4 words in the corpus \"dog\", \"cat\", \"is\", and \"am\" with")
	print ("frequencies/counts 4, 3, 10, and 1, respectively, then we will end up with a")
	print ("vocab {\"cat\":3, \"dog\":4, \"am\":1, \"is\":10}.\n\n\n")
	
	print ("Now trimming vocab and sorting it in decreasing order of the word counts in corpus")
	
	vocab = trim_vocab(vocab, min_count)
	print ("Trimmed")
	print ("After triming vocab we are left with ", len(vocab), "distinct words\n")
	print ("We trim the vocab removing words that are very rare in the corpus.")
	print ("This is controlled by the parameter \"min_count\".")
	print ("If min_count is 2, then \"am\" will be removed in the above example.")
	print ("And we will end up with the vocab {\"cat\":3, \"dog\":4, \"is\":10}.\n\n\n")
	
	sorted_vocab_words, sorted_vocab_words_counts = build_sorted_vocab(vocab)
	print ("Sorted\n")	
	print ("Created two new list structures to store...")
	print ("sorted_vocab_words: a list of words sorted according to frequencies [is, dog, cat]")
	print ("sorted_vocab_words_counts: a list of word counts sorted [10, 4, 3]")
	print ("\n\nDone with trimming and sorting\n\n\n")

	print ("Building a map from vocab words to ints")
	print ("This is done for easy accessing the words using indexes")
	vocab_map = build_vocab_map_word2int(sorted_vocab_words)
	print ("Done with mapping words to ints")
	print ("We would create a map like {\"is\":1, \"dog\":2, \"cat\":3}\n\n\n")

	print ("Initializing parameters W and Z")
	W, Z = init_model(vocab, size)
	print ("Initialized model parameters W (input vectors) and Z (output vectors)\n")
	print ("W in MxN. N is the inherent dims of vectors and M is the total number of words in the vocab")
	print ("Z is also MxN weight/parameter matrix from proj to output.")
	print ("W is initialized randomly, while Z is init as all 0s.")
	print ("  	 _	 					_")
	print ("W = 	 |1		2 	... 		N	|")
	print ("	1|0.1		0.2			0.4 	|")
	print ("	.| .		 .			 .	|")
	print ("	.| .		 .			 .	|")
	print ("	.| .		 .			 .	|")
	print ("	M|0.2		0.1			0.8 	|")
	print ("	 |_					       _|")
	print ("\n\n")
	print ("  	 _	 					_")
	print ("Z =  	 |1		2 		... 	N	|")
	print ("	1|0.0		0.0			0.0 	|")
	print ("	.| .		 .			 .	|")
	print ("	.| .		 .			 .	|")
	print ("	.| .		 .			 .	|")
	print ("	M|0.0		0.0			0.0 	|")
	print ("	 |_					       _|")

	if sg == 1: build_sg_model(contentpath, vocab_map, total_words_in_corpus, sorted_vocab_words, sorted_vocab_words_counts, W, Z, negative,
				   alpha, min_alpha, window, batch_words, iters, workers, ns_exponent)

def main(argv):

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
	model = word2vec(contentpath=processedfilepath, min_count=5, size=500, sg=1, negative=5, iters=1,
					 window=8, compute_loss=True, workers=20, alpha=0.025, batch_words=10000)
	t1 = time.time()
	print ("Done. Took time: ", t1-t0, "secs\n\n")

if __name__ == '__main__':
	main(sys.argv)
