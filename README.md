Title: 

	A multithreaded implementation of word2vec for finding 
	word embeddings from a huge corpus

Team member: 

	Shayan Ali Akbar (sakbar@purdue.edu)

Goals:
 
	Understand how word embeddings are learned using word2vec 
	based algorithms.

	Understand how different flavors of word2vec, like
	Skip-gram (SG)  and Continuous Bag Of Words (CBOW) work.

	Understand how different heuristics used by word2vec,
	like Negative Sampling (NS) and Hierarchical Softmax (HS) 
	work.

	Implement different flavors of word2vec along with
	the heuristics. We will have four different configurations
	in which word2vec can operate: SG with NS, CBOW with NS,
	SG with HS, and CBOW with HS.

	Learn how to implement things in a multithreaded setting
	using Python's ``multiprocessing'' library.
	
	Use multithreaded implementation of word2vec to improve
	efficiency.

	Apply multithreaded implementation of word2vec to a huge
	corpus (I have ``Billion Words Corpus'' dataset in mind). 
	
	Visualize the word vectors (using three dimensions of 
	vectors). We can use PCA to find the most dominant 
	dimensions.

	Compare the performance of different word2vec flavors for
	semantic similarity and analogy	tasks for quantitative
	analysis.

Challenges:
	
	The word2vec paper itself is very ambiguous and does not
	explain the algorithms in detail. Will have do a literature
	review and look for other resources to understand the
	internal working of the algorithms.

	The original code by the authors of word2vec is written
	in C, while I am going to use Python to implement word2vec
	from scratch. The implementation therefore needs to be 
	highly optimized for it to work in practice as Python 
	programs are notoriously slow if not implemented right.

Restrictions:

	Implementing word2vec from scratch without using any machine
	learning libraries (like pytorch) is a challenge unto itself.

