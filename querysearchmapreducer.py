#### CS5344 LAB1 #####
#### A0176605B Wang Jia #####
from pyspark import SparkContext, SparkConf
import os, string, math
import numpy as np

## Stage 1 Convert text to lower case and remove punctuations
def rm_punctuation(text):
    # converted = text.encode('utf-8')
    lowercased_str = text.lower().replace('--', ' ')
    translator = str.maketrans('', '', string.punctuation)
    clean_str = lowercased_str.translate(translator)
    return clean_str

## Stage 2 Compute tf-idf for each word@doc
def compute_tfidf(val):
    freqs = list(val)
    df = len(val)
    result = []
    for freq in freqs:
        doc = freq.split('=')[0]
        tf = int(freq.split('=')[1])
        tfidf = format((1 + np.log(tf)) * np.log(10 / df), '.3f')
        result.append(doc + '=' + tfidf)
    return result

## Stage 2 Convert (word, (doc=tfidf)) to (word@doc, tfidf)
def restore_docid(rdd):
    word, docs = rdd
    result = []
    for doc in docs:
        word_docid = '{0}@{1}'.format(word, doc.split('=')[0])
        tfidf = float(doc.split('=')[1])
        pair = (word_docid, tfidf)
        result.append(pair)
    return result

## Stage 3 Compute normalised tf-idf for each word@doc
def norm_tfidfs(val):
    word_tfidfs = list(val)
    tf_idfs = []
    result = []
    S = 0
    for word_tfidf in word_tfidfs:
        score = float(word_tfidf.split('=')[1])
        tf_idfs.append(score)
        S += score ** 2

    norms = np.array(tf_idfs) / math.sqrt(S)
    for i in range(len(word_tfidfs)):
        word_norm = word_tfidfs[i].split('=')[0] + '=' + str(format(norms[i], '.4f'))
        result.append(word_norm)
    return result

## Stage 3 Convert (word, (doc=norm-tfidf)) to (word@doc, norm-tfidf)
def restore_word(rdd):
    docid, scores = rdd
    result = []
    for score in scores:
        word_docid = '{0}@{1}'.format(score.split('=')[0], docid)
        norm = float(score.split('=')[1])
        pair = (word_docid, norm)
        result.append(pair)    
    return result

## Stage 4 Compute sum of normalised tfidf for each document
def sum_norm_tfidfs(val):
    word_norms = list(val)
    sum_of_norm_tfidfs = 0
    for word_norm in word_norms:
        norm_tfidf = float(word_norm.split('=')[1])
        sum_of_norm_tfidfs += norm_tfidf
    return sum_of_norm_tfidfs


if __name__ == "__main__":
	## Initialize Spark session
	print("Initialize Spark session...")
	conf = SparkConf().setAppName("querySearch")
	sc = SparkContext(conf=conf)

	## Load given stopwords
	f = open('stopwords.txt', 'r')
	stoplist = f.read().splitlines()
	stops = set(stoplist)

	## Collect names of documents
	docs_dir = 'datafiles/'
	doc_names = []
	for filename in os.listdir(docs_dir):
	    doc_names.append(filename)

	## Stage 1 Count word occurences in each document
	print("Stage 1: Count word occurences in each document...")
	no_of_docs = len(doc_names)
	for i in range(no_of_docs):
	    if i == 0:
	        doc = sc.textFile("datafiles/f1.txt")
	        term_freq = doc.flatMap(lambda line: rm_punctuation(line).split()) \
	                	.filter(lambda word: word not in stops) \
	                	.filter(lambda word: len(word) > 3) \
	                	.map(lambda word: (word + '@d1', 1)) \
	                	.reduceByKey(lambda a, b: a + b)
	    else:
	        doc = sc.textFile("datafiles/f" + str(i+1) + ".txt")
	        pairs = doc.flatMap(lambda line: rm_punctuation(line).split()) \
	                .filter(lambda word: word not in stops) \
	                .filter(lambda word: len(word) > 3) \
	                .map(lambda word: (word + '@d' + str(i+1), 1)) \
	                .reduceByKey(lambda a, b: a + b)
	        term_freq = term_freq.union(pairs)
	        print("Number of (word@doc, freq) pairs: " + str(term_freq.count()))
	print("Sample of word-count pairs: " + str(term_freq.take(5)))
	print()

	## Stage 2 Calculate tf-idf based on term frequency and doc frequency
	print("Stage 2: Compute tf-idf...")
	tf_idfs = term_freq.map(lambda pair: (pair[0].split('@')[0], '{0}={1}'.format(pair[0].split('@')[1], pair[1]))) \
                    .groupByKey() \
                    .mapValues(compute_tfidf) \
                    .flatMap(restore_docid)
	print("Sample of word-tfidf pairs: " + str(tf_idfs.take(5)))
	print()

	## Stage 3 Calculate normalised tf-idf
	print("Stage 3: Compute normalised tf-idf...")
	norm_tfidfs = tf_idfs.map(lambda pair: (pair[0].split('@')[1], '{0}={1}'.format(pair[0].split('@')[0], pair[1]))) \
                    	.groupByKey() \
                     	.mapValues(norm_tfidfs) \
                     	.flatMap(restore_word)
	print("Sample of word-normalised_tfidf pairs: " + str(norm_tfidfs.take(5)))
	print()

	## Stage 4 Calculate relevance of every doc with respect to a query
	print("Stage 4: Compute relevance of each doc w.r.t. the given query...")
	query = open('query.txt', 'r')
	query_words = query.read().split()
	relevance = norm_tfidfs.map(lambda pair: (pair[0].split('@')[1], '{0}={1}'.format(pair[0].split('@')[0], pair[1]))) \
                       .filter(lambda x: (x[1].split('=')[0]) in query_words) \
                       .groupByKey() \
                       .mapValues(sum_norm_tfidfs)
	print("Sample of doc-relevance pairs: " + str(relevance.take(5)))
	print()

	## Stage 5 Sort docs by relevance and output the top-k relevant docs
	print("Stage 5: Get top relevant documents...")
	k = 3
	top_docs = relevance.map(lambda pair: (pair[1], pair[0])) \
						.sortByKey(False) \
						.map(lambda pair: ('f' + pair[1][1:] + '.txt', pair[0]))
	top_docs_list = top_docs.take(k)
	print("Top-k relevant documents: " + str(top_docs_list))
	output_file = open('output.txt', 'w')
	for t in top_docs_list:
  		output_file.write(' '.join(str(s) for s in t) + '\n')
	output_file.close()

	sc.stop()