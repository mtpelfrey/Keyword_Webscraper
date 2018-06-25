#!/usr/bin/env python3

'''
This is a script for keyword extraction for a list of webpage URLs that takes as
input a list of URLs and associated priority, enqueues them, and then dequeues
them in the correct priority level. The keyword extraction is implemented using
a Term Frequency - Inverse Document Frequency (TFIDF) algorithm. This requires a
corpus of training data. This particular script uses the Reuters-21578 Corpus of
roughly 10,000 news documents to train the algorithm, though other corpuses
could be used if more information was known about the input. The corpus is
updated and the model retrained on every webpage before it is analyzed.
'''

__author__= "Mason Pelfrey"


import sys #Package for dealing with user inputs
import os #Package for dealing with directories
import requests #Package for requesting the raw HTML from webpages
import re #Regular expression package
import nltk #Natural language processing tools
import itertools #Package to allow for loops to loop over multiple containers
from queue import PriorityQueue #Priority queue data structure
#The lxml, bs4, and string are all used for parsing HTML files
from lxml import html
from bs4 import BeautifulSoup
from string import punctuation
#The following nltk packages are used for cleaning text files and are the source
#of the training data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import reuters
#The following sklearn package are used in the implementation of the TFIDF
#algorithm.
from sklearn.feature_extraction.text import TfidfVectorizer

#Uncomment the following to download the Reuters corpus and the stopwords
#if you are running the script for the first time and the data is not already
#in your file directory.
#nltk.download()

#This adds the current directory to the Reuters corpus and the list of stop words.
#If the file nltk_directory is in the same directory as webscraping.py, the
#Reuters corpus and the list of stopwords should be accessible to the script.
cwd = os.getcwd()
nltk_directory = os.path.join(cwd, "nltk_data")
nltk.data.path.append(nltk_directory)

def enqueueInput(f):
    #This function reads in the URLs and associated Service Priorities and
    #puts them in the priority queue. It returns a priority queue.

    priorityDictionary = {'Service_ASAP' : 0, 'Service_Standard' : 1, 'Service_Whenever' : 2}
    q = PriorityQueue()
    for line in f:
        #Reads the URL and Service Priority into a 2 element list.
        url = line.rstrip('\n').split(' , ')
        #Assigns the URL an integer priority if the priority is in the standard
        #form, and otherwise assigns the URL the least urgent priority level.
        try:
            if url[1] in priorityDictionary.keys():
                url[1] = priorityDictionary[url[1]]
            else:
                url[1] = 2
        except IndexError:
            url.insert(1,2)
        #Enqueues the URLs with associated priority level.
        q.put((url[1], url[0]))
    return q

def getHTML(url):
    #This function requests the URL from the web and returns the raw HTML if
    #the website responds and otherwise logs the bad URL to a text file for
    #later inspection and returns a boolean with a False value.

    try:
        htmlSource = requests.get(url)
        return [htmlSource, url]
    except requests.exceptions.RequestException as e:
        with open('Bad_URLs.txt', 'a+') as f:
            f.write(url)
            f.write("\n")
        return False

def parseHTML(htmlSource):
    #This function converts the raw HTML file to just the body text. It assumes
    #that all body text on a website is contained within a CSS class containing
    #the word paragraph, or within paragraph "p" html tags.

    textToAnalyze = "" #Initialize the string.
    #The following converts the raw HTML to a BeautifulSoup file with a useful
    #tree structure.
    soup = BeautifulSoup(htmlSource.text, 'lxml')
    #print(soup.prettify())
    #Regular expression that allow us to search for CSS classes containing the
    #identifier 'paragraph'.
    regex = re.compile('.*paragraph.*')
    #The following for loop pulls just the text from the body sections of the
    #webpage.
    for EachPart in itertools.chain(soup.find_all("div", {"class" : regex}),
                    soup.find_all("p"), soup.find_all("li"), soup.find_all("title")):
        textToAnalyze += EachPart.get_text()
    return textToAnalyze

def tokenize(text):
    #This function cleans the raw text for the TFIDF algorithm. It returns a
    #list of tokens, which are single lowercase words from the document.

    min_length = 3 #Ignores 1 and 2 letter words.
    stop_words = stopwords.words('english') + list(punctuation)
    #Turns the text into all lower case and splits it into tuples of individual
    #words.
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words #Removes all stop words.
                  if word not in stop_words]
    #This is a catch all regular expression.
    p = re.compile('[a-zA-Z]+')
    #This functions ignores 1 and 2 letter words.
    filtered_tokens = list(filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         words))
    return filtered_tokens

def tf_idf(docs):
    #This function returns the TFIDF vectors without computing their values.
    #It takes as input the training corpus.

    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=0,
                        use_idf=True, sublinear_tf=True,
                        norm='l2')
    tfidf.fit(docs)
    return tfidf

def gatherCorpusText():
    #This function initializes the training set. It returns a list whose
    #elements are the raw text of the training documents. The Reuters Corpus
    #contain both a test set and a train set. This function filters out the
    #test set.

    documents = reuters.fileids()
    #train contains all of the file Ids tagged as being part of the training
    #set.
    train = list(filter(lambda doc: doc.startswith("train"), documents))
    train_docs = []
    #This for loop appends the raw text of the training files to the list of
    #training documents.
    for doc_id in train:
        train_docs.append(reuters.raw(doc_id))
    return train_docs

def keyword_values(doc, representer):
    #This function computes the TFIDF values for doc and returns a list
    #of keywords and associated TFIDF values.

    #doc_representation is the document-term matrix for doc computed with the
    #TFIDF vectorizer fit to the training data and the doc.
    doc_representation = representer.transform([doc])
    #Features is an array of terms (keywords) for the document-term matrix and
    #associated index values.
    features = representer.get_feature_names()
    #The list comprehension in the return statement returns the features
    #(keywords) with the TFIDF value for all the features in the doc.
    return [(features[index], doc_representation[0, index])
                 for index in doc_representation.nonzero()[1]]

def print_to_file(TFIDF_List, url):
    #This writes the output to a file.

    with open("keywords_output.txt", "a+") as output:
        output.write(url)
        output.write("\n")
        output.write("The keywords for this webpage are:")
        output.write("\n")
        for i in range(5):
            output.write(TFIDF_List[i][0])
            output.write("\n")
        output.write("\n")

def print_to_terminal(TFIDF_List, url):
    #This writes the output to the terminal.

    print(url)
    print("The keywords for this webpage are:")
    for i in range(5):
        print(TFIDF_List[i][0])
    print("")

def getKey(item):
    #This function helps print the correct index from an array.
    return item[1]

def getFilename():
    #This function reads in the filename of the input file as a parameter of
    #the script, or asks the user for the name of the input file. If the
    #file is the wrong file type, the program exits.

    if sys.argv[-1].lower().endswith(('.txt','.text')):
        filename = sys.argv[-1]
    else:
        filename = input("The input file was not a valid text file. Please "
                  "provide a valid input text file containing URLs and "
                  "Service Priority's.\nSee the README for the correct format:")

    if filename.lower().endswith(('.txt','.text')):
        return filename
    else:
        print("That was not a valid input file. Run the script again"
              "with a valid input file. The program will exit now.")
        sys.exit()

def main():
    #The main function reads in the sample file and calls all of the functions
    #needed to train the TFIDF model and analyze the sample file.

    filename = getFilename()

    #The try except statement opens the file if possible and reads the contents
    #into the queue. If there is an input/output error, the program exits.
    try:
        with open(filename) as f:
            #Opens the sample file and enqueues the entries according to priority
            #level.
            q = enqueueInput(f)
    except IOError as e:
        print("The input file cannot be found or is corrupted. Please "
              "fix the input file and run the program again. The program will "
              "exit now.")
        sys.exit()

    #This print statement informs the user what files to expect as output.
    print("The results will display in the terminal and be written to the "
          "output file keywords_output.txt. Any bad URLs will be logged "
          "in the file Bad_URLs.txt.\n\n")

    #This initializes the output file.
    with open("keywords_output.txt", "w+") as output:
        output.write("This is the output file for webscraping.py.\n\n")
    output.close()

    #This initializes the list of bad URLs file.
    with open('Bad_URLs.txt', 'w+') as f:
        f.write("This is a list of bas URLs. This file will be created whether "
                "or not there are any bad URLs.\nBad URLs:\n\n")
    f.close()

    #Reads in the Corpus text and converts it to a list of strings.
    train_docs = gatherCorpusText()

    #The following while loop pulls URLs off the top of the queue, adds them
    #to the train_docs training set, trains the TFIDF model, and then computes
    #TFIDF values for the htmlSource of the URL. This continues until the
    #queue is empty.
    while not q.empty():
        htmlToAnalyze = getHTML(q.get()[1]) #Pulls URL off the queue.
        if htmlToAnalyze != False: #Executes if URL is good.
            url = htmlToAnalyze[1]
            textToAnalyze = parseHTML(htmlToAnalyze[0]) #Parses HTML.
            #This appends the textToAnalyze to the training set so that the
            #vocabulary of the model is updated before computing TFIDF values.
            train_docs.append(textToAnalyze)
            #The tf_idf function is called on the train_docs to inititialize
            #a TFIDF object with the train_docs vocabulary and IDF.
            vectorizer = tf_idf(train_docs)
            #This function turns the TFIDF object into a list of keywords
            #sorted by TFIDF values.
            TFIDF_sorted_list = sorted(keyword_values(textToAnalyze, vectorizer),
                                key=getKey, reverse=True)
            print_to_file(TFIDF_sorted_list, url)
            print_to_terminal(TFIDF_sorted_list, url)

if __name__ == '__main__':
    main()
