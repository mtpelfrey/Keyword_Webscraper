# Keyword_Webscraper
A webscraper that reads URLs from a priority queue and then scrapes the raw HTML for keywords that describe the webpage.

The python script webscraper.py takes a text file as a parameter. The text file needs to have the file extension *.txt and should be of the form

URL1 , Service_Priority1

URL2 , Service_Priority2

URL3 , Service_Priority3

where the URLs are addresses for webpages and the service priorities are one of Service_ASAP, Service_Standard, or Service_Whenever. There is a sample input file called sample_input.txt in the repository. It is important that the URLs and the Service_Priorities are separated by a string comprised of a space, then a comma, then a space. If no service priority is specified. the URL will be assigned Service_Whenever.

The script can be run from the command line using the command

python webscraper.py input.txt

where input.txt is the input file.

The only other requirement to run the script is to put the folder nltk_data in the same working directory as webscraper.py and webscraper.py must be run from this directory. The nltk_data directory contains the Reuters-21578 training set used to train the TF-IDF algorithm used to extract keywords as well as a list of stop words.
