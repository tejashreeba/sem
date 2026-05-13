import nltk
from collections import Counter
import math
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

from nltk import word_tokenize,sent_tokenize,pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer

text = "The words of a speech appearing in print.Words, as of a libretto, that are set to music in a composition"
file=open("C:/Users/akash/Downloads/text analytics/input.txt","r")
text=file.read()
file.close()

words = word_tokenize(text)
print(words)
sentences = sent_tokenize(text)

stopwords = stopwords.words('english')
cleaned_words = []

for word in words:
    if word.lower not in stopwords:
        cleaned_words.append(word)

print(cleaned_words)
print(pos_tag(cleaned_words))

stemmed = []
stemmer = PorterStemmer()
for word in cleaned_words:
    stemmed.append(stemmer.stem(word))
print(stemmed)

lemmatized = []
lemmatizer = WordNetLemmatizer()
for word in cleaned_words:
    lemmatized.append(lemmatizer.lemmatize(word))

docs = [
    "The words of a speech appearing in print.",
    "Words, as of a libretto, that are set to music in a composition"
]


for doc in docs:
    words = doc.lower().split()
    total_words = len(words)
    word_count = Counter(words)
    print("\nDocument: ",doc)
    for word,count in word_count.items():
        tf = count/total_words
        print(word,"=",tf)



#idf
all_words = set()
for doc in docs:
    words = doc.lower().split()
    all_words.update(words)

# Calculate IDF
for word in all_words:
    
    doc_count = 0
    
    for doc in docs:
        words = doc.lower().split()
        
        if word in words:
            doc_count += 1
    
    idf = math.log(N / doc_count)
    
    print(word, "=", idf)


















