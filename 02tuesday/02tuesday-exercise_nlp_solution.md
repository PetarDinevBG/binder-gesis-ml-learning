# NLP assignment
Anne Kroon and Damian Trilling

# A possible solution to the assignment

```python3
from glob import glob
import random
import nltk
from nltk.stem.snowball import SnowballStemmer
import spacy


infowarsfiles = glob('articles/*/Infowars/*')
infowarsarticles = []
for filename in infowarsfiles:
    with open(filename) as f:
        infowarsarticles.append(f.read())


# taking a random sample of the articles for practice purposes
articles =random.sample(infowarsarticles, 10)
```

# Task 1: Preprocessing data

### lowercasing articles

```python3
articles_lower_cased = [art.lower() for art in articles]
```

### removing stopwords

```python3
mystopwords = nltk.corpus.stopwords.words("english")


articles_without_stopwords = []
for article in articles:
    articles_no_stop = ""
    for word in article.lower().split():
        if word not in mystopwords:
            articles_no_stop = articles_no_stop + " " + word
    articles_without_stopwords.append(articles_no_stop)
```


Same solution, but with list comprehension:

```python3
articles_without_stopwords = [" ".join([w for w in article.lower().split() if w not in mystopwords]) for article in articles]
```

### Stemming

```python3
stemmer = SnowballStemmer("english")

stemmed_text = []
for article in articles:
    stemmed_words = ""
    for word in article.lower().split():
        stemmed_words = stemmed_words + " " + stemmer.stem(word)
    stemmed_text.append(stemmed_words)
```

Same solution, but with list comprehension:

```python3
stemmed_text  = [" ".join([stemmer.stem(w) for w in article.lower().split()]) for article in articles]
```


# Task 2: Extract information

```Python3
import nltk

tokens = [nltk.word_tokenize(sentence) for sentence in articles]
tagged = [nltk.pos_tag(sentence) for sentence in tokens]
print(tagged[0])
```

playing around with Spacy:

```python3
nlp = spacy.load('en')

doc = [nlp(sentence) for sentence in articles]
for i in doc:
    for ent in i.ents:
        if ent.label_ == 'PERSON':
            print(ent.text, ent.label_ )

```          
