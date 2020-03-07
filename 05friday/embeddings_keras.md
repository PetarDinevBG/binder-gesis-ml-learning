

Anne Kroon and Damian Trilling


# Advanced ML: Working with an embedding vectorizer

#### 1. install embedding vectorizer:
pip install embeddingvectorizer

https://github.com/ccs-amsterdam/embeddingvectorizer

#### 2. download the dataset:
https://surfdrive.surf.nl/files/index.php/s/HKR33cTie8NT6Zh

#### 3. Download a pre-trained embedding model:
https://surfdrive.surf.nl/files/index.php/s/5DVO9b2XdNTxfZQ


# Example working with embedding-based vectorizer

```
import embeddingvectorizer
import gensim
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

df = pd.read_pickle('dataset_vermeer.pkl')
X_train, X_test, y_train, y_test = train_test_split (df['text'], df['topic'], test_size = 0.2 , random_state=42)

#embedding_mdl
model = gensim.models.Word2Vec.load('AEM_small_300')

# get embedding model in right format (vector array for each dictionary word)
embedding_mdl = dict(zip(model.wv.index2word, model.wv.syn0))

# count vectorizer:
embedding_vect_count = embeddingvectorizer.EmbeddingCountVectorizer(embedding_mdl, 'mean')

# tfidf:
embedding_vect_tfidf = embeddingvectorizer.EmbeddingTfidfVectorizer(embedding_mdl, 'mean')

# fit and predict
clf = LogisticRegressionCV()
pipe = make_pipeline(embedding_vect_count, clf)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

```

# Example with Keras


```
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


# convert embedding model correct format
model = Word2Vec.load("AEM_small_300")
model.wv.save_word2vec_format('AEM300.txt', binary=False)
AEM = 'AEM300.txt'

def encodeY(Y):
    '''create one-hot (dummies) for output, encode class values as integers
    '''
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = tf.keras.utils.to_categorical(encoded_Y)
    return dummy_y

X_train, X_test, y_train, y_test = train_test_split([t.translate(str.maketrans('', '', string.punctuation)) for t in df['text']], encodeY(df['topic'].map(int)), test_size = 0.2)

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(X_train)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(X_train)
# pad sequences
max_length = max([len(s.split()) for s in X_train])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

embeddings_index = {}
with open(AEM) as f:
    numberofwordvectors, dimensions = [int(e) for e in next(f).split()]
    for line in tqdm(f):
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
       # word = values[0]
       # coefs = np.asarray(values[1:], dtype='float32')
      #  embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
print('Should be {} vectors with {} dimensions'.format(numberofwordvectors, dimensions))

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    words_not_found = 0
    total_words = 0
    debug = []
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, 300))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in tqdm(vocab.items()):
        e = embedding.get(word, None)
        if e is not None:   # if we do not find the word, we do not want to replace anything but leave the zero's
            weight_matrix[i] = e
            total_words+=1
        else:
            words_not_found+=1
            debug.append(word)
    print('Weight matrix created. For {} out of {} words, we did not have any embedding.'.format(words_not_found, total_words))
    return debug, weight_matrix

missingwords, embedding_vectors = get_weight_matrix(embeddings_index, tokenizer.word_index)
len(embedding_vectors), len(Xtrain)

embedding_layer = Embedding(len(tokenizer.word_index)+1, 300, weights=[embedding_vectors], input_length=max_length, trainable=False)
```

define the model
```
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(4, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
```

alternatief model
```
numberoflabels = 4
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(128, 4, activation='relu'))
model.add(MaxPooling1D(4))
model.add(MaxPooling1D(4))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=numberoflabels, activation='softmax'))   # voor twee categorien sigmoid, voor 1 tanh
```

compile

```
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
```

fit

```
VALIDATION_SIZE=200
model.fit(Xtrain[:-VALIDATION_SIZE], y_train[:-VALIDATION_SIZE],
          epochs=3, verbose=True,
          validation_data=(Xtrain[-VALIDATION_SIZE:], y_train[-VALIDATION_SIZE:]))

loss, acc = model.evaluate(Xtest, y_test, verbose=True)
print('Test Accuracy: %f' % (acc*100))

```


```
