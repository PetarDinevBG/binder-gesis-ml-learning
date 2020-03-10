# Why was the result not in line with the formula?

On the slide, we said that idf was calculated as log(N/n_t). 
Yet, this was not in line with the example.

The reason is that scikit-learn uses a slightly different formular, namely `log((N+1)/(n_t+1) + 1`, and also normalizes using the Eucledian norm. For more info, see https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
