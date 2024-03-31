from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics

nlp = spacy.load('en_core_web_sm') # 'en_core_web_sm'
vader_model = SentimentIntensityAnalyzer()

df = pd.read_csv('sentiment-topic-test.csv',sep=';')
def run_vader(textual_unit,
              lemmatize=False,
              parts_of_speech_to_consider=None,
              verbose=1):
    """
    Run VADER on a sentence from spacy

    :param str textual unit: a textual unit, e.g., sentence, sentences (one string)
    (by looping over doc.sents)
    :param bool lemmatize: If True, provide lemmas to VADER instead of words
    :param set parts_of_speech_to_consider:
    -None or empty set: all parts of speech are provided
    -non-empty set: only these parts of speech are considered.
    :param int verbose: if set to 1, information is printed
    about input and output

    :rtype: dict
    :return: vader output dict
    """
    doc = nlp(textual_unit)

    input_to_vader = []

    for sent in doc.sents:
        for token in sent:

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-':
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add)
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))

    if verbose >= 1:
        print()
        print('INPUT SENTENCE', sent)
        print('INPUT TO VADER', input_to_vader)
        print('VADER OUTPUT')

    return scores


def vader_output_to_label(vader_output):
    """
    map vader output e.g.,
    {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.4215}
    to one of the following values:
    a) positive float -> 'positive'
    b) 0.0 -> 'neutral'
    c) negative float -> 'negative'

    :param dict vader_output: output dict from vader

    :rtype: str
    :return: 'negative' | 'neutral' | 'positive'
    """
    compound = vader_output['compound']

    if compound < 0:
        return 'negative'
    elif compound == 0.0:
        return 'neutral'
    elif compound > 0.0:
        return 'positive'


assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.0}) == 'neutral'
assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.01}) == 'positive'
assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': -0.01}) == 'negative'

sentences = []
all_vader_output = []
gold = []

for sentence in df.text:
    vader_output = run_vader(sentence, lemmatize=True, parts_of_speech_to_consider=None, verbose=1)
    vader_label = vader_output_to_label(vader_output)  # convert vader output to category
    print(vader_label)
    sentences.append(sentence)
    all_vader_output.append(vader_label)

for label in df.sentiment:
    gold.append(label)


print("GOLD",gold)
print(all_vader_output)
print("")
print(classification_report(y_true=gold, y_pred=all_vader_output, target_names=['negative', 'neutral', 'positive']))

confusion_matrix = metrics.confusion_matrix(gold, all_vader_output)

print(confusion_matrix)