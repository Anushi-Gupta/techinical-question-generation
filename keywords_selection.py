import pandas as pd
from extract_keywords import *
from model import loadPickle
from model import *
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')


def generateDf(text):
    words = []
    addWordsForParagrapgh(words, text)

    wordColums = ['text', 'titleId', 'paragrapghId', 'sentenceId','wordCount', 'NER', 'POS', 'TAG', 'DEP','shape']
    df = pd.DataFrame(words, columns=wordColums)
    
    return df


def prepareDf(df):
    #One-hot encoding
    wordsDf = oneHotEncodeColumns(df)


    #Add missing colums 
    predictorFeaturesName = 'data/pickles/nb-predictor-features.pkl'
    featureNames = loadPickle(predictorFeaturesName)

    for feature in featureNames:
        if feature not in wordsDf.columns:
            wordsDf[feature] = 0    
                
    #Drop unused columns
    columnsToDrop = ['text', 'titleId', 'paragrapghId', 'sentenceId', 'shape', 'isAnswer']
    wordsDf = wordsDf.drop(columnsToDrop, axis = 1)


    return wordsDf

def predictWords(wordsDf, df):
    
    predictorPickleName = 'data/pickles/nb-predictor.pkl'
    predictor = loadPickle(predictorPickleName)
    
    y_pred = predictor.predict_proba(wordsDf)

    labeledAnswers = []
    for i in range(len(y_pred)):
        labeledAnswers.append({'word': df.iloc[i]['text'], 'prob': y_pred[i][0]})
    
    return labeledAnswers