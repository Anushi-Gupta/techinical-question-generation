import pandas as pd
from IPython.display import Markdown, display, clear_output
from model import *
from extract_keywords import *
from extract_quest import *
from keywords_selection import *
from extract_keywords import *
from generating_options import *

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

def generateQuestions(text, count):
    
    # Extract words 
    df = generateDf(text)
    wordsDf = prepareDf(df)
    
    print(df)
    print(wordsDf)
    labeledAnswers = predictWords(wordsDf, df)
    
    # Transform questions
    qaPairs = addQuestions(labeledAnswers, text)
    
    # Pick the best questions
    orderedQaPairs = sortAnswers(qaPairs)
    
    # Generate distractors
    questions = addDistractors(orderedQaPairs[:count], 4)
    
    # Print
    for i in range(count):
        display(Markdown('### Question ' + str(i + 1) + ':'))
        print(questions[i]['question'])

        display(Markdown('#### Answer:'))
        print(questions[i]['answer'])
        
        display(Markdown('#### Incorrect answers:'))
        for distractor in questions[i]['distractors']:
            print(distractor)
        
        print()
text = "Oxygen is a chemical element with symbol O and atomic number 8. It is a member of the chalcogen group on the periodic table, a highly reactive nonmetal, and an oxidizing agent that readily forms oxides with most elements as well as with other compounds. By mass, oxygen is the third-most abundant element in the universe, after hydrogen and helium. At standard temperature and pressure, two atoms of the element bind to form dioxygen, a colorless and odorless diatomic gas with the formula O2. Diatomic oxygen gas constitutes 20.8% of the Earth's atmosphere. As compounds including oxides, the element makes up almost half of the Earth's crust."
        
generateQuestions(text,10)
        