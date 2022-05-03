import spacy
from negspacy.negation import Negex
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

df= pd.read_csv("/content/data450test.csv")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("negex", config={"ent_types":["PERSON","ORG"]})

def sen(sentence):
  doc = nlp(sentence)
  countT=0
  countF=0
  for e in doc.ents:
    if e._.negex==True:
      countT=countT+1
    elif e._.negex==False:
      countF=countF+1
      # listL=[]
  if countT==0 and countF==0:
    return 0      
  elif countT>=1:
    return 1
  else:
    return 0        
    # print(e.text, e._.negex)

use = df.index

listD=[]
for i in use:
  value=sen(df.coded_claim4[i])
  listD.append(value)
print(listD)

