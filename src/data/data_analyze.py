import nltk
from nltk.tokenize import word_tokenize as wt
from collections import Counter
import xml.etree.ElementTree as ET
import pickle as pkl
import sys, os
import codecs
import numpy as np


"""
def find_all_aspect_terms(fn):
    train_data = ET.parse(fn).getroot()
    aspect_terms = set()
    for review in train_data:
        for sentences in review:
            for sentence in sentences:
                opinions = sentence.find('Opinions')
                if opinions is not None:
                    for opinion in opinions:
                        if 'target' in opinion.attrib:
                            aspect_terms.add(opinion.attrib['target'])
    return aspect_terms

at = find_all_aspect_terms('../../data/se2016task5/raw/subtask1/rest_en/ABSA16_Restaurants_Train_SB1_v2.xml')
with codecs.open('../../data/se2016task5/proc/data_analysis/aspect_terms_rest_en', 'w', encoding='utf-8') as f:
    for a in at:
        f.write(a + '\n')
"""

def length_distribution(data_fn):
    data = pkl.load(open(data_fn))
    x = data['x']
    lens = [len(xi) for xi in x]
    print(np.histogram(lens))

length_distribution('../../data/se2016task5/proc/aspect_term_extraction/rest_en//data.pkl')

