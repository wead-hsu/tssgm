import nltk
from nltk.tokenize import word_tokenize as wt
from collections import Counter
import xml.etree.ElementTree as ET
import pickle as pkl
import sys, os

def create_data_for_aspect_term_extraction(train_fn, test_fn, save_dir,
        vocab_size=50000):
    train_data = ET.parse(train_fn).getroot()

    def get_spans(txt, tokens):
        offset = 0
        spans = []
        for token in tokens:
            offset = txt.find(token, offset)
            spans.append([token, offset, offset+len(token)])
            offset += len(token)
        return spans

    def proc_sentence(sentence):
        raw_text = sentence.find('text').text.lower()
        opinions = sentence.find('Opinions')
        if opinions is None:
            return (wt(raw_text), [0]*len(wt(raw_text)))

        # make sure the target words will be separated by inserting spaces
        text_for_tokenization = raw_text[:].replace('/', ' / ')
        for opinion in opinions:
            if 'target' in opinion.attrib and opinion.attrib['target'] is not 'NULL':
                text_for_tokenization = text_for_tokenization.replace(opinion.attrib['target'], ' ' + opinion.attrib['target'] + ' ')
        
        tokens = wt(text_for_tokenization)
        spans = get_spans(raw_text, tokens)
        char_idx_to_word_idx = {s[1]: idx for idx, s in enumerate(spans)} # map origin index to the tokenized words
        tags = [0] * len(spans)
        #print(char_idx_to_word_idx)

        if opinions is not None:
            for opinion in opinions:
                if 'from' not in opinion.attrib:
                    continue
                sidx = int(opinion.attrib['from'])
                eidx = int(opinion.attrib['to'])
                if sidx == eidx == 0:
                    continue
                token_sidx, token_edix = 1000, 0
                for idx in range(sidx, eidx):
                    if idx in char_idx_to_word_idx:
                        tags[char_idx_to_word_idx[idx]] = 1
                if sidx not in char_idx_to_word_idx:
                    print('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))
                    #raise Exception('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))

        return (tokens, tags)

    data_samples = []
    words_cnt = Counter()
    for review in train_data:
        for sentences in review:
            for sentence in sentences:
                data_sample = proc_sentence(sentence)
                data_samples.append(data_sample)
                words_cnt.update(data_sample[0])
                #words_cnt.update(wt(sentence.find('text').text.lower()))
    vocab = words_cnt.most_common(50000)
    vocab = {w[0]:idx+2 for idx, w in enumerate(vocab)}
    vocab['EOS'] = 0
    vocab['UNK'] = 1
    
    x = [[vocab[w] for w in sample[0]] for sample in data_samples]
    y = [sample[1] for sample in data_samples]
    
    data = {'x': x,
            'y': y,
            'vocab': vocab,
            }
    
    data_fn = os.path.join(save_dir, 'data.pkl')
    if os.path.exists(data_fn):
        print('data already exists')
    else:
        pkl.dump(data, open(data_fn, 'w'))
        print('data saved')


create_data_for_aspect_term_extraction('../../data/se2016task5/raw/subtask1/rest_en/ABSA16_Restaurants_Train_SB1_v2.xml',
        '../../data/se2016task5/raw/subtask1/rest_en/EN_REST_SB1_TEST.xml.gold',
        '../../data/se2016task5/proc/aspect_term_extraction/rest_en/')

create_data_for_aspect_term_extraction('../../data/se2016task5/raw/subtask1/lapt_en//ABSA16_Laptops_Train_SB1_v2.xml',
        '../../data/se2016task5/raw/subtask2/lapt_en/EN_LAPT_SB2_TEST.xml.gold',
        '../../data/se2016task5/proc/aspect_term_extraction/lapt_en/')
