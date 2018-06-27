import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import sent_tokenize as st
from collections import Counter
import xml.etree.ElementTree as ET
import pickle as pkl
import sys, os
import numpy as np
import codecs 
import logging

def create_labeled_data_for_target_level_absa(train_fn, test_fn, save_dir):
    train_data = ET.parse(train_fn).getroot()
    test_data = ET.parse(test_fn).getroot()

    def get_spans(txt, tokens):
        offset = 0
        spans = []
        for token in tokens:
            offset = txt.find(token, offset)
            spans.append([token, offset, offset+len(token)])
            offset += len(token)
        return spans

    def proc_sentence(sentence):
        records = []
        raw_text = sentence.find('text').text.lower()
        opinions = sentence.find('aspectTerms')
        if opinions is None:
            return []

        # make sure the target words will be separated by inserting spaces
        text_for_tokenization = raw_text[:].replace('/', ' / ')
        for opinion in opinions:
            if 'term' in opinion.attrib and opinion.attrib['term'] is not 'NULL':
                text_for_tokenization = text_for_tokenization.replace(opinion.attrib['term'], ' ' + opinion.attrib['term'] + ' ')
        
        tokens = wt(text_for_tokenization)
        spans = get_spans(raw_text, tokens)
        char_idx_to_word_idx = {s[1]: idx for idx, s in enumerate(spans)} # map origin index to the tokenized words
        #print(char_idx_to_word_idx)

        if opinions is not None:
            for opinion in opinions:
                tags = ['O'] * len(spans)
                if 'from' not in opinion.attrib:
                    continue
                polarity = opinion.attrib['polarity']
                if polarity == 'conflict':
                    continue
                sidx = int(opinion.attrib['from'])
                eidx = int(opinion.attrib['to'])
                if sidx == eidx == 0:
                    continue
                token_sidx, token_eidx = 1000, 0
                tag = 'B'
                for idx in range(sidx, eidx):
                    if idx in char_idx_to_word_idx:
                        token_sidx = min(token_sidx, char_idx_to_word_idx[idx])
                        token_eidx = max(token_eidx, char_idx_to_word_idx[idx])
                        tags[char_idx_to_word_idx[idx]] = 'B'
                for idx in range(token_sidx, token_eidx + 1):
                    tags[idx] = tag
                if sidx not in char_idx_to_word_idx:
                    print('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))
                    #raise Exception('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))
                else:
                    record = {}
                    record['tokens'] = tokens
                    record['tags'] = tags
                    record['polarity'] = polarity
                    records.append(record)
        return records

    train_samples = []
    #words_cnt = Counter()
    cnt = 0
    for sentence in train_data:
        data_sample = proc_sentence(sentence)
        train_samples.extend(data_sample)
        if len(data_sample) != 0:
            cnt += 1
        #words_cnt.update(data_sample[0])
        #words_cnt.update(wt(sentence.find('text').text.lower()))
    #vocab = words_cnt.most_common(50000)
    #vocab = {w[0]:idx+2 for idx, w in enumerate(vocab)}
    #vocab['EOS'] = 0
    #vocab['UNK'] = 1
    random_index = np.random.permutation(len(train_samples))
    #dev_samples = train_samples[:200]
    #train_samples = train_samples[200:]
    dev_samples = []

    test_samples = []
    for sentence in test_data:
        data_sample = proc_sentence(sentence)
        test_samples.extend(data_sample)
    
    print('number of train samples:', len(train_samples))
    print('number of dev samples:', len(dev_samples))
    print('number of test samples:', len(test_samples))
    #x = [[vocab[w] for w in sample[0]] for sample in train_samples]
    #y = [sample[1] for sample in train_samples]
    
    def save_data(data, fn):
        data_fn = os.path.join(save_dir, fn)
        if os.path.exists(data_fn):
            print('data already exists', fn)
        else:
            pkl.dump(data, open(data_fn, 'w'))
            #with codecs.open(data_fn, 'w', encoding='utf-8') as f:
                #for sample in data:
                    #for i in range(len(sample[0])):
                        #f.write(sample[0][i])
                        #f.write(' ')
                        #f.write(sample[1][i])
                        #f.write('\n')
                    #f.write('\n')
            #print('data saved')
    
    ext = 'pkl'
    save_data(train_samples, 'train.' + ext)
    save_data(dev_samples, 'dev.' + ext)
    save_data(test_samples, 'test.' + ext)

def create_unlabeled_data_for_target_level_absa(train_fn, save_dir):
    data_fn = os.path.join(save_dir, 'unlabel.pkl')
    if os.path.exists(data_fn):
        print('data already exists', data_fn)
        return
    f = open(train_fn)
    records = []
    while True:
        line = f.readline()
        if not line:
            break
        while len(line.strip()) == 0:
            line = f.readline()
            if not line:
                break

        tokens, tags = [], []
        while len(line.strip()) != 0:
            tokens.append(line.strip().split(' ')[0])
            tags.append(line.strip().split(' ')[1])
            line = f.readline()
            if not line:
                break

        if not all(map( lambda x: x=='O', tags)):
            i = 0
            while i < len(tags):
                if tags[i] == 'O':
                    i += 1
                else:
                    sidx = i
                    i += 1
                    while i < len(tags) and tags[i] != 'O':
                        i += 1
                    eidx = i
                    record = {}
                    record['tokens'] = tokens
                    record['tags'] = ['O'] * sidx + ['B'] * (eidx - sidx) + ['O'] * (len(tags) - eidx)
                    records.append(record)
        #print(records)
        #print(len(records))
    
    pkl.dump(records, open(data_fn, 'w'))

def clean_unlabel(train_fn, save_fn, min_sen_len=2, max_sen_len=80):
    samples = pkl.load(open(train_fn))
    samples = [sample for sample in samples if len(sample['tokens']) <= max_sen_len and len(sample['tokens']) >= min_sen_len]
    pkl.dump(samples, open(save_fn, 'w'))
    print('cleaned {}'.format(train_fn))
    
if __name__ == '__main__':
    """
    create_labeled_data_for_target_level_absa(train_fn='../../data/se2014task06/Restaurants_Train_Final.xml',
            test_fn='../../data/se2014task06/Restaurants_Test.xml',
            save_dir='../../data/se2014task06/tabsa-rest/')
    
    create_unlabeled_data_for_target_level_absa(train_fn='../../data/extra-rest/eval.seqtag.labeled',
            save_dir='../../data/se2014task06/tabsa-rest/')

    clean_unlabel('../../data/se2014task06/tabsa-rest/unlabel.pkl',
            '../../data/se2014task06/tabsa-rest/unlabel.clean.pkl',
            max_sen_len=80, min_sen_len=2)
    
    """
    """
    create_labeled_data_for_target_level_absa(train_fn='../../data/se2014task06/Laptops_Train_Final.xml',
            test_fn='../../data/se2014task06/Laptops_Test.xml',
            save_dir='../../data/se2014task06/tabsa-lapt/')
    
    create_unlabeled_data_for_target_level_absa(train_fn='../../data/extra-lapt/eval.seqtag.labeled',
            save_dir='../../data/se2014task06/tabsa-lapt/')

    """
    clean_unlabel('../../data/se2014task06/tabsa-lapt/unlabel.pkl',
            '../../data/se2014task06/tabsa-lapt/unlabel.clean.pkl',
            max_sen_len=83, min_sen_len=3)

