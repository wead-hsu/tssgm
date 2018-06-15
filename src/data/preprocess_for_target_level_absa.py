import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import sent_tokenize as st
from collections import Counter
import xml.etree.ElementTree as ET
import pickle as pkl
import sys, os, json
import numpy as np
import codecs 
import logging
import csv

def process_label_semeval2016_for_bilstmcrf(train_fn, test_fn, save_dir, bio=False):
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
        raw_text = sentence.find('text').text.lower()
        opinions = sentence.find('Opinions')
        if opinions is None:
            return (wt(raw_text), ['O']*len(wt(raw_text)))

        # make sure the target words will be separated by inserting spaces
        text_for_tokenization = raw_text[:].replace('/', ' / ')
        for opinion in opinions:
            if 'target' in opinion.attrib and opinion.attrib['target'] is not 'NULL':
                text_for_tokenization = text_for_tokenization.replace(opinion.attrib['target'], ' ' + opinion.attrib['target'] + ' ')
        
        tokens = wt(text_for_tokenization)
        spans = get_spans(raw_text, tokens)
        char_idx_to_word_idx = {s[1]: idx for idx, s in enumerate(spans)} # map origin index to the tokenized words
        tags = ['O'] * len(spans)
        #print(char_idx_to_word_idx)

        if opinions is not None:
            for opinion in opinions:
                if 'from' not in opinion.attrib:
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
                    if bio:
                        tag = 'I'
                if sidx not in char_idx_to_word_idx:
                    print('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))
                    #raise Exception('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))

        return (tokens, tags)

    train_samples = []
    #words_cnt = Counter()
    for review in train_data:
        for sentences in review:
            for sentence in sentences:
                data_sample = proc_sentence(sentence)
                train_samples.append(data_sample)
                #words_cnt.update(data_sample[0])
                #words_cnt.update(wt(sentence.find('text').text.lower()))
    #vocab = words_cnt.most_common(50000)
    #vocab = {w[0]:idx+2 for idx, w in enumerate(vocab)}
    #vocab['EOS'] = 0
    #vocab['UNK'] = 1
    random_index = np.random.permutation(len(train_samples))
    dev_samples = train_samples[:200]
    train_samples = train_samples[200:]

    test_samples = []
    for review in train_data:
        for sentences in review:
            for sentence in sentences:
                data_sample = proc_sentence(sentence)
                test_samples.append(data_sample)
    
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
            #pkl.dump(data, open(data_fn, 'w'))
            with codecs.open(data_fn, 'w', encoding='utf-8') as f:
                for sample in data:
                    for i in range(len(sample[0])):
                        f.write(sample[0][i])
                        f.write(' ')
                        f.write(sample[1][i])
                        f.write('\n')
                    f.write('\n')
            print('data saved')

    save_data(train_samples, 'train.seqtag.bio')
    save_data(dev_samples, 'dev.seqtag.bio')
    save_data(test_samples, 'test.seqtag.bio')

def process_label_semeval2014_for_bilstmcrf(train_fn, test_fn, save_dir, bio=False):
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
        raw_text = sentence.find('text').text.lower()
        opinions = sentence.find('aspectTerms')
        if opinions is None:
            return (wt(raw_text), ['O']*len(wt(raw_text)))

        # make sure the target words will be separated by inserting spaces
        text_for_tokenization = raw_text[:].replace('/', ' / ')
        for opinion in opinions:
            if 'term' in opinion.attrib and opinion.attrib['term'] is not 'NULL':
                text_for_tokenization = text_for_tokenization.replace(opinion.attrib['term'], ' ' + opinion.attrib['term'] + ' ')
        
        tokens = wt(text_for_tokenization)
        spans = get_spans(raw_text, tokens)
        char_idx_to_word_idx = {s[1]: idx for idx, s in enumerate(spans)} # map origin index to the tokenized words
        tags = ['O'] * len(spans)
        #print(char_idx_to_word_idx)

        if opinions is not None:
            for opinion in opinions:
                if 'from' not in opinion.attrib:
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
                    if bio:
                        tag = 'I'
                if sidx not in char_idx_to_word_idx:
                    print('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))
                    #raise Exception('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))

        return (tokens, tags)

    train_samples = []
    #words_cnt = Counter()
    for sentence in train_data:
        data_sample = proc_sentence(sentence)
        train_samples.append(data_sample)
        #words_cnt.update(data_sample[0])
        #words_cnt.update(wt(sentence.find('text').text.lower()))
    #vocab = words_cnt.most_common(50000)
    #vocab = {w[0]:idx+2 for idx, w in enumerate(vocab)}
    #vocab['EOS'] = 0
    #vocab['UNK'] = 1
    random_index = np.random.permutation(len(train_samples))
    dev_samples = train_samples[:200]
    train_samples = train_samples[200:]

    test_samples = []
    for sentence in test_data:
        data_sample = proc_sentence(sentence)
        test_samples.append(data_sample)
    
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
            #pkl.dump(data, open(data_fn, 'w'))
            with codecs.open(data_fn, 'w', encoding='utf-8') as f:
                for sample in data:
                    for i in range(len(sample[0])):
                        f.write(sample[0][i])
                        f.write(' ')
                        f.write(sample[1][i])
                        f.write('\n')
                    f.write('\n')
            print('data saved')
    
    ext = 'bio' if bio else 'bo'
    save_data(train_samples, 'train.seqtag.' + ext)
    save_data(dev_samples, 'dev.seqtag.' + ext)
    save_data(test_samples, 'test.seqtag.' + ext)

def process_unlabel_rest_for_bilstmcrf(input_fn, output_fn):
    if os.path.exists(output_fn):
        print('data already exists', output_fn)
        return
    input_file = open(input_fn, 'r')
    output_file = open(output_fn, 'w')
    lines = input_file.readlines()
    lines = [w  for s in lines for w in s.split('\t')[1].lower().strip().split('\\n') if len(w) != 0]
    for line in lines:
        sents = st(line)
        for sent in sents:
            tokens = wt(sent)
            for token in tokens:
                output_file.write(token)
                output_file.write(' ')
                output_file.write('O')
                output_file.write('\n')
            output_file.write('\n')

def process_unlabel_lapt_for_bilstmcrf(input_fn, output_fn):
    if os.path.exists(output_fn):
        print('data already exists', output_fn)
        return
    res = []
    for f in os.listdir(input_fn):
        if not f.endswith('json'):
            continue
        f = open(input_fn + '/' + f)
        js = json.load(f)
        f.close()
        reviews = js['Reviews']
        contents = [r['Content'] for r in reviews if r['Content'] is not None]
        res.extend(contents)
    
    with open(output_fn, 'w') as f:
        for content in res:
            content = content.strip().lower()
            sents = st(content)
            for sent in sents:
                tokens = wt(sent)
                for token in tokens:
                    f.write(token.encode('utf-8'))
                    f.write(' O\n')
                f.write('\n')
    
if __name__ == '__main__':
    
    process_label_semeval2014_for_bilstmcrf(train_fn='../../data/se2014task06/Restaurants_Train_Final.xml',
            test_fn='../../data/se2014task06/Restaurants_Test.xml',
            save_dir='../../data/se2014task06/bilstmcrf-rest/')

    process_label_semeval2014_for_bilstmcrf(train_fn='../../data/se2014task06/Laptops_Train_Final.xml',
            test_fn='../../data/se2014task06/Laptops_Test.xml',
            save_dir='../../data/se2014task06/bilstmcrf-lapt/')

    process_label_semeval2014_for_bilstmcrf(train_fn='../../data/se2014task06/Restaurants_Train_Final.xml',
            test_fn='../../data/se2014task06/Restaurants_Test.xml',
            save_dir='../../data/se2014task06/bilstmcrf-rest/',
            bio = True)

    process_label_semeval2014_for_bilstmcrf(train_fn='../../data/se2014task06/Laptops_Train_Final.xml',
            test_fn='../../data/se2014task06/Laptops_Test.xml',
            save_dir='../../data/se2014task06/bilstmcrf-lapt/',
            bio = True)

    process_unlabel_rest_for_bilstmcrf(input_fn='../../data/extra-rest/1-restaurant-train.csv',
            output_fn='../../data/extra-rest/eval.seqtag')
    
    process_unlabel_lapt_for_bilstmcrf(input_fn='../../data/extra-lapt/',
            output_fn='../../data/extra-lapt/eval.seqtag')
