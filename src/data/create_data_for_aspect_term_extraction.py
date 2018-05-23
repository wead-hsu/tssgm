import nltk
from nltk.tokenize import word_tokenize as wt
from collections import Counter
import xml.etree.ElementTree as ET
import pickle as pkl
import sys, os
import numpy as np
import codecs 

def create_data_for_aspect_term_extraction(train_fn, test_fn, save_dir,
        vocab_size=50000):
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
            print('data already exists')
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


create_data_for_aspect_term_extraction('../../data/se2016task5/raw/subtask1/rest_en/ABSA16_Restaurants_Train_SB1_v2.xml',
        '../../data/se2016task5/raw/subtask1/rest_en/EN_REST_SB1_TEST.xml.gold',
        '../../data/se2016task5/proc/aspect_term_extraction/rest_en/')

create_data_for_aspect_term_extraction('../../data/se2016task5/raw/subtask1/lapt_en//ABSA16_Laptops_Train_SB1_v2.xml',
        '../../data/se2016task5/raw/subtask2/lapt_en/EN_LAPT_SB2_TEST.xml.gold',
        '../../data/se2016task5/proc/aspect_term_extraction/lapt_en/')
