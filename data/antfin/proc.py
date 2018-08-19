import pandas as pd
import re
import jieba
import numpy as np
import os
import pickle as pkl
from collections import Counter

content = 'content'
tag = '菜名'
data = pd.read_excel('antfin.xls')
data = data.loc[:, [content, tag]]
#print(data[0:100])
print(data)

def proc(data, save_dir = '.'):
    CONTENT = 'content'
    LABEL = '菜名'

    def get_spans(txt, tokens):
        #print(txt, tokens)
        offset = 0
        spans = []
        for token in tokens:
            _offset = txt.find(token, offset)
            #print(token, _offset)
            spans.append([token, _offset, _offset+len(token)])
            if _offset != -1 and _offset <= offset + 2 + len(token):
                offset = _offset + len(token)
        return spans

    def proc_sentence(row):
        records = []
        if CONTENT not in row or LABEL not in row:
            return [], 0, 0
        if type(row[CONTENT]) != str:
            return [], 0, 0
        raw_text = row[CONTENT].lower()
        if type(row[LABEL]) != str: 
            return [], 0, 0
        opinions = row[LABEL].lower()
        #print(opinions)

        if opinions.endswith('<O>') or not opinions:
            return [], 0, 0

        # make sure the target words will be separated by inserting spaces
        text_for_tokenization = raw_text[:].replace('/', ' / ')
        for opinion in opinions.split('#|#'):
            #print(opinion)
            range_list = list(map(lambda x: int(x), re.findall(r"\d+", opinion)))
            #print(range_list)
            if len(range_list) != 2:
                continue
            range_list = [range_list[0] - 1, range_list[1]]
            aspect = raw_text[range_list[0]: range_list[1]]
            #print(aspect)
            text_for_tokenization = text_for_tokenization.replace(aspect, ' ' + aspect + ' ')
        
        tokens = list(jieba.cut(text_for_tokenization))
        tokens = [token for token in tokens if token != ' ']
        #print(list(tokens))
        spans = get_spans(raw_text, tokens)
        #print(spans)
        char_idx_to_word_idx = {s[1]: idx for idx, s in enumerate(spans)} # map origin index to the tokenized words
        #print(char_idx_to_word_idx)
        
        cnt_warn = 0
        cnt_pass = 0
        for opinion in opinions.split('#|#'):
            tags = ['O'] * len(spans)
            range_list = list(map(lambda x: int(x), re.findall(r"\d+", opinion)))
            if len(range_list) != 2:
                continue
            range_list = [range_list[0]-1, range_list[1]]


            polarity = re.findall(r"<.+|>", opinion)[0][1:-2]
            if polarity == '正向':
                polarity = 'positive'
            elif polarity == '负向':
                polarity = 'negative'
            elif polarity == '中立':
                polarity = 'negative'
            else:
                continue

            sidx = range_list[0]
            eidx = range_list[1]
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
                print('Warning', opinions, sidx, tokens, text_for_tokenization, spans, list(zip([s[0] for s in spans], tags)))
                #raise Exception('warning', tokens, text_for_tokenization, sidx, spans, zip([s[0] for s in spans], tags))
                cnt_warn += 1
            else:
                record = {}
                record['tokens'] = tokens
                record['tags'] = tags
                record['polarity'] = polarity
                records.append(record)
                #print(record)
                cnt_pass += 1
        return records, cnt_warn, cnt_pass
    
    train_data = data
    #test_data = data[9000:]
    #print(test_data)
    train_samples = []
    #words_cnt = Counter()
    cnt = 0
    cnt_warn, cnt_pass = 0, 0
    for _, sentence in train_data.iterrows():
        data_sample, n1, n2 = proc_sentence(sentence)
        cnt_warn += n1
        cnt_pass += n2
        print(cnt_warn, cnt_pass)
        train_samples.extend(data_sample)
        if len(data_sample) != 0:
            cnt += 1
        #words_cnt.update(data_sample[0])
        #words_cnt.update(wt(sentence.find('text').text.lower()))
    #vocab = words_cnt.most_common(50000)
    #vocab = {w[0]:idx+2 for idx, w in enumerate(vocab)}
    #vocab['EOS'] = 0
    #vocab['UNK'] = 1
    #dev_samples = train_samples[:200]
    #train_samples = train_samples[200:]
    dev_samples = []

    #test_samples = []
    #for _, sentence in test_data.iterrows():
        #data_sample, _, _ = proc_sentence(sentence)
        #test_samples.extend(data_sample)
    
    print(Counter([s['polarity'] for s in train_samples]))

    random_index = np.random.permutation(len(train_samples))
    samples = [train_samples[idx] for idx in random_index]
    train_samples = samples[:5529]
    dev_samples = []
    test_samples = samples[5529: 7529]
    unlabel_samples = samples[7529:]
    
    print('number of train samples:', len(train_samples))
    print('number of dev samples:', len(dev_samples))
    print('number of test samples:', len(test_samples))
    print('number of unlabel samples:', len(unlabel_samples))
    #x = [[vocab[w] for w in sample[0]] for sample in train_samples]
    #y = [sample[1] for sample in train_samples]
    
    def save_data(data, fn):
        data_fn = os.path.join(save_dir, fn)
        if not os.path.exists(data_fn):
            print('data already exists', fn)
        else:
            pkl.dump(data, open(data_fn, 'wb'))
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
    save_data(unlabel_samples, 'unlabel.' + ext)

proc(data)
