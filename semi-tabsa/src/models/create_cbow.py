def train_word_emb(unlabel, word_emb_path):
    import gensim
    import multiprocessing
    with open(word_emb_path, 'w') as f:
        sentences = []
        for line in unlabel:
            sentences.append(line['tokens'])
        model = gensim.models.Word2Vec(sentences, size=300, min_count=0,
                sg=0, workers=multiprocessing.cpu_count())
        vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
        print(len(vocab))
        for k, v in vocab.items():
            if len(k.split()) == 1:
                f.write(k.strip() + ' ' + ' '.join(map(str, model[k])) + '\n')

import pickle as pkl
unlabel = pkl.load(open('/home/weidi.xwd/workspace//ABSA/data//se2014task06/tabsa-lapt/unlabel.clean.pkl', 'rb'), encoding='latin')
label = pkl.load(open('/home/weidi.xwd/workspace//ABSA/data//se2014task06/tabsa-lapt/train.pkl', 'rb'), encoding='latin')
train_word_emb(unlabel + label, '/home/weidi.xwd/workspace//ABSA/data/se2014task06//tabsa-lapt/cbow.unlabel.300d.txt')

unlabel = pkl.load(open('/home/weidi.xwd/workspace//ABSA/data//se2014task06/tabsa-rest/unlabel.clean.pkl', 'rb'), encoding='latin')
label = pkl.load(open('/home/weidi.xwd/workspace//ABSA/data//se2014task06/tabsa-rest/train.pkl', 'rb'), encoding='latin')
train_word_emb(unlabel + label, '/home/weidi.xwd/workspace//ABSA/data/se2014task06//tabsa-rest/cbow.unlabel.300d.txt')
