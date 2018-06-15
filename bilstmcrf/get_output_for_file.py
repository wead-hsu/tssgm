from model.data_utils import pad_sequences, get_chunks
from model.ner_model import NERModel
from model.config import Config, input_args
import tensorflow as tf
import numpy as np

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned

def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)

class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            raw, words, tags = [], [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield raw, words, tags
                        raw, words, tags = [], [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    raw += [word]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    s_batch, x_batch, y_batch = [], [], []
    for (s, x, y) in data:
        if len(x_batch) == minibatch_size:
            yield s_batch, x_batch, y_batch
            s_batch, x_batch, y_batch = [], [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        s_batch += [s]
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield s_batch, x_batch, y_batch

def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(os.path.join(config.dir_model,'-0'))
    #model.restore_session(config.dir_model)
    #model.restore_session('./results/rest/model.weights/-0')
    #model_file = tf.train.latest_checkpoint("./results/rest/model.weights")
    #print(model_file)
    print(config.vocab_tags)

    # create dataset
    if input_args.eval_filename is not None:
        config.filename_test = input_args.eval_filename
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    #model.evaluate(test)
    #interactive_shell(model)
    if input_args.save_filename is not None:
        save_file = open(input_args.save_filename, 'w')
    else:
        save_file = open('/tmp/tmp.txt', 'w')
    idx_to_word = {config.vocab_words[k]: k for k in config.vocab_words}
    idx_to_tag  = {config.vocab_tags[k]: k for k in config.vocab_tags}

    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for raw, words, labels in minibatches(test, config.batch_size):
        labels_pred, sequence_lengths = model.predict_batch(words)
        
        if config.use_chars:
            _, words = zip(*words)

        for lab, lab_pred, length in zip(labels, labels_pred,
                                         sequence_lengths):
            lab      = lab[:length]
            lab_pred = lab_pred[:length]
            accs    += [a==b for (a, b) in zip(lab, lab_pred)]

            lab_chunks      = set(get_chunks(lab, config.vocab_tags))
            lab_pred_chunks = set(get_chunks(lab_pred, config.vocab_tags))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds   += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
            
        for i in range(len(words)):
            for j in range(sequence_lengths[i]):
                save_file.write(raw[i][j])
                save_file.write(' ')
                save_file.write(idx_to_tag[labels_pred[i][j]])
                save_file.write('\n')
            save_file.write('\n')

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)

    print('acc', acc, 'f1', f1)


if __name__ == "__main__":
    main()
