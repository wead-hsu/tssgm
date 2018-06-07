import time
from sssp.io.random_reader import RandomReader
from sssp.io.batch_iterator import BatchIterator, threaded_generator

def initDataset(fn, prepare_data_func, batch_size, testing=False, in_memory=True):
    if in_memory:
        with open(fn, 'r') as f:
            data = f.read().strip().split('\n')
            dataset = BatchIterator(len(data), batch_size,
                    data=[data],
                    process_func=prepare_data_func,
                    testing=testing)
    else:
        dataset = RandomReader(fn) 
        dataset = BatchIterator(dataset.num_samples, batch_size, 
                data=dataset,
                process_func=prepare_data_func,
                testing=testing)
    return dataset
