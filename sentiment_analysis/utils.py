import logging
import pickle


def load_pickle(filepath):
    """Load a Python object from a pickle file.

    Arguments:
        filepath: The path to load the object from.
    Raises:
        Exceptions raised by `open` and `pickle.dump`.
    """
    logging.info('Loading object from pickle: {}'.format(filepath))
    with open(filepath, 'rb') as infile:
        return pickle.load(infile)


def write_pickle(obj, filepath):
    """Write a Python object to a pickle file.

    Arguments:
        obj: A picklable Python object.
        filepath: The path to write the object to.
    Raises:
        Exceptions raised by `open` and `pickle.dump`.
    """
    logging.info('Writing pickle file to {}'.format(filepath))
    with open(filepath, 'wb') as outfile:
        pickle.dump(obj, outfile)
