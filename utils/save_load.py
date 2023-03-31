import pickle


def save(target, path):
    f = open(path, 'wb')
    pickle.dump(target, f)
    f.close()

def load(path):
    f = open(path, 'rb')
    target = pickle.load(f)
    f.close()
    return target
