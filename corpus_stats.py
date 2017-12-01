import matplotlib.pyplot as plt

def get_counter(filename, verbose=False):
    counter = {}
    with open(filename, 'r') as adverbs:
        for line in adverbs:
            word = line.split()[0]

            if word in counter:
                counter[word] = counter[word] + 1
            else:
                counter[word] = 1

    num_w = 0
    total = 0

    how_many = 0 
    for w in sorted(counter.keys()):
        num_w += 1
        total += counter[w]
        if counter[w] >= 100:
            how_many += 1
            if verbose:
                print w, counter[w]

    if verbose:
        print how_many, num_w, 

    return counter

    # plt.plot(sorted(counter.values(), reverse=True))
    # plt.show()

if __name__ == '__main__':
    get_counter('adverbs.txt', verbose=True)