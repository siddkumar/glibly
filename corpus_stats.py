import matplotlib.pyplot as plt

if __name__ == '__main__':

    counter = {}
    with open('adverbs.txt', 'r') as adverbs:
        for line in adverbs:
            word = line.split()[0]
            cleanword = word[1:-1]

            if cleanword in counter:
                counter[cleanword] = counter[cleanword] + 1
            else:
                counter[cleanword] = 1

    num_w = 0
    total = 0
    for w in sorted( counter.keys()):
        num_w += 1
        total += counter[w]
        # print w, counter[w]

    print num_w, total

    plt.plot(sorted(counter.values(), reverse=True))
    plt.show()