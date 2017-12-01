import matplotlib.pyplot as plt

if __name__ == '__main__':
    counter = {}
    with open('adverbs.txt', 'r') as adverbs:
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
        if counter[w] >= 50:
            how_many += 1
            print w, counter[w]

    print how_many, num_w, total

    plt.plot(sorted(counter.values(), reverse=True))
    plt.show()
