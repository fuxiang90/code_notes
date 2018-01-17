"""Count words."""

import heapq
class TopkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def Push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem[0] > topk_small[0]:
                heapq.heapreplace(self.data, elem)

    def TopK(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in xrange(len(self.data))])]

def cmpF(x,y):

    if x[0] != y[0]:
        return y[0] - x[0]
    else:
        return cmp(x[1],y[1])

def count_words(s, n):
    """Return the n most frequently occuring words in s."""

    # TODO: Count the number of occurences of each word in s

    # TODO: Sort the occurences in descending order (alphabetically in case of ties)

    # TODO: Return the top n most frequent words.
    topk = TopkHeap(n)
    word_cnt = {}
    for word in s.split():
        word_cnt.setdefault(word, 0)
        word_cnt[word] += 1
    nn = []
    for k,v in word_cnt.iteritems():
        elem = (v,k)
        nn.append(elem)

    top_n = [(x[1],x[0]) for x in sorted(nn,cmp=cmpF)[:n]]
    return top_n


def work():
    """Test count_words() with some inputs."""
    print count_words("cat bat mat cat bat cat", 3)
    print count_words("betty bought a bit of butter but the butter was bitter", 3)


if __name__ == '__main__':
    work()
