from __future__ import print_function
from collections import Counter
from contextlib import closing
import cPickle
import gzip
import math
import os
import re
import sys


def gzip_context(path, mode):
    return closing(gzip.open(path, mode))


def save_pickle(data, path):
    with gzip_context(path, 'wb') as o:
        cPickle.dump(data, o)


def load_pickle(path):
    with gzip_context(path, 'rb') as i:
        data = cPickle.load(i)
    return data


#def die(msg):
#    print('\nERROR: {}'.format(msg)
#    sys.exit()


def logit(x, y=1):
    return 1.0 / (1 + math.e ** (-1*y*x))


def get_files(path, pattern):
    """
    Recursively find all files rooted in <path> that match the regexp <pattern>
    """

    def file_match(path):
        return (re.match(pattern,
            os.path.basename(path)) != None) and os.path.isfile(path)

    lst = []

    # base case: path is just a file
    if file_match(path):
        L.append(path)
        return L

    # general case
    if not os.path.isdir(path):
        return L

    contents = os.listdir(path)
    for item in contents:
        item = path + item
        if file_match(path):
            L.append(item)
        elif os.path.isdir(path):
            L.extend(get_files(item + '/', pattern))

    return L

class SplittaCounter(Counter):

   def __getitem__(self, entry):
       return super(SplittaCounter, self).get(entry, 0.0)

   def sorted_keys(self):
       """
       returns a list of keys sorted by their values
       keys with the highest values will appear first
       """
       sortedItems = self.items()
       compare = lambda x,y: cmp(y[1], x[1])
       sortedItems.sort(cmp=compare)
       return [x[0] for x in sortedItems]

   def totalCount(self):
       """
       returns the sum of counts for all keys
       """
       return sum(self.values())

   def incrementAll(self, value=1):
       """
       increment all counts by value
       helpful for removing 0 probs
       """
       for key in self.keys():
           self[key] += value

   def display(self):
       """
       a nicer display than the built-in dict.__repr__
       """
       for key, value in self.items():
           s = str(key) + ': ' + str(value)
           print s

   def displaySorted(self, N=10):
       """
       display sorted by decreasing value
       """
       sortedKeys = self.sortedKeys()
       for key in sortedKeys[:N]:
           s = str(key) + ': ' + str(self[key])
           print s

def normalize(counter):
   """
   normalize a counter by dividing each value by the sum of all values
   """
   counter = Counter(counter)
   normalizedCounter = Counter()
   total = float(counter.totalCount())
   for key in counter.keys():
       value = counter[key]
       normalizedCounter[key] = value / total
   return normalizedCounter
