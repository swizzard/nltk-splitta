# Natural Language Toolkit: Splitta sentence tokenizer
#
# Copyright (C) 2001-2015 NLTK Project
# Algorithm: Gillick (2009)
# Author: Dan Gillick <dgillick@gmail.com> (original Python implementation)
#         Sam Raker <sam.raker@gmail.com> (NLTK-compatible version)
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

from collections import Counter
import cPickle
import math
import os
import re


##############################################################################
# TokenPair Class                                                            #
##############################################################################
class TokenPair(object):
    """
    Stores a pair of tokens and the label associated with them. Note that
    what is labeled is actually the break between the tokens, e.g.
    
    >>> tp = TokenPair('end.', 'The', 'boundary')
    # 'end.' ends a sentence, 'The' starts another one
    """

        # labels are stored as ints internally, but set and retrieved as strings
        label_list = ['non_candidate', 'unlabeled', 'boundary', 'non_boundary']
        label_dict = dict(zip(label_list, xrange(len(label_list))))

        __slots__ = ['tokens', '_label']

        def __init__(self, token1, token2, label):
            """Create a new token pair with the given label"""
            self.tokens = (token1, token2)
            self._label = self.validate_label(label)

        def __str__(self):
            if self.tokens[1] is None:
                return self.tokens[0]
            else:
                return " ".join(self.tokens)

        @property
        def label(self):
            return TokenPair.label_list(self._label)

        def validate_label(self, label_name):
            label_name = label_name or 'unlabeled'
            return TokenPair.label_dict[label_name]

        @label.setter
        def label(self, label_name):
            self._label = self.validate_label(label_name)

        @label.deleter
        def label(self):
            self._label = TokenPair.LABELS['unlabeled']


###############################################################################
# PairIter classes                                                            #
###############################################################################

class PairIterBase(object):
    """
    Base class for PairIters.
    
    PairIters are instantiated with a tokenizer,
    which is any class that inherits from ```nltk.tokenizer.api.TokenizerI```
    or at least implements a method called ```tokenize``` that returns a
    sequence of strings.
    
    Each PairIter child class has an attribute ```preprocessors```, which
    is a list of (`pat`, `replacement`) tuples, where `pat` is a (raw) string
    regular expression, and `replacement` is a replacement string.
    
    ```PairIterBase``` isn't meant to be directly instantiated, but rather
    inherited from.
    """

    default_preprocessors = [(r'[.,\d]*\d', '<NUM>'),
                             (r'--', ' '),
                             (r'[^a-zA-Z0-9,.;:\-\'\/?!$% ]', '')]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def preprocess_token(cls, token):
        """
        Preprocess a token by replacing each instance of the pattern
        with the corresponding replacement string, in order. If the result
        is not an empty string, it is returned, otherwise None is returned
        """
        
        for pat, replacement in cls.preprocessors:
            token = re.sub(pat, replacement, token)
        if token.replace('.', '').isalpha():
            return token
        return None

    def get_tokens(self, text):
        """
        Tokenize ```text``` using ```self.tokenizer.tokenize```, then
        yield non-None preprocessed tokens
        """
        for token in self.tokenizer.tokenize(text):
            preprocessed = self.preprocess_token(token)
            if preprocessed is not None:
                yield preprocessed
        
    def pair_iter(self, text):
        """
        Yield labeled TokenPairs created from ```text```
        """
        raise NotImplementedError


class RawPairIter(PairIterBase):
    """
    A PairIter that returns 'raw' (i.e. unprocessed) TokenPairs

    >>> s = "Hey there. You're cool."
    >>> tps = [TokenPair('Hey', 'there.', 'unlabeled'),
    ...     TokenPair('there.', 'You\'re', 'unlabeled'),
    ...     TokenPair('You\'re', 'cool.', 'unlabeled'),
    ...     TokenPair('cool.', None, 'non_boundary')]
    >>> pair_iter = RawPairIter(nltk.tokenizer.regexp.WhiteSpaceTokenizer())
    >>> tps == list(pair_iter.pair_iter(s)) # True
    """

    # no preprocessors--tokens returned 'as-is'
    preprocessors = []

    def __init__(self, tokenizer):
        super(RawPairIter, self).__init__(tokenizer)

    def pair_iter(self, text):
        """
        Yield unprocessed TokenPair objects. Note that all TokenPairs
        are labeled 'unlabeled', except the final pair, which is always
        labeled 'non_candidate'.
        """
        tokens = self.get_tokens(text)
        prev = tokens.next()
        for token in self._iter:
            yield TokenPair(prev, token, TokenPair.LABELS['unlabeled'])
            prev = token
        # final pair is always labeled as a non-candidate
        yield TokenPair(prev, token, TokenPair.LABELS['non_candidate'])


class TrainingPairIter(PairIterBase):
    """
    A PairIter that returns tokens from a training text. Training texts
    should be labeled with `<SB>` between sentences. Tokens on either side
    of a sentence boundary annotation will be joined together to create a
    TokenPair object with a 'boundary' label.

    >>> s = "Hey there. <SB> You're cool."
    >>> tps = [TokenPair('Hey', 'there.', 'non_candidate'),
    ...     TokenPair('there.', 'You\'re', 'boundary'),
    ...     TokenPair('You\'re', 'cool.', 'non_candidate'),
    ...     TokenPair('cool.', None, 'non_candidate')]
    >>> pair_iter = TrainingPairIter(nltk.tokenizer.regexp.WhiteSpaceTokenizer())
    >>> tps = list(pair_iter.pair_iter(s)) # True
    """

    # default preprocessors, plus strip out all angle brackets that aren't
    # part of annotations
    preprocessors = PairIter.default_preprocessors + \
            [(r'(<(?!SB)|(?<!SB)>)', '')]

    def __init__(self, tokenizer):
        super(TrainingPairIter, self).__init__(tokenizer)

    def pair_iter(self, text):
        prev = self._iter.next()
        for token in self._iter():
            yield self.process_pair(prev, token)
            prev = token
        yield TokenPair(prev, None, TokenPair.LABELS['non_candidate'])

    def process_pair(self, prev, token):
        if prev == '<SB>':
            return self.process_pair(token, self._iter.next())
        elif token == '<SB>':
            while token == '<SB>':
                token = self._iter.next()
            return TokenPair(prev, token, TokenPair.LABELS['boundary'])
        elif prev.endswith('.'):
            return TokenPair(prev, token, TokenPair.LABELS['non_boundary'])
        else:
            return TokenPair(prev, token, TokenPair.LABELS['non_candidate'])


class TokenizingPairIter(PairIterBase):
    """
    A PairIter that returns TokenPairs suitable for tokenizing text
    using a trained classifier. TokenPairs whose first token ends in a
    period are labeled 'unlabeled'; other TokenPairs are labeled
    'non_candidate'.

    >>> s = "Hey there. You're cool."
    >>> tps = [TokenPair('Hey', 'there.', 'non_candidate'),
    ...     TokenPair('there.', 'You\'re', 'unlabeled'),
    ...     TokenPair('You\'re', 'cool.', 'non_candidate'),
    ...     TokenPair('cool.', None, 'non_candidate')]
    >>> pair_iter = TokenizingPairIter(nltk.tokenize.regexp.WhiteSpaceTokenizer())
    >>> tps == list(pair_iter.pair_iter(s)) # True
    """

    # default preprocessors plus stripping out all angle brackets
    preprocessors = PairIter.default_preprocessors + [(r'[<>]', '')]

    def __init__(self, tokenizer):
        super(TestPairIter, self).__init__(tokenizer)

    def pair_iter(self, text):
        prev = self._iter.next()
        for token in self._iter():
            if prev.endswith('.'):
                yield TokenPair(prev, token, TokenPair.LABELS['unlabeled'])
            else:
                yield TokenPair(prev, token, TokenPair.LABELS['non_candidate'])
            prev = token
        yield TokenPair(prev, None, TokenPair.LABELS['non_candidate'])

###############################################################################
# FeatureExtractor class                                                      #
###############################################################################

class FeatureExtractor(object):
    """
    A class to extract features from TokenPairs in a format that can be passed
    to ```nltk.classifier.api.ClassifierI.train```. The features are:
        1) the first token
        2) the second token
        3) both words
        4) the length of the first word
        5) whether the second word is titlecased
        6) the log count of occurrences of the first word occurring without
           a final period
        7) the log count of lower-cased occurrences of the second word
        8) the first word and whether the second word is titlecased

    Features 6 and 7 are determined based on counts extracted from
    a text, which should ideally be the same text the classifier is
    trained on. These counts can independently be serialized to and
    retrieved from pickle files, or can be extracted from a training text.
    """

    def __init__(self,model_non_abbrs, model_lower_words):
        self.model_non_abbrs = model_non_abbrs
        self.model_lower_words = model_lower_words

    @property
    def has_counts(self):
        """
        Whether the FeatureExtractor's model counts have been
        set.
        """
        return all([count for count in (self.model_non_abbrs,
                                        self.model_lower_words)
                    if count is not None])

    def train(self, tokenized_text):
        """
        'Train' the FeatureExtractor by setting its model counts.
        :param tokenized_text: the tokenized version of the text to train
                               from
        :type tokenized_text: list of strings
        """
        non_abbrs = Counter()
        lower_words = Counter()
        for token in tokenized_text:
            if not token.endswith('.'):
                non_abbrs[token] += 1
            if token.islower();
                lower_words[token.replace('.', '')] += 1
        self.model_non_abbrs = non_abbrs
        self.model_lower_words = lower_words

    def save_model(self, dest_dir=None):
        """
        Save the FeatureExtractor's model counts to either the current
        directory or a destination directory
        """
        dest_dir = dest_dir or ''
        with open(os.path.join(dest_dir, 'non_abbrs'), 'wb') as out_file:
            cPickle.dump(self.model_non_abbrs, out_file)
        with open(os.path.join(dest_dir, 'lower_words'), 'wb') as out_file:
            cPickle.dump(self.model_lower_words, out_file)

    def load_model(self, non_abbrs_file, lower_words_file):
        """
        Load model counts
        """
        with open(non_abbrs_file, 'rb') as in_file:
            self.model_non_abbrs = cPickle.load(in_file)
        with open(lower_words_file, 'rb') as in_file:
            self.model_lower_words = cPickle.load(in_file)

    @staticmethod
    def save_features(self, features, dest):
        """
        Serialize features
        """
        with open(dest, 'wb') as out_file:
            cPickle.dump(features, out_file)

    def get_features(self, pair_iter):
        """
        Extract features from TokenPair objects and yield (features, label)
        tuples.
        """
        if not self.has_counts:
            raise AttributeError("Can't get features without counts")
        for token_pair in pair_iter:
            if token_label != 'non_candidate':
                feats = self.features_from_token_pair(token_pair)
                yield (feats, token_pair.label)

    def features_from_token_pair(self, token_pair):
        """
        Extract features from a TokenPair
        """
        word1 = token_pair.tokens[0]
        word2 = token_pair.tokens[1]
        features = {'word1': word1,
                    'word2': word2,
                    'word1_word2': token_pair.tokens
                    'word1_len': len(word1)
                    'word2_istitle': word2.istitle(),
                    'word1_abbr': self.get_log_count(self.model_non_abbrs,
                                                     word1.rstrip('.')),
                    'word2_lower': self.get_log_count(self.model_lower_words,
                                                      word2),
                    'word1_word2_istitle': (word1, word2.istitle())
                    }
        return features

    @staticmethod
    def get_log_count(model_count, key):
        """
        Get the normalized log of a count from one of our model counts
        """
        count = model_count.get(key)
        if count is None:
            return math.log(1)
        else:
            return math.log(1 + count)


###############################################################################
# Splitta classes                                                             #
###############################################################################

class SplittaBaseClass(object):
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor


class SplittaTokenizer(SpilttaBaseClass, TokenizerI):
    def __init__(self, tokenizer, feature_extractor, classifier):
        super(SplittaTokenizer, self).__init__(tokenizer, feature_extractor)
        self.raw_pair_iter = util.RawPairIter(tokenizer)
        self.features_pair_iter = util.TokenizingPairIter(tokenizer)
        self.classifier = classifier

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def tokenize_sents(self, text):
        sents = []
        raw_pairs = self.raw_pair_iter.pair_iter(text)
        token_pairs = self.features_pair_iter.pair_iter(text)
        features = self.feature_extractor.get_features(token_pairs)
        sent = []
        while True:
            try:
                pair = raw_pairs.next()
                feats = features.next()
            except StopIteration:
                break
            else:
                label = self.classifier.classify(feats[0])
                if label == 'boundary':
                    sent.append(pair[0])
                    sents.append(' '.join((str(token) for token in sent
                                           if token is not None)))
                    sent = [pair[1]]
                elif label in ('non_boundary', 'non_candidate'):
                    sent.extend(pair.tokens)
        return sents


class SplittaTrainer(SplittaBaseClass):
    def __init__(self, tokenizer, classifier, feature_extractor=None):
        feature_extractor = feature_extractor or FeatureExtractor(None, None)
        super(SplittaTrainer, self).__init__(tokenizer, feature_extractor)
        self.features_pair_iter = util.TrainingPairIter(tokenizer)
        self.classifier = classifier

    def train(self, training_text, **classifier_kwargs):
        if not self.feature_extractor.has_counts:
            self.feature_extractor.train(
                    self.tokenizer.tokenize(training_text))
        features = list(self.feature_extractor.get_features(
            self.features_pair_iter.pair_iter(training_text)))
        self.classifier.train(features, **classifier_kwargs)

