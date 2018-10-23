from bisect import bisect_right as threshold

import numpy as np

from deep_notebooks.data_util import bin_gradient_data
from deep_notebooks.text_util import get_nlg_phrase, deep_join, fix_sentence, generate_from_file


EXPLANATION_VECTOR_NLG = ['deep_notebooks/grammar', 'explanation_vector_description.nlg']

class NodeNotReadyException(Exception):
    pass


class ExplanationNode:

    def __init__(self):
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_text(self):
        text = [c.get_text() for c in self.children if
                c.get_text() is not None]
        if text:
            return deep_join(text, '\n')
        else:
            return ''


class IntroNode(ExplanationNode):

    def __init__(self, feature_name, feature_instance, class_name,
                 class_instance, type_):
        self.feature_name = feature_name
        self.feature_instance = feature_instance
        self.class_name = class_name
        self.class_instance = class_instance
        self.type = type_
        self.ready = False

    def compute_strength(self, gradients):
        self.ready = True
        direction = np.mean(gradients)
        strength = ['very weak', 'weak', 'moderate', 'strong', 'very strong']
        strength_values = [0.05, 0.15, 0.3, 0.5]
        direction_descriptor = ['negative', 'positive']
        self.strength = strength[threshold(strength_values, np.abs(direction))]
        self.direction = 'positive' if direction > 0 else 'negative'

    def get_text(self):
        if not self.ready:
            raise NodeNotReadyException
        if self.type == 'categorical':
            result = 'The instance "{}" of feature "{}" is a {} {} predictor \
                      for the instance "{}" of class "{}".'.format(
                self.feature_instance,
                self.feature_name,
                self.strength,
                self.direction,
                self.class_instance,
                self.class_name)
        else:
            result = 'The feature "{}" is a {} {} predictor for the instance \
                      "{}" of class "{}".'.format(
                self.feature_name,
                self.strength,
                self.direction,
                self.class_instance,
                self.class_name)
        return result


class BinNode(ExplanationNode):

    def __init__(self, data, percentage, descriptor, _bin):
        self.data = data
        self.percentage = percentage
        self.descriptor = descriptor[0]
        self.direction = descriptor[1]
        # magic, don't touch
        self.bin = (_bin - 3) * 0.25 - 0.125
        self.text = None
        self.ready = False
        self.useful = True

    def analyze(self):
        self.ready = True
        if self.percentage < 0.05 or len(
                self.data) < 3 or self.descriptor == 'very weak':
            self.useful = False
        else:
            self.mean = self.data.mean()
            self.var = self.data.var()


class BodyNode(ExplanationNode):

    def __init__(self, type_):
        ExplanationNode.__init__(self)
        self.type = type_
        self.text = None

    def analyze(self):
        for c in self.children:
            c.analyze()

    def parse(self, abstract_phrase, nodes):
        means = [n.mean for n in nodes]
        descriptors = [n.descriptor for n in nodes]
        directions = [n.direction for n in nodes]
        grammar = abstract_phrase.split(' ')
        sentence = []
        phrase_end = '.'
        node_counter = -1
        for i, comp in enumerate(grammar):
            if comp == '<CONJ>':
                continue
            else:
                node_counter += 1
            phrase = 'for data points centered around {}, the feature has a {} {} impact on the classification'.format(
                np.round(means[node_counter], 2),
                descriptors[node_counter],
                directions[node_counter])
            if i == len(grammar) - 1:
                sentence.append(phrase)
            else:
                if phrase_end == '.':
                    phrase = phrase.capitalize()
                if grammar[i + 1] == '<CONJ>':
                    phrase_end = ', '
                    conj = np.random.choice(CONJUNCTIONS, 1)[0]
                    phrase_end += conj
                else:
                    phrase_end = '.'
                phrase += phrase_end
                sentence.append(phrase)
        return ' '.join(sentence)

    def pure_description(self, direction, useful_nodes):
        descriptor = 'increases' if direction == 'positive' else 'decreases'
        if self.type == 'categorical':
            return 'Choosing another instance of this feature always \
                    {} the probability of this classification.'.format(
                descriptor)
        else:
            comp_main = 'Generally, a higher value for this feature \
                    will {} the class probability. \n\n'.format(
                descriptor)
            comp_body = fix_sentence(
                generate_from_file(*EXPLANATION_VECTOR_NLG))
            return ' '.join([comp_main, comp_body])

    def description(self, useful_nodes):
        if self.type == 'categorical':
            node_strengths = [c.bin for c in self.children]
            node_percentages = [c.percentage for c in self.children]
            direction = sum(
                [p * b for p, b in zip(node_percentages, node_strengths)])
            descriptor = 'increases' if direction > 1 else 'decreases'
            return 'In general, this value for the feature {} the probability of the prediction.'.format(
                descriptor)
        else:
            # comps = ['<PHRASE>' for n in useful_nodes]
            counter = 0
            phrases = []
            while counter < len(useful_nodes):
                phrase = get_nlg_phrase(*EXPLANATION_VECTOR_NLG)
                if 'and' in phrase or 'but' in phrase:
                    if counter == len(useful_nodes) - 1:
                        continue
                    node1 = useful_nodes[counter]
                    node2 = useful_nodes[counter + 1]
                    phrase = phrase.format(
                        strength=node1.descriptor,
                        strength_adv=node1.descriptor + 'ly',
                        strength_2=node2.descriptor,
                        strength_2_adv=node2.descriptor + 'ly',
                        mean=np.round(node1.mean, 2),
                        mean_2=np.round(node2.mean, 2),
                    )
                    counter += 2
                    phrases.append(phrase)
                else:
                    node1 = useful_nodes[counter]
                    phrase = phrase.format(
                        strength=node1.descriptor,
                        strength_adv=node1.descriptor + 'ly',
                        mean=np.round(node1.mean, 2),
                    )
                    counter += 1
                    phrases.append(phrase)
            return ' '.join(phrases)

    def get_text(self):
        useful_nodes = [c for c in self.children if c.useful]
        if len(useful_nodes) == 0:
            return ''

        pure = len(
            set([c.direction for c in self.children if len(c.data) > 0])) == 1
        if pure:
            return self.pure_description(
                list(set([c.direction for c in self.children]))[0],
                useful_nodes)
        else:
            return self.description(useful_nodes)


class ExplanationVectorDescription:

    def __init__(self):
        self.type = None
        self.feature_idx = None
        self.feature_name = None
        self.feature_instance = None
        self.class_idx = None
        self.class_name = None
        self.class_instance = None
        self.raw_data = None
        self.gradients = None

    def add_type(self, type_):
        self.type = type_

    def add_feature(self, name, idx, instance):
        self.feature_name = name
        self.feature_idx = idx
        self.feature_instance = instance

    def add_class(self, name, idx, instance):
        self.class_name = name
        self.class_idx = idx
        self.class_instance = instance

    def add_data(self, data):
        self.raw_data = data

    def add_gradients(self, gradients):
        self.gradients = gradients

    def compute_body(self):
        binned_data = bin_gradient_data(self.raw_data, self.gradients, 8)
        bin_counts = [b.shape[0] for b in binned_data]
        percentual = bin_counts / np.sum(bin_counts)

        description = BodyNode(self.type)
        descriptors = [
            ('very strong', 'negative'),
            ('strong', 'negative'),
            ('moderate', 'negative'),
            ('weak', 'negative'),
            ('weak', 'positive'),
            ('moderate', 'positive'),
            ('strong', 'positive'),
            ('very strong', 'positive')]
        for data, percent, descriptor, _bin in zip(binned_data, percentual,
                                                   descriptors, range(8)):
            node = BinNode(data[:, self.feature_idx], percent, descriptor,
                           _bin)
            description.add_child(node)

        return description

    def build_description(self):
        description = ExplanationNode()

        intro = IntroNode(self.feature_name,
                          self.feature_instance,
                          self.class_name,
                          self.class_instance,
                          self.type)
        intro.compute_strength(self.gradients)
        description.add_child(intro)

        body = self.compute_body()
        body.analyze()
        description.add_child(body)
        self.text = description.get_text()

    def get_text(self):
        return self.text
