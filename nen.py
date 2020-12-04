import nengo
import nengo.spa as spa
from nengo.spa import Vocabulary
import numpy as np

dim = 64
rng = np.random.RandomState(0)
vocab = Vocabulary(dimensions=dim, rng=rng)

mysent = "Boys chase dogs.".upper()

def custom_parser(sent):
    # Assume S V O
    S, V, O = sent[:-1].split()
    return (S, V, O)

for w in custom_parser(mysent):
    w_up = w.upper()
    w_sp = vocab.parse(w_up.upper())
    exec("{} = w_sp".format(w_up))

# BOY = vocab.parse('BOY')
# DOG = vocab.parse('DOG')
# CHASE = vocab.parse('CHASE')
# HUG = vocab.parse('HUG')

AGENT = vocab.parse('AGENT')
VERB = vocab.parse('VERB')
THEME = vocab.parse('THEME')

def conv_expression(parsed):
    S, V, O = parsed
    return "p = VERB * {} + AGENT * {} + THEME * {}".format(V, S, O)

model = spa.SPA(label=mysent, vocabs=[vocab])
with model:
    model.p = spa.State(dimensions=dim, label='p')
    # model.t = spa.State(dimensions=dim, label='t')
    # model.z = spa.State(dimensions=dim, label='z')

    model.out_agent = spa.State(dimensions=dim, label='out_agent')
    model.out_verb = spa.State(dimensions=dim, label='out_verb')
    model.out_theme = spa.State(dimensions=dim, label='out_theme')

    actions = spa.Actions(
        # 'p = VERB * CHASE + AGENT * DOG + THEME * BOY',
        conv_expression(custom_parser(mysent)),
        'out_agent = p * ~AGENT',
        'out_verb = p * ~VERB',
        'out_theme = p * ~THEME',
    )

    model.cortical = spa.Cortical(actions)
