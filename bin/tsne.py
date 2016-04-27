#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from depnn_py.embeddings import Embeddings
from sklearn.manifold import TSNE
import sys
import logging

# Input

model_dir = sys.argv[1]

# Code

logging.basicConfig(filename="output/tsne.log", filemode="w", level=logging.INFO, format="%(message)s")

# https://github.com/oreillymedia/t-SNE-tutorial

def scatter(points, labels):
    points *= 100
    f = plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(points[:,0], points[:,1], lw=0, s=10)

    for x, y, label in zip(points[:,0], points[:,1], labels):
        logging.info(str(x) + " " + str(y) + " " + label)
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")

    ax.axis("off")
    ax.axis("tight")

    return f, ax

def inverse_map(embeddings):
    res = [None] * len(embeddings._map)

    for key, value in embeddings._map.iteritems():
        res[value] = key

    return res

try:
    logging.info("Initializing embeddings")

    cat_embeddings = Embeddings(model_dir + "/cat.emb", 50, 0, False)
    slot_embeddings = Embeddings(model_dir + "/slot.emb", 50, 0, False)
    dist_embeddings = Embeddings(model_dir + "/dist.emb", 50, 0, False)
    pos_embeddings = Embeddings(model_dir + "/pos.emb", 50, 0, False)

    logging.info("cat: " + str(len(cat_embeddings._map)))
    logging.info("slot: " + str(len(slot_embeddings._map)))
    logging.info("dist: " + str(len(dist_embeddings._map)))
    logging.info("pos: " + str(len(pos_embeddings._map)))

    logging.info("Computing cat TSNE")
    cat_proj = TSNE().fit_transform(cat_embeddings._embeddings)
    scatter(cat_proj, inverse_map(cat_embeddings))
    plt.savefig(model_dir + "/cat.png", dpi=120)

    logging.info("Computing slot TSNE")
    slot_proj = TSNE().fit_transform(slot_embeddings._embeddings)
    scatter(slot_proj, inverse_map(slot_embeddings))
    plt.savefig(model_dir + "/slot.png", dpi=120)

    logging.info("Computing dist TSNE")
    dist_proj = TSNE().fit_transform(dist_embeddings._embeddings)
    scatter(dist_proj, inverse_map(dist_embeddings))
    plt.savefig(model_dir + "/dist.png", dpi=120)

    logging.info("Computing pos TSNE")
    pos_proj = TSNE().fit_transform(pos_embeddings._embeddings)
    scatter(pos_proj, inverse_map(pos_embeddings))
    plt.savefig(model_dir + "/pos.png", dpi=120)

    logging.info("Done")
except Exception as e:
    logging.exception("Exception on main handler")
