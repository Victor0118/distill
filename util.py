import os
import argparse

from easydict import EasyDict as edict
import argconf
import jieba
import torch
import torch.nn as nn


def unwrap(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return model

def save_checkpoint(model, workspace, best=False, **save_kwargs):
    model = unwrap(model)
    save_point = dict(state_dict=model.state_dict())
    save_point.update(save_kwargs)
    prefix = "best" if best else "last"
    torch.save(save_point, os.path.join(workspace, f"{prefix}_model.pt"))

def make_checkpoint_incrementer(model, workspace, best_loss=10000, save_last=False):
    best_loss = [best_loss]
    def increment(dev_loss, **save_kwargs):
        is_best = False
        save_kwargs["best_dev_loss"] = best_loss[0]
        if dev_loss < best_loss[0]:
            save_kwargs["dev_loss"] = save_kwargs["best_dev_loss"] = best_loss[0] = dev_loss
            save_checkpoint(model, workspace, best=True, **save_kwargs)
            is_best = True
        if save_last:
            save_checkpoint(model, workspace, best=False, **save_kwargs)
        return is_best
    return increment

def uniform_unk_init(a=-0.25, b=0.25):
    return lambda tensor: tensor.uniform_(a, b)

def tokenize_sent(sent):
    words = jieba.cut(sent, cut_all=False)
    words = [str(word) for word in words]
    return words

def read_args(default_config="confs/base.json", **parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--config", "-c", type=str, default=default_config)
    args, _ = parser.parse_known_args()
    options = argconf.options_from_json("options.json")
    config = argconf.config_from_json(args.config)
    return edict(argconf.parse_args(options, config))

def init_embedding(model, config):
    if config.mode == "rand":
        rand_embed_init = torch.Tensor(config.words_num, config.words_dim).uniform_(-0.25, 0.25)
        model.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        model.embedding_dim = config.words_dim
    elif config.mode == "static":
        model.static_embed = nn.Embedding.from_pretrained(config.dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        model.embedding_dim = config.words_dim
    elif config.mode == "non-static":
        model.non_static_embed = nn.Embedding.from_pretrained(config.dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        model.embedding_dim = config.words_dim
    elif config.mode == "multichannel":
        model.static_embed = nn.Embedding.from_pretrained(config.dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        model.non_static_embed = nn.Embedding.from_pretrained(config.dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        model.embedding_dim = config.words_dim * 2
    else:
        raise ValueError("Unsupported mode.")


def fetch_embedding(model, mode, x, squash=False):
    if mode == "rand":
        word_input = model.embed(x) # (batch, sent_len, embed_dim)
        x = word_input # (batch, channel_input, sent_len, embed_dim)
        if not squash:
            x = x.unsqueeze(1)
    elif mode == "static":
        static_input = model.static_embed(x)
        x = static_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        if not squash:
            x = x.unsqueeze(1)
    elif mode == "non-static":
        non_static_input = model.non_static_embed(x)
        x = non_static_input
        if not squash:
            x = x.unsqueeze(1)
    elif mode == "multichannel":
        non_static_input = model.non_static_embed(x)
        static_input = model.static_embed(x)
        if squash:
            x = torch.cat([non_static_input, static_input], dim=2)
        else:
            x = torch.stack([non_static_input, static_input], dim=1)
    else:
        raise ValueError("Unsupported mode.")
    return x
