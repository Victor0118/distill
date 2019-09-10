from torchtext.data import Field, TabularDataset, Iterator
from torchtext.vocab import Vectors
import torch

from util import tokenize_sent, uniform_unk_init


_REGISTRY = {}

class RegisteredDataset(TabularDataset):

    def __init_subclass__(cls, name):
        _REGISTRY[name.lower()] = cls


class Sim_ZH_Combine(RegisteredDataset, name="sim_zh_combine"):
    TEXT_FIELD = Field(batch_first=True, tokenize=tokenize_sent)
    LOGITS = Field(sequential=True, use_vocab=False, batch_first=True, dtype=torch.float, tokenize=lambda x: [float(i) for i in x.split()])

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence)

    @classmethod
    def splits(cls, folder_path, train="train.tsv", dev="dev.tsv", test="test.tsv"):
        fields = [("sentence", cls.TEXT_FIELD), ("logits", cls.LOGITS)]
        return super(Sim_ZH_Combine, cls).splits(folder_path, train=train, validation=dev, test=test, format="tsv", 
            fields=fields, skip_header=False)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, vectors=None,
              unk_init=uniform_unk_init(), device="cuda:0", train="train.tsv", dev="dev.tsv", test="test.tsv"):
        #if vectors is None:
        #    vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path, train=train, dev=dev, test=test)
        cls.TEXT_FIELD.build_vocab(train, val, test) # , vectors=vectors)
        return Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, 
            sort_within_batch=False, device=device, sort=False)
