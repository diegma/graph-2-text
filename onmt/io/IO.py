# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, OrderedDict
from itertools import count
import torch.autograd as autograd

import torch
import torchtext.data
import torchtext.vocab

from onmt.io.DatasetBase import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.io.TextDataset import TextDataset
from onmt.io.ImageDataset import ImageDataset
from onmt.io.AudioDataset import AudioDataset
from onmt.io.GCNDataset import GCNDataset

import numpy as np

def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def get_fields(data_type, n_src_features, n_tgt_features):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    if data_type == 'text':
        return TextDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'img':
        return ImageDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'audio':
        return AudioDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'gcn':
        return GCNDataset.get_fields(n_src_features, n_tgt_features)


def load_fields_from_vocab(vocab, data_type="text"):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[UNK_WORD, PAD_WORD,
                                           BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def get_num_features(data_type, corpus_file, side):
    """
    Args:
        data_type (str): type of the source input.
            Options are [text|img|audio].
        corpus_file (str): file path to get the features.
        side (str): for source or for target.

    Returns:
        number of features on `side`.
    """
    assert side in ["src", "tgt"]

    if data_type == 'text':
        return TextDataset.get_num_features(corpus_file, side)
    elif data_type == 'img':
        return ImageDataset.get_num_features(corpus_file, side)
    elif data_type == 'audio':
        return AudioDataset.get_num_features(corpus_file, side)
    elif data_type == 'gcn':
        return GCNDataset.get_num_features(corpus_file, side)


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Variable): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src', 'tgt']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    elif data_type == 'gcn':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]

def get_morph(batch):

    #Not very nice but we do not have access to value comming from opt.gpuid command line parameter here.
    use_cuda = batch.src[0].is_cuda

    # morph_index = batch.morph.data.transpose(0, 1)  # [ seqLen x batch_size ] ==> [ batch_size x seqLen ]

    # morph_voc = batch.dataset.fields['morph'].vocab.stoi

    morph_index = batch.morph.view((batch.src[0].data.size()[0], 6, batch.src[0].data.size()[1]))
    morph_index = morph_index.permute(2, 0, 1).contiguous()



    # morph_index = torch.LongTensor(morph_index)
    morph_mask = torch.lt(torch.eq(morph_index, 1), 1).float()
    # morph_index = autograd.Variable(morph_index)
    # morph_mask = autograd.Variable(torch.FloatTensor(morph_mask), requires_grad=False)
    if use_cuda:
        morph_index = morph_index.cuda()
        morph_mask = morph_mask.cuda()

    return morph_index, morph_mask




def get_adj(batch):

    #Not very nice but we do not have access to value comming from opt.gpuid command line parameter here.
    use_cuda = batch.src[0].is_cuda

    node1_index = batch.node1.data.transpose(0, 1)  # [ seqLen x batch_size ] ==> [ batch_size x seqLen ]
    node2_index = batch.node2.data.transpose(0, 1)
    label_index = batch.label.data.transpose(0, 1)

    node1_voc = batch.dataset.fields['node1'].vocab.itos
    node2_voc = batch.dataset.fields['node2'].vocab.itos
    label_voc = batch.dataset.fields['label'].vocab.itos

    batch_size = batch.batch_size

    _MAX_BATCH_LEN = batch.src[0].data.size()[0]   # data is [ seqLen x batch_size ]

    _MAX_DEGREE = 10  # If the average degree is much higher than this, it must be changed.

    sent_mask = torch.lt(torch.eq(batch.src[0].data, 1), 1)

    adj_arc_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE, 2), dtype='int32')
    adj_lab_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE, 1), dtype='int32')
    adj_arc_out = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE, 2), dtype='int32')
    adj_lab_out = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE, 1), dtype='int32')


    mask_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE), dtype='float32')
    mask_out = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE), dtype='float32')
    mask_loop = np.ones((batch_size * _MAX_BATCH_LEN, 1), dtype='float32')

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(node1_index):  # iterates over the batch
        for a, arc in enumerate(de):

            arc_0 = label_voc[label_index[d, a]]

            if arc_0 == '<unk>' or arc_0 == '<pad>':
                pass
            else:

                arc_1 = int(node1_voc[arc])
                arc_2 = int(node2_voc[node2_index[d, a]])

                if arc_1 in tmp_in:
                    tmp_in[arc_1] += 1
                else:
                    tmp_in[arc_1] = 0

                if arc_2 in tmp_out:
                    tmp_out[arc_2] += 1
                else:
                    tmp_out[arc_2] = 0

                idx_in = (d * _MAX_BATCH_LEN * _MAX_DEGREE) + arc_1 * _MAX_DEGREE + tmp_in[arc_1]

                idx_out = (d * _MAX_BATCH_LEN * _MAX_DEGREE) + arc_2 * _MAX_DEGREE + tmp_out[arc_2]

                if tmp_in[arc_1] < _MAX_DEGREE:

                    adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                    adj_lab_in[idx_in] = np.array([label_index[d, a]])  # incoming arcs
                    mask_in[idx_in] = 1.

                if tmp_out[arc_2] < _MAX_DEGREE:

                    adj_arc_out[idx_out] = np.array([d, arc_1])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array([label_index[d, a]])  # outgoing arcs
                    mask_out[idx_out] = 1.

        tmp_in = {}
        tmp_out = {}

    adj_arc_in = autograd.Variable(torch.LongTensor(np.transpose(adj_arc_in).tolist()))
    adj_arc_out = autograd.Variable(torch.LongTensor(np.transpose(adj_arc_out).tolist()))

    adj_lab_in = autograd.Variable(torch.LongTensor(np.transpose(adj_lab_in).tolist()))
    adj_lab_out = autograd.Variable(torch.LongTensor(np.transpose(adj_lab_out).tolist()))

    mask_in = autograd.Variable(torch.FloatTensor(mask_in.reshape((_MAX_BATCH_LEN * node1_index.size()[0], _MAX_DEGREE)).tolist()), requires_grad=False)
    mask_out = autograd.Variable(torch.FloatTensor(mask_out.reshape((_MAX_BATCH_LEN * node1_index.size()[0], _MAX_DEGREE)).tolist()), requires_grad=False)
    mask_loop = autograd.Variable(torch.FloatTensor(mask_loop.tolist()), requires_grad=False)
    sent_mask = autograd.Variable(torch.FloatTensor(sent_mask.tolist()), requires_grad=False)
    if use_cuda:
        adj_arc_in = adj_arc_in.cuda()
        adj_arc_out = adj_arc_out.cuda()
        adj_lab_in = adj_lab_in.cuda()
        adj_lab_out = adj_lab_out.cuda()
        mask_in = mask_in.cuda()
        mask_out = mask_out.cuda()
        mask_loop = mask_loop.cuda()
        sent_mask = sent_mask.cuda()
    return adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop, sent_mask


def collect_features(fields, side="src"):
    """
    Collect features from Field object.
    """
    assert side in ["src", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def collect_feature_vocabs(fields, side):
    """
    Collect feature Vocab objects from Field object.
    """
    assert side in ['src', 'tgt']
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs



def build_dataset(fields, data_type, src_path, tgt_path, src_dir=None,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True):

    # Build src/tgt examples iterator from corpus files, also extract
    # number of features.
    src_examples_iter, num_src_feats = \
        _make_examples_nfeats_tpl(data_type, src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio)

    # For all data types, the tgt side corpus is in form of text.
    tgt_examples_iter, num_tgt_feats = \
        TextDataset.make_text_examples_nfeats_tpl(
            tgt_path, tgt_seq_length_trunc, "tgt")

    if data_type == 'text':
        dataset = TextDataset(fields, src_examples_iter, tgt_examples_iter,
                              num_src_feats, num_tgt_feats,
                              src_seq_length=src_seq_length,
                              tgt_seq_length=tgt_seq_length,
                              dynamic_dict=dynamic_dict,
                              use_filter_pred=use_filter_pred)

    elif data_type == 'img':
        dataset = ImageDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, num_tgt_feats,
                               tgt_seq_length=tgt_seq_length,
                               use_filter_pred=use_filter_pred)

    elif data_type == 'audio':
        dataset = AudioDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, num_tgt_feats,
                               tgt_seq_length=tgt_seq_length,
                               sample_rate=sample_rate,
                               window_size=window_size,
                               window_stride=window_stride,
                               window=window,
                               normalize_audio=normalize_audio,
                               use_filter_pred=use_filter_pred)

    return dataset


def build_dataset_gcn(fields, data_type, src_path, tgt_path,
                  label_path, node1_path, node2_path, morph_path, src_dir=None,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True):

    # Build src/tgt examples iterator from corpus files, also extract
    # number of features.
    src_examples_iter, num_src_feats = \
            _make_examples_nfeats_tpl(data_type+"_src", src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio)

    # For all data types, the tgt side corpus is in form of text.
    tgt_examples_iter, num_tgt_feats = \
    TextDataset.make_text_examples_nfeats_tpl(
            tgt_path, tgt_seq_length_trunc, "tgt")

    label_examples_iter, num_label_feats =  \
            _make_examples_nfeats_tpl(data_type+"_label", label_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio)

    node1_examples_iter, num_node1_feats = \
            _make_examples_nfeats_tpl(data_type+"_node1", node1_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio)

    node2_examples_iter, num_node2_feats = \
            _make_examples_nfeats_tpl(data_type+"_node2", node2_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio)
    morph_examples_iter = ''
    if morph_path !='':
        morph_examples_iter, num_morph_feats = \
            _make_examples_nfeats_tpl(data_type + "_morph", morph_path, src_dir,
                                      src_seq_length_trunc, sample_rate,
                                      window_size, window_stride,
                                      window, normalize_audio)

    dataset = GCNDataset(
                fields, src_examples_iter, tgt_examples_iter,
                label_examples_iter, node1_examples_iter,
                node2_examples_iter, morph_examples_iter,
                num_src_feats, num_tgt_feats,
                src_seq_length=src_seq_length,
                tgt_seq_length=tgt_seq_length,
                dynamic_dict=dynamic_dict,
                use_filter_pred=use_filter_pred
                )

    return dataset


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in
        [field.unk_token, field.pad_token, field.init_token,field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}
    for k in fields:
        counter[k] = Counter()

    for path in train_dataset_files:
        dataset = torch.load(path)
        print(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                if k == 'morph' and hasattr(ex, 'morph'):
                    for m in ex.morph:
                        val =[(m)]
                        if m is not None and not fields[k].sequential:
                            val = [(m)]
                        counter[k].update(val)
                        # for m_r in m.split('_'):
                        #     # val = getattr(m_r, k, None)
                        #     val =[(m_r)]
                        #     if m_r is not None and not fields[k].sequential:
                        #         val = [(m_r)]
                        #     counter[k].update(val)
                else:
                    val = getattr(ex, k, None)
                    if val is not None and not fields[k].sequential:
                        val = [val]
                    counter[k].update(val)

    _build_field_vocab(fields["tgt"], counter["tgt"],
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)
    print(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    # All datasets have same num of n_tgt_features,
    # getting the last one is OK.
    for j in range(dataset.n_tgt_feats):
        key = "tgt_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key])
        print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

    if data_type == 'text':
        _build_field_vocab(fields["src"], counter["src"],
                           max_size=src_vocab_size,
                           min_freq=src_words_min_frequency)
        print(" * src vocab size: %d." % len(fields["src"].vocab))

        # All datasets have same num of n_src_features,
        # getting the last one is OK.
        for j in range(dataset.n_src_feats):
            key = "src_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

        # Merge the input and output vocabularies.
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            print(" * merging src and tgt vocab...")
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab

    elif data_type == 'gcn':
        _build_field_vocab(fields["src"], counter["src"],
                           max_size=src_vocab_size,
                           min_freq=src_words_min_frequency)
        print(" * src vocab size: %d." % len(fields["src"].vocab))

        # All datasets have same num of n_src_features,
        # getting the last one is OK.
        for j in range(dataset.n_src_feats):
            key = "src_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

        counter["node1"].update([str(x) for x in range(200)])
        counter["node2"].update([str(x) for x in range(200)])
        _build_field_vocab(fields["node1"], counter["node1"],
                           max_size=src_vocab_size,
                           min_freq=0)
        print(" * node1 vocab size: %d." % len(fields["node1"].vocab))

        _build_field_vocab(fields["node2"], counter["node2"],
                           max_size=src_vocab_size,
                           min_freq=0)
        print(" * node2 vocab size: %d." % len(fields["node2"].vocab))

        _build_field_vocab(fields["label"], counter["label"],
                           max_size=src_vocab_size,
                           min_freq=0)
        print(" * label vocab size: %d." % len(fields["label"].vocab))

        _build_field_vocab(fields["morph"], counter["morph"],
                           max_size=src_vocab_size,
                           min_freq=0)
        print(" * morph vocab size: %d." % len(fields["morph"].vocab))



        # Merge the input and output vocabularies.
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            print(" * merging src and tgt vocab...")
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab

    return fields


def _make_examples_nfeats_tpl(data_type, src_path, src_dir,
                              src_seq_length_trunc, sample_rate,
                              window_size, window_stride,
                              window, normalize_audio):
    """
    Process the corpus into (example_dict iterator, num_feats) tuple
    on source side for different 'data_type'.
    """

    if data_type == 'text':
        src_examples_iter, num_src_feats = \
            TextDataset.make_text_examples_nfeats_tpl(
                src_path, src_seq_length_trunc, "src")

    elif data_type == 'img':
        src_examples_iter, num_src_feats = \
            ImageDataset.make_image_examples_nfeats_tpl(
                src_path, src_dir)

    elif data_type == 'audio':
        src_examples_iter, num_src_feats = \
            AudioDataset.make_audio_examples_nfeats_tpl(
                src_path, src_dir, sample_rate,
                window_size, window_stride, window,
                normalize_audio)

    elif data_type.startswith('gcn_'):
        src_examples_iter, num_src_feats = \
            GCNDataset.make_text_examples_nfeats_tpl(
                src_path, src_seq_length_trunc, data_type.split("_")[1])

    return src_examples_iter, num_src_feats


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            # print(self.data().src)
            # print(self.data().node1)

            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                # print(b[0].src)
                # print(b[0].node1)

                self.batches.append(sorted(b, key=self.sort_key))
