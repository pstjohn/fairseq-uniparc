import logging
import os

import numpy as np
from fairseq.data import (
    ConcatSentencesDataset, IdDataset,
    NestedDictionaryDataset, NumSamplesDataset, NumelDataset,
    PrependTokenDataset,
    RawLabelDataset, RightPadDataset,
    RollDataset, SortDataset, data_utils)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask
from scipy.sparse import load_npz
from scipy.special import logit

from go_annotation.fairseq_layers.dataset import CSRLabelDataset

logger = logging.getLogger(__name__)


@register_task("sentence_labeling")
class SentenceLabelingTask(SentencePredictionTask):

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        label_dict = data_dict
        return cls(args, data_dict, label_dict)

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        head_name = getattr(args, "classification_head_name", "sentence_classification_head")

        update_bias = False
        if head_name == 'go_prediction' and head_name not in model.classification_heads:
            update_bias = True

        model.register_classification_head(
            head_name,
            num_classes=self.args.num_classes,
        )

        if update_bias:

            train_annotations = load_npz(os.path.join(self.args.data, 'label', 'train.npz'))
            epsilon = 1E-8
            initial_bias = logit((np.asarray(train_annotations.sum(0)).squeeze() /
                                  train_annotations.shape[0]) + epsilon)

            assert len(initial_bias) == self.args.num_classes, "Data / num_classes mismatch"
            bias = model.classification_heads[head_name].out_proj.bias.data
            model.classification_heads[head_name].out_proj.bias.data = bias.new(initial_bias)
            logger.info("Updated GO prediction head bias")

        return model

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(key, split):
            return os.path.join(self.args.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )
        input1 = make_dataset("input1", self.source_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions,
            self.args.seed,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset["net_input"].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        label_path = "{0}.npz".format(get_path("label", split))
        if os.path.exists(label_path):
            csr_matrix = load_npz(label_path)
            dataset.update(target=CSRLabelDataset(csr_matrix))

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]
