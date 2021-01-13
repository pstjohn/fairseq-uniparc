import logging

import numpy as np
from fairseq.tasks import register_task
from fairseq.tasks.masked_lm import MaskedLMTask

logger = logging.getLogger(__name__)


@register_task("masked_lm_bias")
class MaskedLMTaskBiasInit(MaskedLMTask):
    def build_model(self, args):
        model = super(MaskedLMTaskBiasInit, self).build_model(args)

        bias = model.encoder.lm_head.bias.data
        dictionary_counts = np.array(self.dictionary.count)
        assert len(bias) == len(dictionary_counts)
        bias_init = np.log(dictionary_counts / dictionary_counts.sum() + np.finfo(float).eps)
        model.encoder.lm_head.bias.data = bias.new(bias_init)

        return model
