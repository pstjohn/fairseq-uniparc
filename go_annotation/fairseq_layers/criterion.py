import math
import logging

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion

from go_annotation.ontology import Ontology
ont = Ontology()

logger = logging.getLogger(__name__)


@register_criterion("go_prediction")
class GOPredictionCriterion(SentencePredictionCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super(GOPredictionCriterion, self).__init__(task, classification_head_name, regression_target)

        self._ancestor_array = ont.ancestor_array()
        head_nodes = ont.get_head_node_indices()
        self._ont_indicies = {
            'bp': ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes[0]])),
            'mf': ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes[1]])),
            'cc': ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes[2]]))}

    def forward(self, model, sample, reduce=True):
        assert (
                hasattr(model, "classification_heads")
                and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )

        assert logits.shape[1] == ont.total_nodes, \
            f"classification head must match ontology nodes, {ont.total_nodes}"

        targets = model.get_targets(sample, [logits])
        sample_size = targets.shape[0]

        # Normalize logits by ontology logic, requiring that child nodes have a lower score than parents.
        # Fairly verbose, since I'm doing this without torch_scatter.
        bsz = logits.shape[0]
        index_tensor = logits.new_tensor(self._ancestor_array, dtype=torch.int64)
        index_tensor = index_tensor.unsqueeze(0).expand((bsz, -1, -1))  # Array of ancestors, offset by one
        padded_logits = torch.nn.functional.pad(logits, (1, 0), value=float('inf'))  # Make 0 index return inf
        padded_logits = padded_logits.unsqueeze(-1).expand((-1, -1, index_tensor.shape[2]))
        normed_logits = torch.gather(padded_logits, 1, index_tensor)
        normed_logits, _ = torch.min(normed_logits, -1)

#         normed_logits = logits
        loss = F.binary_cross_entropy_with_logits(normed_logits, targets, reduction="sum")

        logging_output = {
            "loss": loss.data,
            "nsentences": sample_size,
            "sample_size": targets.numel(),
        }


        def convert_and_resize(x):
            return targets.new_tensor(x, dtype=torch.int64).unsqueeze(0).expand((logits.shape[0], -1))
        
        with torch.no_grad():
            y_pred = normed_logits > 0
            y_true = targets > 0.5
            
#             logging_output['tp'] = (y_true & y_pred).sum()
#             logging_output['tn'] = (~y_true & ~y_pred).sum()
#             logging_output['fp'] = (~y_true & y_pred).sum()
#             logging_output['fn'] = (y_true & ~y_pred).sum()
            
            for ont_split in ['bp', 'mf', 'cc']:
                y_pred_split = torch.gather(y_pred, -1, convert_and_resize(self._ont_indicies[ont_split]))
                y_true_split = torch.gather(y_true, -1, convert_and_resize(self._ont_indicies[ont_split]))

                logging_output[f'{ont_split}_tp'] = (y_true_split & y_pred_split).sum()
                logging_output[f'{ont_split}_tn'] = (~y_true_split & ~y_pred_split).sum()
                logging_output[f'{ont_split}_fp'] = (~y_true_split & y_pred_split).sum()
                logging_output[f'{ont_split}_fn'] = (y_true_split & ~y_pred_split).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, priority=5, round=3
        )
                                                                                
        metrics.log_scalar(
            "size", nsentences, sample_size, round=3
        )                                                                                
        
#         tp = sum(log.get("tp", 0) for log in logging_outputs)
#         tn = sum(log.get("tn", 0) for log in logging_outputs)
#         fp = sum(log.get("fp", 0) for log in logging_outputs)
#         fn = sum(log.get("fn", 0) for log in logging_outputs)
        
#         metrics.log_scalar("tp", tp, 0, round=3)        
#         metrics.log_scalar("tn", tn, 0, round=3)        
#         metrics.log_scalar("fp", fp, 0, round=3)        
#         metrics.log_scalar("fn", fn, 0, round=3)        
        
        
        for ont_split in ['bp', 'mf', 'cc']:
            tp = sum(log.get(f"{ont_split}_tp", 0) for log in logging_outputs).float()
            # tn = sum(log.get(f"{ont_split}_tn", 0) for log in logging_outputs)
            fp = sum(log.get(f"{ont_split}_fp", 0) for log in logging_outputs).float()
            fn = sum(log.get(f"{ont_split}_fn", 0) for log in logging_outputs).float()

            epsilon = 1e-7

            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)

            metrics.log_scalar(
                f"{ont_split}_f1", f1, 0, round=3
            )
                                                                                
#             metrics.log_scalar(
#                 f"{ont_split}_tp", tp, sample_size, round=3
#             )
#             metrics.log_scalar(
#                 f"{ont_split}_fp", fp, sample_size, round=3
#             )                                    