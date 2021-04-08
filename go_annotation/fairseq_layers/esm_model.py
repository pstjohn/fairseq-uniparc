import torch
from fairseq.data import Dictionary
from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from fairseq.models.roberta import RobertaModel, utils
from torch import nn


@register_model("esm")
class ESMModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.args = args
        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--esm-architecture", type=str, help="ESM pretrained architecture"
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )

        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )

    @classmethod
    def build_model(cls, args, task):

        print(args.esm_architecture)
        model, alphabet = torch.hub.load("facebookresearch/esm", args.esm_architecture)  # fix
        dictionary = alphabet_to_dictionary(alphabet)
        encoder = ESMEncoder(model, dictionary)
        assert args.max_positions == model.args.max_positions

        return cls(args, encoder)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError


class ESMEncoder(FairseqEncoder):
    """ESM encoder."""

    def __init__(self, model, dictionary):
        super(ESMEncoder, self).__init__(dictionary)
        self.model = model

    def forward(
            self,
            src_tokens,
            features_only=False,
            return_all_hiddens=False,
            masked_tokens=None,
            **unused,
    ):
        assert features_only, "PSJ: lm head not currently implemented"

        result = self.model(src_tokens, repr_layers=[self.model.num_layers])
        x = result.pop('representations')[self.model.num_layers]
        return x, result


def alphabet_to_dictionary(alphabet):
    _dict = Dictionary(bos='<cls>', eos='<eos>')
    _dict.symbols = alphabet.all_toks
    _dict.indices = {token: i for token, i in enumerate(alphabet.all_toks)}
    _dict.count = [1] * len(alphabet.all_toks)

    _dict.bos_index = _dict.index('<cls>')
    _dict.pad_index = _dict.index('<pad>')
    _dict.eos_index = _dict.index('<eos>')
    _dict.unk_index = _dict.index('<unk>')

    return _dict


@register_model_architecture('esm', 'esm1_t6')
def esm1_t6(args):
    # These we'll need to change for any new architecture to match the pretrained shapes.
    args.esm_architecture = getattr(args, 'esm_architecture', 'esm1_t6_43M_UR50S')
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)

    # These are just for the classification head, I don't think we use any of these options
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )