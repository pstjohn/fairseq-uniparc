from fairseq.tasks import register_task, SentencePredictionTask

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

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, "label", "dict.txt"),
                source=False,
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(args, data_dict, label_dict)