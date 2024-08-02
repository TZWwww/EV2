import argparse

from transformers import MODEL_MAPPING, SchedulerType

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--openai_key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--generative_format",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--data_type",
        nargs='+',
        type=str,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=5000,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=64,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--dataset_augment",
        action="store_true",
    )
    parser.add_argument(
        "--loss_scale",
        action="store_true",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--load_path",
        type=str,
        help="Path to load model.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--prompter_learning_rate",
        type=float,
        default=1e-3,
        help="Initial prompter learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data file..",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether or not to use gpu.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="logging steps..",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='adafactor',
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--r_type_num",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--t_type_num",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--expand_trigger",
        action="store_true",
        help="Whether to expand trigger. ",
    )
    parser.add_argument(
        "--use_instruction",
        action="store_true",
        help="Whether to use instruction . ",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default='',
        help="description.",
    )
    parser.add_argument(
        "--test",
        # action="store_true",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--loss_weight",
        type=str,
        default='none',
        choices=['label_size', 'none', 'dataset_size'],
        help="loss weight.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default='none',
        choices=['prompt', 'prefix', 'contrastive', 'none'],
        help="prompt type.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank.",
    )
    parser.add_argument(
        "--prefix_drop",
        type=float,
        default=0.3,
        help="prefix drop.",
    )
    parser.add_argument(
        "--base_num",
        type=int,
        default=20,
        help="number of base.",
    )
    parser.add_argument(
        "--freeze_lm",
        action='store_true',
        help="Whether to freeze language model.",
    )
    parser.add_argument(
        "--use_event_symbol",
        action='store_true',
        help="Whether to use event symbol.",
    )
    parser.add_argument(
        "--prompt_attn_entropy",
        action='store_true',
        help="Whether to compute prompt attention entropy loss.",
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
        help="Whether to use fp16.",
    )
    parser.add_argument(
        "--freeze_lm_epochs",
        type=int,
        default=0,
        help="Freeze language model in first many epochs. ",
    )
    parser.add_argument(
        "--evaluate_per_epoch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--number_of_folds",
        type=int,
        nargs='+',
        default=1,
    )
    parser.add_argument(
        "--remove_event_in_doc",
        action='store_true',
        help="remove events in doc for unlabeled data. ",
    )
    parser.add_argument(
        "--remove_checkpoint",
        action='store_true',
    )
    parser.add_argument(
        "--minus_stm_prob",
        action="store_true",
    )
    parser.add_argument(
        "--C_num",
        type=int,
    )
    parser.add_argument(
        "--C_len",
        type=int,
    )
    parser.add_argument(
        "--resume_from_step",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--missing_steps",
        nargs='+',
        type=int,
    )
    parser.add_argument(
        "--use_cot",
        action='store_true',
    )
    parser.add_argument(
        "--split",
        action='store_true',
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cluster_dir",
        type=str,
        default=None,
    )


    args = parser.parse_args()

    # # Sanity checks
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    #
    # if args.push_to_hub:
    #     assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args
