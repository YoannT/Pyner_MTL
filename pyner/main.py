from modules import NER, Vocabulary
from _datasets import BRATDataset
import string
import torch
import pytorch_lightning as pl
from rich_logger import RichTableLogger

bert_name = "camembert/camembert-base"
model = NER(
    seed=42,
    preprocessor=dict(
        module="preprocessor",
        bert_name=bert_name, # transformer name
        sentence_split_regex=r"((?:\s*\n)+\s*|(?:(?<=[a-z0-9)]\.)\s+))(?=[A-Z-])", # regex to use to split sentences (must not contain consuming patterns)
        sentence_balance_chars=('()',), # try to avoid splitting between parentheses
        sentence_entity_overlap="raise", # raise when an entity spans more than one sentence, or use "split" to split entities in 2 when this happens
        word_regex='[\\w\']+|[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
        substitutions=( # Apply these regex substitutions on sentences before tokenizing
            (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "), # insert a space before punctuations
            (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "), # insert a space after punctuations
            #("(?<=[a-zA-Z])(?=[0-9])", r" "), # insert a space between letters and numbers
            #("(?<=[0-9])(?=[A-Za-z])", r" "), # insert a space between numbers and letters
        ),
        max_tokens=512,         # Maximum number of tokens in a sentence (will split if more than this number)
                                # Must be equal to or lower than the max number of tokens in the Bert model
        large_sentences="equal-split", # for these large sentences, split them in equal sub sentences < max_tokens tokens 
        empty_entities="raise", # when an entity cannot be mapped to any word, "raise" or "drop"
        vocabularies=torch.nn.ModuleDict({ # vocabularies to use, call .train() before initializing to fill/complete them automatically from training data
            "char": Vocabulary(string.punctuation + string.ascii_letters + string.digits, with_unk=True, with_pad=True),
            "label": Vocabulary(with_unk=False, with_pad=False),
        }).train(),
    ),

    # Word encore parameters
    word_encoders=[
        dict(
            module="char_cnn",
            n_chars=None, # automatically inferred from data
            in_channels=8,
            out_channels=50,
            kernel_sizes=(3, 4, 5),
        ),
        dict(
            module="bert",
            path=bert_name,
            n_layers=4,
            freeze_n_layers=0, # unfreeze all
            dropout_p=0.1,
        )
    ],
    
    # Decoder parameters
    decoder=dict(
        module="exhaustive_biaffine_ner",
        dim=192,
        label_dim=64,
        n_labels=None, # automatically inferred from data
        dropout_p=0.,
        use_batch_norm=False,
        contextualizer=dict(
            module="lstm",
            # use gate = False for better performance but slower convergence (needs ~50 epochs)
            gate=dict(
                module="sigmoid_gate",
                ln_mode="pre",
                init_value=0,
                proj=False,
                dim=192,
            ),
            input_size=768 + 150,
            hidden_size=192,
            num_layers=4,
            dropout_p=0.,
        )
    ),

    # Initialize last classifying layer bias with log frequencies from labels in data
    init_labels_bias=True,

    batch_size=24,
    
    # Use learning rate schedules (linearly decay with warmup)
    use_lr_schedules=True,
    warmup_rate=0.1,

    gradient_clip_val=5.,
    
    # Learning rates
    main_lr=1.5e-3,
    top_lr=1.5e-3,
    bert_lr=4e-5,
    
    # Optimizer, can be class or str
    optimizer_cls="transformers.AdamW",
).train()

flt_format = (5, "{:.4f}".format)
trainer = pl.Trainer(
    gpus=1,
    progress_bar_refresh_rate=False,
    move_metrics_to_cpu=True,
    logger=[
        #        pl.loggers.TestTubeLogger("path/to/logs", name="my_experiment"),
        RichTableLogger(key="epoch", fields={
            "epoch": {},
            "step": {},
            "train_loss": {"goal": "lower_is_better", "format": "{:.4f}"},
            "train_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_f1"},
            "train_precision": {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_p"},
            "train_recall": {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_r"},

            "val_loss": {"goal": "lower_is_better", "format": "{:.4f}"},
            "val_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_f1"},
            "val_precision": {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_p"},
            "val_recall": {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_r"},

            "main_lr": {"format": "{:.2e}"},
            "top_lr": {"format": "{:.2e}"},
            "bert_lr": {"format": "{:.2e}"},
        }),
    ],
    max_epochs=10)
dataset = BRATDataset(
    train="/home/ytaille/data/resources/mantra/Mantra-GSC/English/EMEA_ec22-cui-best_man",
    test=None,#"path/to/brat/test",    # None for training only, test directory otherwise
    val=0.2,  # first 20% doc will be for validation
    seed=False,  # don't shuffle before splitting
)
trainer.fit(model, dataset)
model.save_pretrained("ner.pt")