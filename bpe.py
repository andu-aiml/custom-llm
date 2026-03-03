
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, Sequence
from tokenizers.processors import TemplateProcessing
from dataset import train_texts



# Initialize empty BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Normalization (optional but recommended)
tokenizer.normalizer = Sequence([NFD(), Lowercase()])

# Pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()

# Trainer
trainer = BpeTrainer(
    vocab_size=30000,          # You can change (20k–40k ideal)
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

# Train
tokenizer.train_from_iterator(train_texts, trainer=trainer)

tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[BOS] $A [EOS] $B:1 [EOS]:1",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ],
)

tokenizer.save("bpe_tokenizer.json")