from tokenizers import Tokenizer

def load_bpe_tokenizer(path="bpe_tokenizer.json"):
    tok = Tokenizer.from_file(path)

    pad_id = tok.token_to_id("[PAD]")
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    unk_id = tok.token_to_id("[UNK]")

    assert pad_id is not None and bos_id is not None and eos_id is not None and unk_id is not None
    return tok, pad_id, bos_id, eos_id, unk_id