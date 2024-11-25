from transformers import AutoTokenizer

def get_tokenizer(model_name="bert-base-uncased"):
    """Load a pre-trained tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_data(df, tokenizer, max_length=128):
    """Tokenize the dataset."""
    def tokenize_function(example):
        return tokenizer(
            example["text"], truncation=True, padding="max_length", max_length=max_length
        )

    tokenized = df.apply(tokenize_function, axis=1)
    return tokenized
