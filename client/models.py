from transformers import AutoModelForSequenceClassification

def load_model(model_name="bert-base-uncased", num_labels=42):
    """Load a pre-trained Transformer model for sequence classification."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
