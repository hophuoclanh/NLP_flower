from utils.data_preprocessing import load_data, preprocess_data, split_data
from utils.tokenization import get_tokenizer, tokenize_data
from utils.data_partitioning import partition_data

DATASET_PATH = "News_Category_Dataset_v3.json"

# Step 1: Load and preprocess data
df = load_data(DATASET_PATH)
df, label_encoder = preprocess_data(df)
train, val, test = split_data(df)

# Step 2: Tokenize data
tokenizer = get_tokenizer()
train_tokens = tokenize_data(train, tokenizer)
val_tokens = tokenize_data(val, tokenizer)
test_tokens = tokenize_data(test, tokenizer)

# Step 3: Partition data
client_data = partition_data(train_tokens, num_partitions=4)