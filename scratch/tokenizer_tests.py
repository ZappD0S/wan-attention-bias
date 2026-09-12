# %%
import warnings

from wan import WanI2V
from wan.configs.wan_i2v_14B import i2v_14B

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# %%
wan_i2v = WanI2V(
    config=i2v_14B,
    checkpoint_dir="../weights/Wan2.1-I2V-14B-480P/",
    device_id=0,
    t5_cpu=True,
)

# %%
text_encoder = wan_i2v.text_encoder
hf_tokenizer = text_encoder.tokenizer
t5_tokenizer = hf_tokenizer.tokenizer

# %%
# print(len(t5_tokenizer.get_vocab()))

# %%
# print(t5_tokenizer.get_added_vocab())


# %%
def find_subsequence(seq, subseq):
    seq_len = len(seq)
    subseq_len = len(subseq)

    if subseq_len > seq_len:
        return -1

    for i in range(seq_len - subseq_len + 1):
        window = seq[i : i + subseq_len]
        if window == subseq:
            return i

    return -1


# %%
a = "New York"
b = "a city"
text = f"{a} is {b}"


text_token_ids = hf_tokenizer(text)[0].tolist()
a_token_ids = t5_tokenizer.encode(a, add_special_tokens=False)
b_token_ids = t5_tokenizer.encode(b, add_special_tokens=False)

start_index_a = find_subsequence(text_token_ids, a_token_ids)
start_index_b = find_subsequence(text_token_ids, b_token_ids)

# %%
text = "New York is a city"

encoding = t5_tokenizer(
    text,
    return_tensors="pt",
    # padding="max_length",
    # truncation=True,
    # max_length=hf_tokenizer.seq_len,
    # return_offsets_mapping=True,
    add_special_tokens=False,
)
print(encoding)
encoding = t5_tokenizer(
    text,
    return_tensors="pt",
    # padding="max_length",
    # truncation=True,
    # max_length=hf_tokenizer.seq_len,
    # return_offsets_mapping=True,
    # add_special_tokens=False,
)
list(encoding.keys())
# t5_tokenizer.encode(
#     text,
#     padding="max_length",
#     truncation=True,
#     max_length=hf_tokenizer.seq_len,
#     add_special_tokens=True
# )

# In this way we can
bare_ids = t5_tokenizer.encode(
    text,
    padding=False,
    add_special_tokens=False,
)
eos = t5_tokenizer.special_tokens_map.get("eos_token")
eos_id = t5_tokenizer.convert_tokens_to_ids(eos)

unpadded_ids = bare_ids + [eos_id]


padded_ids = t5_tokenizer.pad(
    [{"input_ids": unpadded_ids}],
    padding="max_length",
    max_length=hf_tokenizer.seq_len,
    return_tensors="pt",
)
