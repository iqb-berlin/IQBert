from dataclasses import dataclass

@dataclass
class PreparedText:
    sentences: list[str]
    vocab_size: int
    token_list: list[list[int]]
    word_dict: dict[str, int]
    number_dict: dict[int, str]
