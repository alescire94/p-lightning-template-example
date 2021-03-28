from typing import Dict, List


class Vocab:
    """
    This class allows you to build vocabulary to encode tokens, such as labels and words.
    The encoding is a function from str -> int and decoding is the inverse of it, where the int is an identifier.
    @param is_label: must be True for labels vocabulary in order to remove the <unk> entry in vocab.
    """
    def __init__(self, unk_token: str = "<unk>", pad_token: str = "<pad>", is_label: bool = False):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, List[str, int]] = {}  # id: (words, occ)
        self.unk_token: str = unk_token
        self.pad_token: str = pad_token
        self.add_token(self.pad_token)
        self._special_tokens = [pad_token]
        self.pad_id: int = self.token_to_id(pad_token)
        if not is_label:
            self.add_token(self.unk_token)
            self.unk_id = self.token_to_id(unk_token)
            self._special_tokens.append(self.unk_token)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __len__(self) -> int:
        return len(self._id_to_token)

    def add_token(self, token) -> int:
        if token not in self._token_to_id:
            id = len(self._token_to_id)
            self._token_to_id[token] = id
            self._id_to_token[id] = [token, 1]
        else:
            id = self._token_to_id[token]
            self._id_to_token[id][1] += 1
            # each time we obtain the same word in the text we increment the occurrence counter
        return id

    def id_to_token(self, id: int) -> str:
        return self._id_to_token[id][0]

    def token_to_id(self, token) -> int:
        return self._token_to_id[token]

    def drop_frequency(self, freq_to_drop: int) -> None:
        """
        This function take a integer and drop all entries in the vocabulary below a occurrence threshold.
        Allowing the model to train with OOVs.
        """
        self._id_to_token = {
            k: v for k, v in self._id_to_token.items() if v[1] > freq_to_drop or v in self._special_tokens
        }
