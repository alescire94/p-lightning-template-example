from typing import Dict, List


class Vocab:
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

    def is_present(self, token: str) -> bool:
        return token in self._token_to_id

    def add_token(self, token) -> int:
        if token not in self._token_to_id:
            id = len(self._token_to_id)
            self._token_to_id[token] = id
            self._id_to_token[id] = [token, 1]
        else:
            id = self._token_to_id[token]
            self._id_to_token[id][1] += 1
        return id

    def id_to_token(self, id: int) -> str:
        return self._id_to_token[id][0]

    def token_to_id(self, token) -> int:
        return self._token_to_id[token]

    def drop_frequency(self, freq_to_drop: int):
        self._id_to_token = {
            k: v for k, v in self._id_to_token.items() if v[1] > freq_to_drop or v in self._special_tokens
        }
