from typing import List

# I've been using this where possible for uniformity
# as well as to avoid weird null strings that aren't
# handled by split() for whatever reason
def ctakes_tok(s: str) -> List[str]:
    return [token for token in s.split() if token]

