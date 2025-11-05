import re

def tokenize(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())
