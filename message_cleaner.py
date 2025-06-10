import Stemmer
import regex as re

stemmer = Stemmer.Stemmer('lt')
stemmer_ru = Stemmer.Stemmer('ru')

def tokenize(text):
    text = text.lower()
    tokens = re.split(r"\W+", text)
    modified_tokens = []
    for token in tokens:
        modified_tokens.append(token)
    return " ".join(modified_tokens)
def tokenize2(text):
    text = text.lower()
    tokens = re.split(r"\W+", text)
    modified_tokens = []
    for token in tokens:
        if token:    
            token = token if not token.isdigit() else None
        if token and len(token) < 4:
            token = None
        if token in stop_words:
            token = None
        if token:
            modified_tokens.append(token)
    return modified_tokens

def tokenize_and_stem(text):
    tokenized = tokenize(text)
    return " ".join(stemmer.stemWords(tokenized.split()))

# sutvarko entity su skaiciais
def tokenize_ent(e):
    if not e.isalnum():
        if len(e) < 4:
            return ""
        else:
            e = re.split(r"\W+", e)
            e = [c for c in e if len(c) > 0]
            e = " ".join(e)

    return e

# nenaudoju
# common_punctuation = r".,!?;:'\"()\-\[\]"
# allowed_chars_pattern = rf"[\p{{Latin}}\p{{Cyrillic}}\d\s{re.escape(common_punctuation)}]+"
# def tokenize_ent2(e):
#     # allowed_parts = re.findall(allowed_chars_pattern, e)
#     # e = "".join(allowed_parts)
#     if e.isnumeric():
#         return ""
#     if not e.isalnum():
#         if len(e) < 4:
#             return ""
#         else:
#             e = re.split(r"\W+", e)
#             e = [c for c in e if len(c) > 1]
#             e = " ".join(e)

#     return e

#grazina sujungta
def stem_ru(text):
    return " ".join(stemmer_ru.stemWords(text.split()))
#grazina tokenizuota
def stem_ru2(text):
    ret = stemmer_ru.stemWords(text.split())
    # print(ret)
    return ret

def tokenize_and_stem2(text):
    tokenized = tokenize2(text)
    stemmed1 = stemmer.stemWords(tokenized)
    stemmed2 = stemmer_ru.stemWords(stemmed1)
    return stemmed2
    # return " ".join(stemmed2)
    # return " ".join(stemmer_ru.stemWords(tokenized))



from numba.typed import Dict

# cia is tiesu ne tik url pravalo, o viviskas cleaninimas, BUTINAS pries embeddinant
# @njit
def url_removal(text_list):
    """
    Removes URLs, phone numbers, and emails from a dictionary of text.

    Args:
        text_list: A dictionary where keys are identifiers and values are strings.
        pp: An empty dictionary to store the processed strings.
    """
    # for p, nnnk in text_list.items():

    # pp = []

    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)

    pp = {}
    for k, nnnk in text_list.items():
        # print(nnnk)
        # Remove URLs (http and www)
        # print(nnk)

        nnnk = emoji_pattern.sub(r'', nnnk)

        nnnk = re.sub(r"https?://\S+|www\.\S+", "", nnnk)

        # Remove emails
        nnnk = re.sub(r"\S+@\S+", "", nnnk)

        # Remove phone numbers (more than 6 digits, starting with + or a digit)
        nnnk = re.sub(r"\+?\d[\d -]{5,}\d", "", nnnk) 

        #Remove extra spaces
        nnnk = re.sub(r"\s+", " ", nnnk).strip()

        # if nnnk != "":
            # pp[p] = nnnk
        # pp.append(nnnk)
        pp[k] = nnnk
    
    return pp