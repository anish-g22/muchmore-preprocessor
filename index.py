from tqdm import tqdm
import xmltodict
import json

doc_names_path = "D:/OTH/MuchMore/{}_files.txt"
file_path = "D:/OTH/MuchMore/springer_{}_train_V4.2.tar/{}.{}.abstr.chunkmorph.annotated.xml"
save_path = "D:/OTH/MuchMore_pre_processed/{}/{}.txt"
error_path = "D:/OTH/MuchMore_pre_processed/error.txt"

Errors = []
# Returns the python dictionary of a file
def get_file_lines(file):
    try:
        with open(file) as f:
            xml_content = f.read()
        xml_dict = xmltodict.parse(xml_content)
        json_op = json.dumps(xml_dict, indent=1)
        py_dict = json.loads(json_op)
        lines = []
        for line in py_dict["document"]["sentence"]:
            lines.append(line)
        return lines
    except Exception as e:
        # print(f"[-] File Open {file}\n{e}\n\n")
        Errors.append(f"[-] File Open {file}\n{e}\n\n")
    finally:
        pass

# Returns the sentence pair ids in a given file
def get_sentence_pairs(lines):
    pairs = {}
    for line in lines:
        if (line['@id'] in pairs.keys()):
            pairs[line['@id']].append(line['@corresp'])
        else:
            pairs[line['@id']] = [line['@corresp']]
    return pairs

# Get the senetence for a given file name and sentence id
def get_sentence(line):
    str = ""
    for word in line:
        temp = ""
        if (word["@pos"] != "PUNCT") or (word["#text"] in "()"):
            temp += " "
        str = str.strip() + temp + word["#text"].strip()
    return str

# Returns all the sentences in order
def get_sentences_all(file):
    lines = get_file_lines(file)
    op = []
    for line in lines:
        op.append(get_sentence(line["text"]["token"]))
    return [op, lines]    

# For each pair of file return the sentences in the order
def get_file_pair(file_name):
    en_file_path = file_path.format("english", file_name, "eng")
    de_file_path = file_path.format("german", file_name, "ger")
    en_sentences, en_lines = get_sentences_all(en_file_path)
    de_sentences, de_lines = get_sentences_all(de_file_path)
    return align_file_pairs([en_sentences, de_sentences], [en_lines, de_lines])

def align_file_pairs(sentences, pairs):
    en_sentences, de_sentences = sentences
    en_lines, de_lines = pairs
    en_pairs = get_sentence_pairs(en_lines)
    de_pairs = get_sentence_pairs(de_lines)
    en_len = len(en_pairs)
    de_len = len(de_pairs)

    en_bool = [1 for i in range(en_len)]
    de_bool = [1 for i in range(de_len)]

    de_updated_sentences = []
    for val in en_pairs.values():
        curr = ""
        for sen in val[0].split(" "):
            if (len(sen) == 0):
                break
            index = int(sen.split("s")[1]) - 1
            if (de_bool[index]):
                curr  = curr + " " + de_sentences[index].strip()
                de_bool[index] = 0
        if (curr!=""):
            de_updated_sentences.append(curr)

    en_updated_sentences = []
    for val in de_pairs.values():
        curr = ""
        for sen in val[0].split(" "):
            if (len(sen) == 0):
                break
            index = int(sen.split("s")[1]) - 1
            if (en_bool[index]):
                curr  = curr + " " + en_sentences[int(sen.split("s")[1]) - 1].strip()
                en_bool[index] = 0
        if (curr != ""):
            en_updated_sentences.append(curr)
    
    return [en_updated_sentences, de_updated_sentences]
    
# Retrieve the file names that are in both the languages
def get_file_names():
    lang = ["en", "de"]
    names = {}
    for l in lang:
        try:
            with open(doc_names_path.format(l)) as f:
                lines = set()
                for line in  f.readlines():
                    sp_line = line.split(".")
                    lines.add("{}.{}".format(sp_line[0], sp_line[1]))
                names[l] = lines
        except Exception as e:
            # print(f"[-] File Names\n{e}\n\n")
            Errors.append(f"[-] File Names\n{e}\n\n")
        finally:
            pass
    return names["en"].intersection(names["de"])

def save_new_file(file_name, lang, data):
    saved_path = save_path.format(lang, file_name)
    try:
        with open(saved_path, 'a') as f:
            f.truncate(0)
            for line in data:
                f.write(line + "\n")
    except Exception as e:
        # print(f"[-] save new file {file_name} | Language : {lang} \n{e}\n\n")
        Errors.append(f"[-] save new file {file_name} | Language : {lang} \n{e}\n\n")
    finally:
        # print(f"[+] Saved file {file_name} | Language: {lang}")
        pass

def get_error_files():
    try:
        fname = set()
        with open(error_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fname.add(line.strip())
        return fname
    except Exception as e:
        # print(f"[-] error.txt\n\n{e}\n\n")
        Errors.append(f"[-] error.txt\n\n{e}\n\n")

def main_fn():
    file_names = sorted(list(get_file_names()))
    print(f"Total Files available: {len(file_names)}")
    count = 0
    # saved_ckpt = 13
    error_files = get_error_files()
    erroneous_files = []
    for fname in tqdm(file_names, desc="Reading files"):
        if (fname in error_files):
            continue
        try:
            en_data, de_data = get_file_pair(fname)
        except Exception as e:
            # print(f"[-] get file pair\n{e}\n\n")
            Errors.append(f"[-] get file pair\n{e}\n\n")
            erroneous_files.append(fname)
            continue

        save_new_file(fname, "en", en_data)
        save_new_file(fname, "de", de_data)
        count += 1
        # print(f"[+] Count: {count}")

    print("\n\n--------\n\nRun completed")
    
def check_punct():
    file_names = sorted(list(get_file_names()))
    print(f"Total Files available: {len(file_names)}")
    count = 0
    # saved_ckpt = 13
    error_files = get_error_files()
    erroneous_files = []
    for fname in tqdm(file_names, desc="Reading files"):
        if (fname in error_files):
            continue
        try:
            with open(save_path.format("de",fname)) as f:
                de_data = f.readlines()
                for line in de_data:
                    if ("(" in line):
                        count += 1
                        print(f"{fname}")
                        break
        except Exception as e:
            # print(f"[-] get file pair\n{e}\n\n")
            Errors.append(f"[-] get file pair\n{e}\n\n")
            erroneous_files.append(fname)
            continue        
        
    
    print(f"Number of files with more than one full stop: {count}\n\n")

    print("\n\n--------\n\nRun completed")

def run():
    # main_fn()
    check_punct()
    # print(len(get_error_files()))

run()
