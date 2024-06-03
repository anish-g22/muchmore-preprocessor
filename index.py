from tqdm import tqdm
import xmltodict
import json
import brew_distance
import signal
import sys

doc_names_path = "D:/OTH/MuchMore/{}_files.txt"
file_path = "D:/OTH/MuchMore/springer_{}_train_V4.2.tar/{}.{}.abstr.chunkmorph.annotated.xml"
save_path = "D:/OTH/MuchMore_pre_processed/{}/{}.txt"
error_path = "D:/OTH/MuchMore_pre_processed/error.txt"
utf_file_path = "D:/OTH/MuchMore/plain_springer_english_train_V4.2.tar/UTF/{}.ger.abstr.utf8"
edit_dist_log_path = "D:/OTH/MuchMore_pre_processed/edit_log.txt"
ckpt_path = "D:/OTH/MuchMore_pre_processed/ckpt.txt"

global_ckpt_data = 0

def signal_handler(sig, frame):
    global global_ckpt_data
    if global_ckpt_data:
        data = {"curr": global_ckpt_data}
        save_ckpt(data)
        print("\nCheckpoint data saved.")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

Errors = []
Edit_log = ""
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
    return align_file_pairs(file_name, [en_sentences, de_sentences], [en_lines, de_lines])

def align_de_sentences(file_name, text_reference, data):
    xml_text = ''.join(data)
    xml_sentences = data

    cost, edits = brew_distance.distance(text_reference, xml_text, "both", cost=(0, 1, 0.8, 1))
    rel_cost = cost / len(text_reference)
    if (rel_cost > 0.2):
        # print("these are really different: ", (cost, rel_cost, text_reference, xml_text))
        Edit_log += f"{file_name}: Cost <{cost}> Rel Cost <{rel_cost}>\n"
        return data
    #rel_cost, edits
    sentence_counter = 0
    text_sentence_start = 0
    sentences = []

    xml_sentence_lengths = [len(sent) for sent in xml_sentences]
    xml_sentence_ends = [0] * len(xml_sentence_lengths) + [-1]
    for i in range(len(xml_sentence_lengths)):
        xml_sentence_ends[i] = sum(xml_sentence_lengths[0:i+1]) - 1
    #print(xml_sentence_lengths, xml_sentence_ends, len(xml_text), sum(xml_sentence_lengths))
    text_counter, xml_counter = 0, 0
    for edit in edits:
        if edit in ("MATCH", "SUBST"):
            text_counter += 1
            xml_counter += 1
        elif edit == "INS":
            xml_counter += 1
        elif edit == "DEL": 
            text_counter += 1
        else:
            raise AssertionError("unknown edit type" + edit)
        if xml_counter == xml_sentence_ends[sentence_counter]:
            sentences += [text_reference[text_sentence_start:text_counter+1].lstrip()]
            text_sentence_start = text_counter + 1
            sentence_counter += 1
    
    return sentences
    

def align_file_pairs(file_name, sentences, pairs):
    en_sentences, de_sentences = sentences
    reference = ""
    try:
        with open (utf_file_path.format(file_name), 'r', encoding='utf-8') as f:
            reference = f.read()     
        de_sentences = align_de_sentences(file_name, reference, de_sentences)
    except Exception as e:
        Errors.append(e)
    
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
                f.write(line.strip() + "\n")
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

def save_edit_log(log):
    with open(edit_dist_log_path,'a+') as f:
        f.write(log)
        
def save_ckpt(data):
    with open(ckpt_path, 'w') as f:
        data = json.dumps(data)
        f.write(data)

def load_ckpt():
    with open(ckpt_path,'r') as f:
        data = f.read()
        return json.loads(data)


def main_fn():
    global global_ckpt_data
    file_names = sorted(list(get_file_names()))
    print(f"Total Files available: {len(file_names)}")
    count = 0
    saved_ckpt = load_ckpt()["curr"]
    error_files = get_error_files()
    global_ckpt_data = saved_ckpt
    print(f"Saved CheckPoint: {saved_ckpt}")
    erroneous_files = []
    progress_bar = tqdm(total=len(file_names), desc="Reading files", initial=saved_ckpt)
    for fname in file_names[saved_ckpt:]:
        if (fname in error_files):
            count += 1
            global_ckpt_data = count 
            continue
        try:
            global_ckpt_data = count
            en_data, de_data = get_file_pair(fname)
            count += 1
        except Exception as e:
            # print(f"[-] get file pair\n{e}\n\n")
            Errors.append(f"[-] get file pair\n{e}\n\n")
            erroneous_files.append(fname)
            continue

        # save_new_file(fname, "en", en_data)
        save_new_file(fname+" UTF version", "de", de_data)
        progress_bar.update(1)
        count += 1
        global_ckpt_data = count
        # print(f"[+] Count: {count}")
    ckpt = {"curr": count}
    save_ckpt(ckpt)
    save_edit_log(Edit_log)
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
    main_fn()
    # check_punct()
    # print(len(get_error_files()))

run()
