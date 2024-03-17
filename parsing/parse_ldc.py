import os
import argparse
import pickle
import pandas as pd
from collections import OrderedDict, defaultdict
import glob
from zipfile import ZipFile
from bs4 import BeautifulSoup as bs
import copy
from copy import deepcopy
from collections import OrderedDict, defaultdict
from tqdm.autonotebook import tqdm
import spacy
from spacy.tokens import Doc
import re
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
import string

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)

def add_patterns_to_phrasematcher(nlp, term_list, attr=None):

    if attr is None:
        nlp = English()

    matcher = PhraseMatcher(nlp.vocab, attr, validate=True)

    # If default PhraseMatcher (i.e., attr='ORTH')
    if attr is None:
        term_patterns_list = [nlp.make_doc(term) for term in term_list]

    # If attribute is specified
    else:
        term_patterns_list = [nlp(term) for term in term_list]

    # Add term patterns in list to PhraseMatcher
    term_label = 'USE_' + str(attr)
    matcher.add(term_label, None, *term_patterns_list)

    return matcher


def print_matches(matches):
    for match_id, start, end in matches:
        #label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        return span, span.lemma_

def read_csv(file_path: str, delim='\t', return_dict=True):
    """
    Converts a csv/tsv file into a list of dicts

    Parameters
    ----------
    file_path : str
        The path of the csv/tsv file

    delim : str
        The delimiter

    return_dict: bool
        To return the csv as a list of dictionaries or just rows

    Returns
    -------
    list
        A list of dictionaries. Column names are the keys of the dicts.
                                (OR)
        A list of rows
    """
    if not return_dict:
        csv = pd.read_csv(file_path, sep=delim, header=None)
        return csv.values.tolist()
    else:
        csv = pd.read_csv(file_path, sep=delim)
        return csv.to_dict('records')

def strip_punct(token):
    #text = re.sub('\ |\?|\.|\"\\,|\!|\/|\;|\:', '', token)
    remove = string.punctuation
    remove = remove.replace("-", "")
    pattern = r"[{}]".format(remove)

    clean_token = re.sub(pattern, "", token)
    return clean_token

def closest(lst, k):
    """
    Gets the closest element <= k in lst
    :param lst:
    :param k:
    :return:
    """
    lst = [a for a in list(lst) if a <= k]
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]


def get_sentence(doc_id, start_char, doc_sent_map):
    """
    Gets the sentence in which the mention resides

    param doc_id: (str) document id
    param start_char: (int) start character offset of the mention
    param doc_sent_map: (dict)
    return: (list of dict) sent_map
    """
    sent_map = doc_sent_map[doc_id]
    try:
        closest_sent_offset = closest(sent_map.keys(), start_char)
    except ValueError:
        print("Problem at " + doc_id)
        return sent_map[list(sent_map.keys())[0]]
    return sent_map[closest_sent_offset]


def get_ltf_doc_info_map_aida(parent_children_path):
    """
    Gets the language and topic/subtopic information of the ltf documents

    Parameters
    ----------
    parent_children_path : str
        The path of the tsv file parent_children.tab

    Returns
    -------
    dict
        a dict of dict containing the desired information. child doc ids are the keys for main dict
            - second dict has these keys: {topic, language}
    """
    csv_dict_rows = read_csv(parent_children_path, delim='\t')
    info_cols = ['parent_uid', 'topic', 'lang_id']
    return {
        row['child_uid']: {
            col: row[col] for col in info_cols
        } for row in csv_dict_rows if 'ltf' in row['child_asset_type']
    }


def get_ltf_map(ltf_xml):
    """
    Gets an ordereddict of the segments of a document based on the start offset of segment

    Parameters
    ----------
    ltf_xml : str
        The xml content of the ltf document

    Returns
    -------
    doc_id, OrderedDict
        a dict of the form {start_offset: segment}
    """
    file_bs = bs(ltf_xml, from_encoding='utf-8')
    if 'UTF-8' not in str(ltf_xml):
        pass
    doc = file_bs.doc
    doc_id = doc['id']
    segments = doc.find_all('seg')
    sent_dict = OrderedDict()
    for sent_id, sentence in enumerate(segments):
        tokens = sentence.find_all('token')
        token_map = {int(tok['start_char']): tok.text for tok in tokens}
        seg_start_char = int(sentence['start_char'])
        sent_text = sentence.original_text.text
        sent_dict[seg_start_char] = {
            'sent_id': sent_id,
            'start_char': seg_start_char,
            'token_map': token_map,
            'sent_text': sent_text
        }
    return doc_id, sent_dict


def get_doc_sent_map_zipped(source_dir, working_folder):
    """
    Create doc to sentences map

    Parameters
    ----------
    source_dir
    working_folder

    Returns
    -------

    """
    #doc_sent_map_path = working_folder + '/doc_sent_map.pkl'  #original source files for LDC2019E77 Annotations and LDC2019E42
    doc_sent_map_path = working_folder + '/doc_sent_map_practice.pkl' #practice files
    ltf_dir = source_dir + "/data/ltf/"
    if not os.path.exists(doc_sent_map_path):
        doc_sent_map = {}
        if len(glob.glob(ltf_dir + '/*.ltf.zip')) == 0:
            print("Error. ltf files not found!")
            raise FileNotFoundError
        for ltf_zip_file in glob.glob(ltf_dir + '/*.ltf.zip'):
            with ZipFile(ltf_zip_file) as myzip:
                for ltf_file in myzip.filelist:
                    with myzip.open(ltf_file) as my_file:
                        xml_content = my_file.read()
                    doc_id, sent_map = get_ltf_map(xml_content)
                    doc_sent_map[doc_id] = sent_map
        pickle.dump(doc_sent_map, open(doc_sent_map_path, 'wb'))
    else:
        doc_sent_map = pickle.load(open(doc_sent_map_path, 'rb'))
    return doc_sent_map


def get_all_topic_rows(ann_dir, file_name):
    """
    Read the tab file in all the topics and return one unified list

    Parameters
    ----------
    ann_dir: str
    file_name: str

    Returns
    -------
    list
    """
    # store the rows from all topics
    rows = []

    # read the file in all topic folders
    for tab_file in glob.glob(ann_dir + f"/data/*/*_{file_name}.tab"):
        topic_rows = read_csv(tab_file)
        rows.extend(topic_rows)

    return rows


def get_mention_map_from_ann(ann_dir, ltf_doc_info_map, doc_sent_map, only_text=True, lang='eng', mention_type='evt'):
    """

    Parameters
    ----------
    ann_dir: str
        The path of the annotation directory that contains mention information in tab files
    ltf_doc_info_map: dict
        The dictionary containing doc topic lang info
    doc_sent_map: dict
        The dictionary containing the mapping of sentences in a document
    mention_type: str
        Either 'evt' or 'ent'
    lang: str
        Either eng, rus, esp, or None (all langs)
    only_text: bool
        Only use mentions from text documents

    Returns
    -------
    dict: The dictionary of the mentions
    """
    # store the mention rows from all topics
    mention_rows = []

    # read the mention annotation files
    ann_mention_rows = get_all_topic_rows(ann_dir, f"{mention_type}_mentions")

    # check modes (text, image, sound, etc)
    for row in ann_mention_rows:
        if not only_text:
            # all modes
            mention_rows.append(row)
        else:
            # only text
            doc_id = row['child_uid']
            if doc_id in ltf_doc_info_map:
                lang_id = ltf_doc_info_map[doc_id]['lang_id']
                if lang is None or lang == lang_id:
                    # check if all languages or matching language
                    mention_rows.append(row)

    # read the linking annotation files
    linking_rows = get_all_topic_rows(ann_dir, "kb_linking")

    # mention linking map
    linking_map = {row['mention_id']: row['kb_id'] for row in linking_rows}

    # annoying! the ldc column names for the id of the mention is different for events and entities :/
    mention_id_col_name = 'eventmention_id'
    if mention_type != 'evt':
        mention_id_col_name = 'argmention_id'

    # make sure the mentionids are unique. no duplicate annotations
    mention_ids = set([row[mention_id_col_name] for row in mention_rows])
    mention_to_m_id = {id_: i for i, id_ in enumerate(mention_ids)}

    # format the mention map. use only mentions from text documents
    mention_map = {
        row[mention_id_col_name]: {
            'm_id': mention_to_m_id[row[mention_id_col_name]],
            'mention_id': row[mention_id_col_name],
            'doc_id': row['child_uid'],
            'start_char': int(row['textoffset_startchar']),
            'end_char': int(row['textoffset_endchar']),
            'gold_cluster': linking_map[row[mention_id_col_name]],
            'topic': ltf_doc_info_map[row['child_uid']]['topic'],
            'lang_id': ltf_doc_info_map[row['child_uid']]['lang_id']
        } for row in mention_rows if row['child_uid'] in ltf_doc_info_map
    }

    # add <m> and </m> around the trigger text of the mention
    for m in mention_map.values():
        if m['doc_id'] in doc_sent_map:
            m_start_char = m['start_char']
            m_end_char = m['end_char'] + 1
            sent_map = get_sentence(m['doc_id'], m_start_char, doc_sent_map)
            s_start_char = sent_map['start_char']
            sent_text = str(sent_map['sent_text'])
            m['mention_text'] = sent_text[m_start_char - s_start_char: m_end_char - s_start_char]
            m['sentence_start_char'] = s_start_char
            m['sentence'] = sent_text
            m['bert_sentence'] = sent_text[: m_start_char - s_start_char] + ' <m> ' + \
                                 m['mention_text'] + ' </m> ' + \
                                 sent_text[m_end_char - s_start_char:]
    add_bert_docs(mention_map, doc_sent_map)

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    sent_ids = [(doc_id, sent_id) for doc_id, sent_map in doc_sent_map.items()
                for sent_id, sent_val in sent_map.items()]
    sentences = [sent_val['sent_text'] for doc_id, sent_map in doc_sent_map.items()
                for sent_id, sent_val in sent_map.items()]

    sent_tuples = list(zip(sentences, sent_ids))

    sent_doc_map = {}
    for doc, sent_id in tqdm(nlp.pipe(sent_tuples, as_tuples=True), desc='spacifying docs', total=len(sent_tuples)):
        sent_doc_map[sent_id] = doc

    for doc_id, sent_map in doc_sent_map.items():
        for sent_id, sent_val in sent_map.items():
            sent_val['sentence_tokens'] = [w.lemma_.lower() for w in sent_doc_map[doc_id, sent_id] if (not (w.is_stop or w.is_punct)) or
                                        w.lemma_.lower() in {'he', 'she', 'his', 'him', 'her'}]
    pickle.dump(doc_sent_map, open(dataset_folder + '/doc_sent_map_practice.pkl', 'wb'))

    add_lexical_features(mention_map, sent_doc_map) # add the lexical features like sentence tokens, lemma, pos tag and pronoun presence etc

    return mention_map



def add_bert_docs(mention_map, doc_sent_map):
    """
    Add the entire document of the mention with the sentence corresponding to the mention
    replaced by the bert_sentence (sentence in which the mention is surrounded by <m> and </m>)
    The key for this is 'bert_doc'

    Parameters
    ----------
    mention_map: dict
    doc_sent_map : dict

    Returns
    -------
    None
    """
    # for each mention create and add bert_doc
    for mention in mention_map.values():
        m_sentence_start_char = mention['sentence_start_char']
        doc_id = mention['doc_id']
        # create the copy of doc_id's sent_map
        m_sent_map = copy.deepcopy(doc_sent_map[doc_id])
        # replace the sentence associated with the mention with bert_sentence
        m_sent_map[m_sentence_start_char]['sent_text'] = mention['bert_sentence']
        # convert sent_map to text
        bert_doc = '\n'.join([sent_map['sent_text'] for sent_map in m_sent_map.values()])
        # add bert_doc in mention
        mention['bert_doc'] = bert_doc

def add_lexical_features(mention_map, sent_doc_map):
    """
    Add lemma, derivational verb, pos tag, pronoun presence(bool) etc
    Parameters
    ----------
    nlp: spacy.tokens.Language
    mention_map: dict

    Returns
    -------
    None
    """
    nlp = spacy.load('en_core_web_md', disable=['ner'])

    mentions = list(mention_map.values())

    mention_sentences = [mention['sentence'] for mention in mentions]
    for mention in mentions:
    if len(mention['mention_text'].split()) > 1:
        m_span = mention['mention_text']
        lemma_matcher = add_patterns_to_phrasematcher(nlp, [m_span], attr='LEMMA')
        text_matcher = add_patterns_to_phrasematcher(nlp, [m_span])
        m_span = mention['mention_text']
        sentence = mention['sentence']
        doc = nlp(sentence)

        try:
            mention_span, mention_span_lemma = print_matches(text_matcher(doc))
        except:
            mention_span, mention_span_lemma = print_matches(lemma_matcher(doc))

        root_span = max(mention_span, key=lambda x: len(x))
        mention['lemma'] = root_span.lemma_
        mention['pos'] = root_span.pos_
        mention['sentence_tokens'] = [w.lemma_.lower() for w in doc if (not (w.is_stop or w.is_punct)) or
                                  w.lemma_.lower() in {'he', 'she', 'his', 'him', 'her'}]
        mention['has_pron'] = len(set(mention['sentence_tokens']).intersection({'he', 'she', 'his', 'him', 'her'})) > 0
        print(m_span, mention_span, root_span.lemma_, root_span.pos_,mention['has_pron'] )

    else:
        doc = sent_doc_map[mention['doc_id'], mention['sentence_start_char']]
        tok_idx = [(token.text,token.i) for token in doc]
        tok_dict = dict(tok_idx)
        m_span = mention['mention_text']

        try:
            mention_span_list = [y for x,y in tok_dict.items() if strip_punct(x) == m_span]
        except KeyError:
            tok_keys = list(tok_dict.keys())
            tok_values = list(tok_dict.values())
            tok_keys = [strip_punct(x) for x in tok_keys]

            toc_dict = {tok_keys:tok_values}
            mention_span_list = [y for x,y in tok_dict.items() if x.rstrip(".")  == m_span]

        mention_span = doc[mention_span_list[0]]
        root_span = mention_span
        mention['lemma'] = root_span.lemma_ # get lemma for single mention
         # pos tag
        mention['pos'] = root_span.pos_

        # sentence tokens
        mention['sentence_tokens'] = [w.lemma_.lower() for w in doc if (not (w.is_stop or w.is_punct)) or
                                    w.lemma_.lower() in {'he', 'she', 'his', 'him', 'her'}]
        # presence of pronouns
        mention['has_pron'] = len(set(mention['sentence_tokens']).intersection({'he', 'she', 'his', 'him', 'her'})) > 0
        print(m_span, mention_span, root_span.lemma_, root_span.pos_,mention['has_pron'] )


def extract_mentions(ann_dir, source_dir, working_folder):
    """
    Extracts the mention information from the ltf files using the annotations tab file

    Parameters
    ----------
    ann_dir : str
        The path of the annotation directory that contains mention information in tab files

    source_dir: str
        The source directory of the LDC source (ltf and topics files)

    working_folder: str
        Path to the working folder to save intermediate files

    Returns
    -------
    dict

        dict of dicts representing the mentions
        dict has these keys: {mention_id, mention_text, doc_id, sent_id, sent_tok_numbers, doc_tok_numbers}
    """
    # path to the parent_children.tab file
    parent_children_path = source_dir + '/docs/parent_children.tab'

    ltf_doc_info_map = get_ltf_doc_info_map_aida(parent_children_path)

    # get the doc sentences map
    doc_sent_map = get_doc_sent_map_zipped(source_dir, working_folder)

    # augment topics from annotations doc_lang_topic.tab
    doc_lang_topic_file = ann_dir + "/docs/doc_lang_topic.tab"
    doc_lang_topic_rows_dict = read_csv(doc_lang_topic_file)
    rootid_dlt_map = {row['root_uid']: row for row in doc_lang_topic_rows_dict}
    for val in ltf_doc_info_map.values():
        if val['parent_uid'] in rootid_dlt_map:
            val['topic'] = rootid_dlt_map[val['parent_uid']]['topic_id']

    # generate mention maps
    # eve_mention_map_file = working_folder + '/evt_mention_map.pkl'
    # ent_mention_map_file = working_folder + '/ent_mention_map.pkl'
    eve_mention_map_file = working_folder + '/evt_mention_map_practice.pkl'
    ent_mention_map_file = working_folder + '/ent_mention_map_practice.pkl'

    if os.path.exists(eve_mention_map_file) and os.path.exists(ent_mention_map_file):
        # if files already there, just load the pickles
        eve_mention_map = pickle.load(open(eve_mention_map_file, 'rb'))
        ent_mention_map = pickle.load(open(ent_mention_map_file, 'rb'))
    else:
        # read the annotation files
        eve_mention_map = get_mention_map_from_ann(ann_dir, ltf_doc_info_map, doc_sent_map, mention_type='evt')
        ent_mention_map = get_mention_map_from_ann(ann_dir, ltf_doc_info_map, doc_sent_map, mention_type='arg')

        # pickle them
        pickle.dump(eve_mention_map, open(eve_mention_map_file, 'wb'))
        pickle.dump(ent_mention_map, open(ent_mention_map_file, 'wb'))

    # stores the event and entity relations
    relations = []

    # add the event arg relations - col names: eventmention_id	slot_type	argmention_id
    relations.extend([(row['eventmention_id'], row['slot_type'], row['argmention_id'])
                      for row in get_all_topic_rows(ann_dir, 'evt_slots')])

    # add the entity-entity relations - col names: relationmention_id	slot_type	argmention_id
    # each relationmention_id has two arguments - head and tail separately
    rel_slots_rows = get_all_topic_rows(ann_dir, 'rel_slots')
    rel_map = defaultdict(list)
    for row in rel_slots_rows:
        rel_map[row['relationmention_id']].append(row)
    for rels in rel_map.values():
        if len(rels) == 2:
            rel1 = rels[0]
            rel2 = rels[1]
            # arg in rel2 is head in rel1 and vice versa
            relations.append((rel2['argmention_id'], rel1['slot_type'], rel1['argmention_id']))
            relations.append((rel1['argmention_id'], rel2['slot_type'], rel2['argmention_id']))

    return eve_mention_map, ent_mention_map, relations, doc_sent_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse the LDC2019E77 Annotations and LDC2019E42 source files to \
    generate a common representation of events')
    parser.add_argument('--ann', '-a', help='Path to the LDC Annotation Directory')  #Phase 1 practice Anno: LDC2019E07, Source: LDC2019E04
    parser.add_argument('--source', '-s', help='Path to the LDC Source Directory')
    args = parser.parse_args()
    print("Using the Annotation Directory: ", args.ann)
    print("Using the Source     Directory: ", args.source)

    extract_mentions(args.ann, args.source, 'mention_maps_ldc_mmcdcr/')
