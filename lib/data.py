from __future__ import print_function
import nltk, glob, sys, os, re, io
from nltk import sent_tokenize
import nltk.tag.stanford
from nltk.tag.stanford import StanfordPOSTagger as POSTagger
nltk.data.path.append("venv/nltk_data")


current_dir = os.getcwd()
_path_to_jar = current_dir + '/stanford-postagger/stanford-postagger.jar'
default_pos_model = POSTagger(model_filename=current_dir + '/stanford-postagger/models/english-left3words-distsim.tagger', path_to_jar=_path_to_jar)
caseless_pos_model = POSTagger(model_filename=current_dir + '/stanford-postagger/models/english-caseless-left3words-distsim.tagger', path_to_jar=_path_to_jar)



class Text(object):
	
	def __init__(self, text, span_annotations={}, span_pair_annotations={}, id=0, lowercase=True, pos=False, transitive_closure=[], conflate_digits=True):
		self.span_annotations = span_annotations
		self.reverse_span_annotations = reverse_dict_list(self.span_annotations)
		self.span_pair_annotations = span_pair_annotations			
		self.reverse_span_pair_annotations = reverse_dict_list(self.span_pair_annotations)
		if len(transitive_closure) > 0:
			for label in transitive_closure:
				self.take_transitive_closure(label)
		self.text = text
		self.tokens, self.spans = tokenize(text, lowercase=lowercase, conflate_digits=conflate_digits)
		self.span_starts, self.span_ends = {s:i for (i,(s,e)) in enumerate(self.spans)}, {e:i for (i,(s,e)) in enumerate(self.spans)}
		self.id = id
		self.pos = []
		self.lowercased = lowercase
		if pos:
			self.pos = self.parse_pos()
		else:
			self.pos = ['no_pos' for tok in self.tokens]
		self.vocabulary, self.pos_vocabulary = self.get_vocabs()
				
		self.sentence_boundaries, self.character_index_to_sentence_index = get_sentence_boundaries(text)
		self.paragraph_boundaries, self.character_index_to_paragraph_index = get_paragraph_boundaries(text)
		if 'CONTAINS' in self.span_pair_annotations and len(self.span_pair_annotations['CONTAINS']) > 0 and not 'clinic' in self.id:
			print(self.id, len(self.span_pair_annotations['CONTAINS']))

	def take_transitive_closure(self, span_pair_label):
		# update reverse and normal annotations
		added_relations = 0
		if not span_pair_label in self.span_pair_annotations:
			return added_relations
		for span_a, span_b in self.span_pair_annotations[span_pair_label]:
			for span_c, span_d in self.span_pair_annotations[span_pair_label]:
				if span_b == span_c:
					new_relation = (span_a, span_d) 
					if not new_relation in self.reverse_span_pair_annotations:
						self.span_pair_annotations[span_pair_label].append(new_relation)
						self.reverse_span_annotations[new_relation] = [span_pair_label]
						added_relations += 1
					elif not span_pair_label in self.reverse_span_pair_annotations[(span_a, span_d)]:
						self.span_pair_annotations[span_pair_label].append(new_relation)
						self.reverse_span_pair_annotations[new_relation].append(span_pair_label)
						added_relations += 1
		print('added', added_relations, 'x', span_pair_label, '(transitive closure)')
		return added_relations						
					
		
	

		

	def parse_pos(self):
		if self.lowercased:
			tagger = default_pos_model
		else:
			tagger = caseless_pos_model
		
		exceptions = {'\n':'NEWLINE','\t':'TAB', '':'NOTHING'} # not handled by stanford POS Tagger

		selected_pos =  [pos for (word,pos) in tagger.tag(self.tokens)]

		final_pos = []
		i=0
		for tok in self.tokens:
			if tok in exceptions:
				final_pos.append(exceptions[tok])
			else:
				final_pos.append(selected_pos[i])
				i+=1

		return final_pos
		
	def get_vocabs(self):

		vocab = {}
		pos_vocab = {}
		for i,token in enumerate(self.tokens):
			pos = self.pos[i] 
			if token in vocab:
				vocab[token].append(i)
			else:
				vocab[token] = [i]
				
			if pos in pos_vocab:
				pos_vocab[pos].append(i)
			else:
				pos_vocab[pos] = [i]
		return vocab, pos_vocab

	def get_sentence_index(self, span):
		return self.character_index_to_sentence_index[span[0]]
		
	def get_paragraph_index(self, span):
		return self.character_index_to_paragraph_index[span[0]]

	def spans_lie_within_one_sentence(self, span1, span2):
		return self.get_sentence_index(span1) == self.get_sentence_index(span2)
	
	def spans_lie_within_one_paragraph(self,span1, span2):
		return self.get_paragraph_index(span1) == self.get_paragraph_index(span2)
		

	def update_annotations(self, span_update=None, span_pair_update=None):
		if span_update:
			for label in span_update:
				self.span_annotations[label] = span_update[label]
			self.reverse_span_annotations = reverse_dict_list(self.span_annotations)
		if span_pair_update:
			for label in span_pair_update:
				self.span_pair_annotations[label] = span_pair_update[label] 
			self.reverse_span_pair_annotations = reverse_dict_list(self.span_pair_annotations)

	def token_distance(self, span_1, span_2):
		first, second = min(span_1[-1],span_2[-1]), max(span_1[0],span_2[0])

		if not second in self.span_starts or not first in self.span_ends:
			return None

		return self.span_starts[second] - self.span_ends[first]

	def tokens_inbetween(self, first, second, pos=False):
		end, start = first[-1], second[0]
		if not end in self.span_ends:
			end = self.get_closest_viable_token_end(end)
		if not start in self.span_starts:
			start = self.get_closest_viable_token_start(start)
		first_index, last_index = self.span_ends[end], self.span_starts[start]
		
		if pos:
			return self.pos[first_index+1:last_index]
		else:
			return self.tokens[first_index+1:last_index]

	def span_to_tokens(self, span, pos=False):
		start, end = span
		if not start in self.span_starts:
			start = self.get_closest_viable_token_start(start)
		if not end in self.span_ends:
			end = self.get_closest_viable_token_end(end)
			
		first_index, last_index = self.span_starts[start], self.span_ends[end]
		
		if pos:
			return self.pos[first_index:last_index+1]
		else:
			return self.tokens[first_index:last_index+1]

	def n_left_tokens_from_span(self, span, length, pos=False):
		start, _ = span
		if not start in self.span_starts:
			start = self.get_closest_viable_token_start(start)
		first_index = self.span_starts[start]
		if pos:
			return self.pos[max(0, first_index-length):first_index]	
		else:
			return self.tokens[max(0, first_index-length):first_index]		
	
	def n_right_tokens_from_span(self, span, length, pos=False):
		_, end = span
		if not end in self.span_ends:
			end = self.get_closest_viable_token_end(end)
		last_index = self.span_ends[end]
		if pos:
			return self.pos[last_index+1:min(last_index + length + 1, len(self.tokens))]
		else:
			return self.tokens[last_index+1:min(last_index + length + 1, len(self.tokens))]

	def span_to_string(self, span):
		return self.text[span[0]:span[1]]
	
	def get_closest_viable_token_start(self, init): # find closest character index of a viable token
		i = 0
		while init + i < len(self.text) and init - i >= 0:
			if init + i in self.span_starts:
				return init + i
			if init - i in self.span_starts:
				return init - i
			i += 1

	def get_closest_viable_token_end(self, init): # find closest character index of a viable token
		i = 0
		while init + i < len(self.text) and init - i >= 0:
			if init + i in self.span_ends:
				return init + i
			if init - i in self.span_ends:
				return init - i	
			i += 1



def transform_to_unidirectonal_relations_text(text):
	new_annotations = transform_to_unidirectonal_relations(text.span_pair_annotations)
	text.span_pair_annotations=new_annotations	
	text.reverse_span_pair_annotations = reverse_dict_list(text.span_pair_annotations)
	return text
	
def transform_to_unidirectonal_relations(span_pair_annotations):
		'''Adds an extra label for each relation pair label, indicating that the relation is in the other direction (w.r.t. word order).
		e.g. If "The dog was walked by John"  would give WALKS(John, dog) we instead introduce the label WALKS_BY, and label it as WALKS_BY(dog, John) instead.
		This allows for unidrectional candidate generation (at the cost of adding an extra label).
		'''
		counter=0
		new_annotations = {}
		for label in span_pair_annotations:
			reverse_label = label + '_INVERSE'
			new_annotations[reverse_label] = []
			new_annotations[label] = []

			for (span_a1, span_a2) in span_pair_annotations[label]:
				if span_a1[0] > span_a2[0]:
					counter+=1
					new_annotations[reverse_label].append((span_a2, span_a1))
				else:
					new_annotations[label].append((span_a1, span_a2))
		return new_annotations			

def transform_to_bidirectonal_relations_text(text):
	new_annotations = transform_to_bidirectonal_relations(text.span_pair_annotations)
	text.span_pair_annotations=new_annotations	
	text.reverse_span_pair_annotations = reverse_dict_list(text.span_pair_annotations)
	return text

	
def transform_to_bidirectonal_relations(span_pair_annotations):
		counter=0
		new_annotations = {}
		for label in span_pair_annotations:
			if label[-8:]=='_INVERSE':
				original_label = label[:-8]
				if not original_label in new_annotations:
					new_annotations[original_label] = []
				
				for (span_a2, span_a1) in span_pair_annotations[label]:
					new_annotations[original_label].append((span_a1, span_a2))
					counter+=1
			else:
				if not label in new_annotations:
					new_annotations[label]=span_pair_annotations[label]
				else:
					new_annotations[label]+=span_pair_annotations[label]
					
		return new_annotations



def read_text_files(directory, lowercase=False, pos=False, verbose=False):
	if verbose:
		print('\nReading txt files from', directory)
	texts = []
	for txt_file in glob.glob(directory +"/*.txt"):
		if verbose:
			print(txt_file)
		with io.open(txt_file, 'r',encoding="UTF8") as f:
			text = Text(f.read(), pos=pos, lowercase=lowercase)
			texts.append(text)
	return texts
			

def get_sentence_boundaries(text):
	sents = sent_tokenize(text)
	sent_id = 0
	j=0
	correction=0
	boundaries = []
	character_index_to_sentence_index = {}
	for i, char in enumerate(text):
		character_index_to_sentence_index[i] = len(boundaries)
		if j==len(sents[sent_id]):
			boundaries.append(i)
			j=0
			sent_id+=1
		if sent_id >= len(sents):
			sent_id = sent_id-1
		sent_char=sents[sent_id][j]
		
		if char==sent_char:
			j+=1
			continue
		if char!=sent_char:
			correction+=1
	return boundaries, character_index_to_sentence_index

def get_paragraph_boundaries(text):
	boundaries = []
	character_index_to_paragraph_index = {}
	prev = ''
	for i, char in enumerate(text):
		character_index_to_paragraph_index[i] = len(boundaries)
		if char == '\n' and prev =='\n':
			boundaries.append(i)
		prev=char
	return boundaries, character_index_to_paragraph_index

def reverse_dict_list(d):
	d_new = {}
	for k,l in d.items():
		for v in l:
			if v in d_new and not k in d_new[v]:
				d_new[v].append(k)
			if not v in d_new:
				d_new[v] = [k]					
	return d_new	


	



def tokenize(text, lowercase=True, conflate_digits=True):
	inclusive_splitters = set([',','.','/','\\','"','\n','=','+','-',';',':','(',')','!','?',"'",'<','>','%','&','$','*','|','[',']','{','}'])
	exclusive_splitters = set([' ','\t'])
	tokens = []
	spans = []
	mem = ""
	start = 0
	for i,char in enumerate(text):	
		if char in inclusive_splitters:
			if mem!="":
				tokens.append(text[start:i])
				spans.append((start,i))
				mem = ""
			tokens.append(text[i:i+1])
			spans.append((i,i+1))
			start = i+1
		elif char in exclusive_splitters:
			if mem!="":
				tokens.append(text[start:i])
				spans.append((start,i))
				mem = ""
				start = i+1
			else:
				start = i+1
		else:
			mem += char

	if not mem=="":
		tokens.append(mem)
	
	if lowercase:
		tokens = [t.lower() for t in tokens]
	if conflate_digits:
		tokens = [re.sub('\d', '5', t) for t in tokens]
	return tokens, spans	



class Logger(object):
	
	def __init__(self, stream, file_name, log_prefix=None):
		self.log = open(file_name, "w")
		self.stream = stream
		if log_prefix:
			self.log.write('LOG_PREFIX:\n' + log_prefix + '\n\nLOG:\n')

	def write(self, message):
		self.stream.write(message)
		self.log.write(message)  

	def write_to_file(self, message):
		self.log.write(message)
		
	def flush(self):
		pass		
