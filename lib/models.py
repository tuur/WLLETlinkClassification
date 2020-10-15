from __future__ import print_function
from .losses import label_entropy
from .candidates import SpanGenerator, SpanPairGenerator
from itertools import chain
from keras.models import Model, model_from_json
from keras.layers import Input, Lambda, Embedding, LSTM, Dense, Dropout, Reshape, GRU, Bidirectional
from keras.layers.merge import Concatenate, Dot
from keras.utils import plot_model
from keras.preprocessing import sequence as keras_sequence
import numpy as np
import os, sys, pickle, random
from copy import copy, deepcopy
from time import time
from gensim.models import Word2Vec
from time import time
from keras import backend as K
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from gensim.models.keyedvectors import KeyedVectors

sys.setrecursionlimit(100000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def train_word2vec_embeddings(texts, emb_dim=50, window_size=5, min_count=1, sg=1):
	print('\nTrained word2vec embs of dim', emb_dim, 'with window size', window_size)	
	tokens = [text.tokens for text in texts]
	model = Word2Vec(tokens, size=emb_dim, window=window_size, min_count=min_count, sg=sg)
	return model.wv	

def visualize_word_vectors(word_vector_model, num=100, print_words=False, pca=True, selected_words=None, plot=True, show=True, to_file=False):

	try:
		word_vector_model.vocab	
	except:
		word_vector_model = word_vector_model.wv
		
	groups = None
	if not selected_words:
		words = random.sample(word_vector_model.vocab, num)
	elif type(selected_words) == dict:
		print('!')
		groups = [group for group in selected_words for w in selected_words[group]]
		words = [w for group in selected_words for w in selected_words[group]]
	else:
		words = selected_words
	
	selection = []
	selection_groups = []
	for i, word in enumerate(words):
		if word in word_vector_model.vocab:
			selection.append(word)
			if groups:
				selection_groups.append(groups[i])
			if print_words:
				most_similar = word_vector_model.most_similar(word)
				print ('\t',word, '<>', most_similar[0][0], '/', most_similar[1][0], '/', most_similar[2][0], '/', most_similar[3][0], '/', most_similar[4][0], '\t',str(word_vector_model[word])[:10], '\tsum:',sum(word_vector_model[word]))
		else:
			print('warning: word <', word, '> not in embedding model.')
			
	if plot:
		X = word_vector_model[[w for w in words if w in word_vector_model]]
		

		if pca:
			pca = PCA(n_components=2)
			X_trans = pca.fit_transform(X)
		else:
			tsne = TSNE(n_components=2)
			X_trans = tsne.fit_transform(X)			
		
		fig, ax = plt.subplots()#figsize=(10, 10), dpi=300)
	
		if groups:
			colors = {}
			for group in groups:	
				colors[group] = np.random.rand(3,)
			for x,y,c,l in zip(X_trans[:, 0], X_trans[:, 1], [colors[gr] for gr in selection_groups], selection_groups):
				ax.scatter(x, y, color=c,label=l)
			
			handles, labels = ax.get_legend_handles_labels()
			legend_labels, new_handles, new_labels = set([]), [], []
			for handle, label in zip(handles, labels):
				if not label in legend_labels:
					new_handles.append(handle)
					new_labels.append(label)
					legend_labels.add(label)
			ax.legend(new_handles,new_labels, loc='upper right', shadow=False)

		else:
			ax.scatter(X_trans[:, 0], X_trans[:, 1])
		
		
		for i, word in enumerate(selection):
			ax.annotate(word, (X_trans[:, 0][i],X_trans[:, 1][i]))


		if to_file:
			fig.savefig(to_file)
		
		if show:
			try:
				plt.show()
			except:
				print('ERROR: not able to show plot.')

			
	

def prepare_word_embedding_matrix(word_embeddings, embedding_dim, vocabulary):
	embedding_matrix = np.zeros((len(vocabulary) , embedding_dim))
	print (embedding_matrix.shape)
	coverage = 0
	for i,word in enumerate(vocabulary):
		if word in word_embeddings.wv.vocab:

			coverage+=1
			embedding_matrix[i] = word_embeddings[word]		
		else:
			print ('missed emb init:', word)
	print('word_embedding coverage', coverage, '/', len(vocabulary))
	return embedding_matrix


def build_word_embedding_layer(name, vocabulary, embedding_dim, word_embedding_weights=None,embedding_layer=None, pos=False, extra=50): # adds some extra embeddings in case of vocabulary extension
	prefix= 'pos_embs_' if pos else 'word_embs_'
	
	if embedding_layer:
		word_embs = embedding_layer
	elif word_embedding_weights:
		
		embedding_matrix = prepare_word_embedding_matrix(word_embedding_weights, embedding_dim, vocabulary)
		word_embs = Embedding(output_dim=embedding_dim, input_dim=len(vocabulary), name= prefix+ name, trainable=True,weights=[embedding_matrix])
	else:
		word_embs = Embedding(output_dim=embedding_dim, input_dim=len(vocabulary), name=prefix + name)
	return word_embs

def build_basic_lstm_encoder(max_sequence_length, vocabulary, lstm_dim, name, embedding_dim=50, word_embedding_weights=None, embedding_layer=None, pos_embedding_layer=None, pos=False, pos_vocab={}, input_dropout=None, bi=None, recurrent_unit='LSTM'):
		word_input = Input(shape=(max_sequence_length,), dtype='int32', name=name)
		word_embs = build_word_embedding_layer(name, vocabulary, embedding_dim, word_embedding_weights,embedding_layer)			
		total_in = word_embs (word_input)
		
		if pos:
			pos_input = Input(shape=(max_sequence_length,), dtype='int32', name=name+'_pos')
			pos_embs = build_word_embedding_layer(name+'_pos', pos_vocab, 40, None, pos_embedding_layer, pos=True)		
			pos_activations = pos_embs (pos_input)
			total_in = Concatenate(name='concat_'+name) ([total_in, pos_activations])
		else:
			pos_input, pos_embs=None, None
		
		if input_dropout:
			total_in = Dropout(input_dropout) (total_in)
		
		if recurrent_unit == 'LSTM':
			recurrent_layer = LSTM(lstm_dim, name='lstm_'+name)
		elif recurrent_unit == 'GRU':
			recurrent_layer = GRU(lstm_dim, name='gru'+name)
		
		if bi:
			lstm = Bidirectional(recurrent_layer) (total_in)
		else:
			lstm = recurrent_layer (total_in)
				
		return word_input, lstm, word_embs, pos_input, pos_embs

def load_extraction_model(in_file):
	print ('\nLoading model from',in_file)
	with open(in_file, 'rb') as fin:
		model, model_structure, model_weights = pickle.load(fin)
	model.model = model_from_json(model_structure)
	model.model.set_weights(model_weights)
	return model
	
def save_extraction_model(model, out_file, with_training_data=False):
	print ('\nSaving',model.name, ' model to',out_file)
	model_copy = copy(model)
	model_copy.model = None
	if not with_training_data:
		model_copy.annotated_texts = []
		model_copy.unannotated_texts = []
	model_structure = model.model.to_json()
	model_weights = model.model.get_weights()
	
	with open(out_file, 'wb') as fout:
		pickle.dump((model_copy,model_structure,model_weights), fout, protocol=pickle.HIGHEST_PROTOCOL)



			

		
		
		
class ExtractionModel(object):
		
	def __init__(self,  annotated_texts, target_label, label_groups, embedding_dim=50, word_embeddings=None, unannotated_texts=[], input_types=[], token_vocabulary=None, pos_vocabulary=None):
		self.annotated_texts = annotated_texts
		self.unannotated_texts = unannotated_texts
		self.target_label = target_label
		self.label_groups = label_groups

		self.unk_token, self.padding_token = '<UNK>', '<PADDING>'
		self.embedding_dim = embedding_dim
		self.word_embeddings = word_embeddings

		if not token_vocabulary:
			self.token_vocab = self.get_vocab(self.annotated_texts + self.unannotated_texts)
		else:
			self.token_vocab = token_vocabulary
			

		self.reverse_token_vocab = {i:w for w,i in self.token_vocab.items()}
			
		self.target_label_vocab = self.get_target_label_vocab()
		self.target_label_vocab_reverse = {index:label for label,index in self.target_label_vocab.items()}
		self.model = None
		self.duo_model = None
		self.input_types = input_types
		self.name = self.target_label
		if not pos_vocabulary:
			self.pos_vocab = self.get_pos_vocab(self.annotated_texts + self.unannotated_texts)
		else:
			self.pos_vocab = pos_vocabulary
		self.prediction_threshold=None
		
	def get_layer_name(self, name):
		# Name convention for input layer names
		return self.target_label + '_' + name 

	def word_sequence_to_vector(self, word_sequence, max_sequence_length, pos=False):
		if len(word_sequence) > max_sequence_length:
			print(word_sequence)
			print ('ERROR in preproc_word_sequence in models.py: max_sequence_length exceeded! (',len(word_sequence), '>', max_sequence_length, ')')
			exit()
		if pos:


			seq = keras_sequence.pad_sequences([[self.pos_vocab[w] if w in self.pos_vocab else self.pos_vocab[self.unk_token] for w in word_sequence]], maxlen=max_sequence_length, dtype='int32', padding='pre', truncating='pre', value=self.pos_vocab[self.padding_token])[0]

		else:

			seq =  keras_sequence.pad_sequences([[self.token_vocab[w] if w in self.token_vocab else self.token_vocab[self.unk_token] for w in word_sequence]], maxlen=max_sequence_length, dtype='int32', padding='pre', truncating='pre', value=self.token_vocab[self.padding_token])[0]

		return seq



	def labels_to_vector(self, labels):
		v = np.zeros(len(self.target_label_vocab))
		for label in labels:
			if label in self.target_label_vocab:
				v[self.target_label_vocab[label]]  = 1.0
		return v

	def get_vocab(self, texts):
		return {w:i for i,w in enumerate(sorted(list(set(chain.from_iterable([text.vocabulary for text in texts] + [[self.unk_token, self.padding_token]])))))}

	def get_pos_vocab(self, texts):
		return {w:i for i,w in enumerate(sorted(list(set(chain.from_iterable([text.pos_vocabulary for text in texts] + [[self.unk_token, self.padding_token]])))))}


	def get_target_label_vocab(self):

		if self.target_label in self.label_groups:
			labels = self.label_groups[self.target_label]
		else:
			labels = [self.target_label]
		
		if len(labels)<10:
			print('\nLabels:',self.target_label, labels)
		else:
			print('\nNum labels:',self.target_label, len(labels),':\t', ', '.join(list(labels)[:9]), ', ...')
			
		return {l:i for (i,l) in enumerate(sorted(list(labels)))}	
		

	def get_target_labels(self, candidate, annotated_text):
		if type(candidate[0])== int and candidate in annotated_text.reverse_span_annotations:
			labels = [l for l in annotated_text.reverse_span_annotations[candidate] if l in self.target_label_vocab]
		elif type(candidate[1]) == tuple and candidate in annotated_text.reverse_span_pair_annotations:
			labels = [l for l in annotated_text.reverse_span_pair_annotations[candidate] if l in self.target_label_vocab]
		else:
			labels = ['OTHER']			
		return labels


	def get_balanced_batch(self, X, Y, input_names, output_names=['output'], batch_size=32):
		# Samples batch_size random items from the labeled and unlabeled set
		x, y = {},{}
		for labeled in [True, False]:
			data_shape = Y[self.get_layer_name(output_names[0])].shape
			random_indices = [np.random.randint(0,data_shape[0]) for i in range(batch_size)]
			for input_name in input_names:
				x[self.get_layer_name(input_name)] = X[self.get_layer_name(input_name)][random_indices,:]
			for output_name in output_names:
				y[self.get_layer_name(output_name)] = Y[self.get_layer_name(output_name)][random_indices,:]	
		return x,y

	def preproc_X(self, texts, labeled=True):
		X = {self.get_layer_name(input_type):[]  for input_type in self.input_types }
		for text in texts:
			for candidate in self.generate_candidates(text):
				x = self.preproc_candidate_x(candidate ,text, labeled)
				for input_type in x:
					X[input_type].append(x[input_type][0])
		return {k:np.array(X[k]) for k in X}

	def score_candidate(self, candidate, text):
		x = self.preproc_candidate_x(candidate ,text)
		prediction_probs = self.model.predict(x)[0]
		return {label:prediction_probs[self.target_label_vocab[label]] for label in self.target_label_vocab}

	def visualize_preproc_x(self, candidate, text, input_type='r_tokens'):
		x = self.preproc_candidate_x(candidate ,text)
		selected_input_type = [inp for inp in x if input_type in inp][0]
		x = x[selected_input_type][0]
		print('VIZX:',' '.join([self.reverse_token_vocab[i] if not self.reverse_token_vocab[i] =='\n' else '<\N>' for i in x if not self.reverse_token_vocab[i] == self.padding_token]))
		

	def prediction_difference_analysis(self, candidate, text, label, input_type='r_tokens'):
		x = self.preproc_candidate_x(candidate ,text)
		selected_input_type = [inp for inp in x if input_type in inp][0]
		baseline_probs = self.model.predict(x)[0]
		baseline_prob = {label:baseline_probs[self.target_label_vocab[label]] for label in self.target_label_vocab}[label]
		
		words = [self.reverse_token_vocab[i] if not self.reverse_token_vocab[i] =='\n' else '<\N>' for i in x[selected_input_type][0]]
		contributions = []
		for i in range(len(x[selected_input_type][0])):
			xmod = deepcopy(x)
			xmod[selected_input_type][0][i] = self.token_vocab[self.unk_token]

			prediction_probs = self.model.predict(xmod)[0]
			
			preds = {label:prediction_probs[self.target_label_vocab[label]] for label in self.target_label_vocab}
			contributions.append((words[i],baseline_prob - preds[label]))
		
		#normalise contributions
		minvalue = min([c for w,c in contributions])
		contributions = [(w,(c-minvalue)) for w,c in contributions]
		summed = sum([c for w,c in contributions])
 		contributions = [(w,float(c)/summed) for w,c in contributions]
			
		return (baseline_prob, contributions)		


	def predict_with_threshold(self, pred_probs, threshold_value):
		pred_labels = []
		for dim_index, dim_value in enumerate(pred_probs):
			if dim_value > threshold_value:

				pred_labels.append(self.target_label_vocab_reverse[dim_index])
		if len(pred_labels) == 0:
			pred_labels.append(self.target_label_vocab_reverse[np.argmax(pred_probs)])
		if len(pred_labels) > 1:
			pred_labels.remove('OTHER')
		
		return pred_labels[0]
		
		
	def predict_with_argmax(self, pred_probs):
		return self.target_label_vocab_reverse[np.argmax(pred_probs)]



	def predict_X(self, X, ignore_labels):
		return [self.get_label_from_vector(yv) for fv in self.model.predict(X)]
		
	def get_label_from_vector(self, vector):
		if self.prediction_threshold:
			predicted_label = self.predict_with_threshold(vector, self.prediction_threshold)
		else:
			predicted_label = self.predict_with_argmax(vector)
		return predicted_label	
		

	def predict(self, texts, ignore_labels=[], verbose=None):
		if verbose:
			print('\nPredicting',self.name, 'for', len(texts), 'texts:')
		all_labels = []
		ignore_labels = set(ignore_labels)
		for text in texts:
			time_0 = time()
			text_labels = {l:[] for l in self.target_label_vocab if not l in ignore_labels}
			candidates = self.generate_candidates(text)
			X = self.preproc_X([text], labeled=True)
			predictions = self.model.predict(X)
			for i, candidate in enumerate(candidates):
				prediction_probs = predictions[i]

				if self.prediction_threshold:
					predicted_label = self.predict_with_threshold(prediction_probs, self.prediction_threshold)
				else:
					predicted_label = self.predict_with_argmax(prediction_probs)
					
				if not predicted_label in ignore_labels:
					text_labels[predicted_label].append(candidate)
			all_labels.append(text_labels)
			if verbose:
				print('text:',str(text.id) + '\tt:',round(float(time()-time_0),1),'s')
		return all_labels


	def preproc_Y(self, texts, labeled=True):
		output = self.get_layer_name('output')
		Y = {output:[]}
		
		for text in texts:
			for candidate in self.generate_candidates(text):
				if labeled:
					target_labels = self.get_target_labels(candidate, text)
					Y[output].append(self.labels_to_vector(target_labels))
				else:
					Y[output].append([1.0]*len(self.target_label_vocab))
		return {k:np.array(Y[k]) for k in Y}


	def get_word_embedding_layers(self, pos=False):
		if pos:
			return [l for l in self.model.layers if 'pos_embs' in l.name] 
		else:
			return [l for l in self.model.layers if 'word_embs' in l.name] 

	def generate_candidates(self, text):
		return self.candidate_generator.generate_candidates(text)
	
	def get_candidate_generation_statistics(self, annotated_texts=None):
		if not annotated_texts:
			annotated_texts = self.annotated_texts
		
		stats = {l:{'labeled':0,'overlap':0} for l in self.target_label_vocab}
		num_candidates = 0
		for text in annotated_texts:
			candidates = self.generate_candidates(text)
			num_candidates += len(candidates)
			
			if self.__class__.__name__ == 'EntityModel' or self.__class__.__name__ == 'EntityLSTMModel':
				labels = text.span_annotations
			elif self.__class__.__name__ == 'RelationModel' or self.__class__.__name__ == 'RelationLSTMModel': 
				labels = text.span_pair_annotations
			for label in self.target_label_vocab:
				missed = []
				if label in labels:
					labeled = set(labels[label])
					overlap = labeled.intersection(candidates)
					stats[label]['labeled'] += len(labeled)
					stats[label]['overlap'] += len(overlap)
					missed = labeled.difference(candidates)
				
				
					
		print('\nCandidate generation stats (', self.name, '):')
		print('num_candidates:\t', num_candidates)
		for label in stats:
			max_recall = float(stats[label]['overlap']) / (stats[label]['labeled'] + 0.000001)
			prior_precision = float(stats[label]['overlap']) / (num_candidates + 0.000001)
			
			print('max_recall:\t', str(round(max_recall,4)).ljust(6), '\tclass_proportion:\t', str(round(prior_precision,4)).ljust(6), '\toverlap', stats[label]['overlap'],'\tlabeled',stats[label]['labeled'] ,'\t', label)
					

	def get_word_vectors(self):
		word_emb_layers = self.get_word_embedding_layers()
		word_vectors = {}
		for word_embs in word_emb_layers:
			name = word_embs.name
			word_vectors[name] = {}
			normalized_weights = word_embs.get_weights()[0]
			for word in self.token_vocab:
				vector = normalized_weights[self.token_vocab[word]]
				word_vectors[name][word] = vector
		return word_vectors	

	def write_embeddings_to_files(self, directory='embs', binary=True):
		if not os.path.exists(directory):
			os.makedirs(directory)
		suffix = '.bin' if binary else '.txt'
		for vector_set_name, vectors in self.get_gensim_word_vectors().items():
			vectors.save_word2vec_format(directory + '/' + vector_set_name + suffix, binary=binary)
	
	
	def get_gensim_word_vectors(self, normalize=True):
		word_vector_dict = self.get_word_vectors()
		forbidden_dict = {'\n':'<newline>',' ':'<space>', '\t':'<tab>', '\r':'<return>'}
		gensim_word_vectors = {}
		tmp_file =  'tmp_' + str(random.randint(0, 10000)) + '.txt'
		for word_embs in word_vector_dict:
			with open(tmp_file, 'w') as f:
				for forbidden_char in forbidden_dict:
					if forbidden_char in word_vector_dict[word_embs]:
						word_vector_dict[word_embs][forbidden_dict[forbidden_char]] = word_vector_dict[word_embs][forbidden_char]
						del word_vector_dict[word_embs][forbidden_char]
				f.write(str(len(word_vector_dict[word_embs])) + ' ' + str(self.embedding_dim) + '\n')
				for word in word_vector_dict[word_embs]:
					f.write(word + ' ' + ' '.join(['{0:.6f}'.format(value) for value in word_vector_dict[word_embs][word]]) + '\n')
			word_vectors = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
			gensim_word_vectors[word_embs] = word_vectors
			os.remove(tmp_file)
		return gensim_word_vectors
		
			
	def build_duo_model(self):
		inputs = {'labeled':[], 'unlabeled':[]}
		inputs['labeled'] = [Input(shape=(self.model.get_layer(self.get_layer_name(i)).input_shape[1],), dtype='int32', name=self.get_layer_name(i)) for i in self.input_types]
		inputs['unlabeled'] = [Input(shape=(self.model.get_layer(self.get_layer_name(i)).input_shape[1],), dtype='int32', name=self.get_layer_name(i))  for i in self.input_types]

		dummy_layer_labeled = Lambda(lambda x: x, name=self.get_layer_name('output'))
		output_labeled = self.model (inputs['labeled'])
		output_labeled = dummy_layer_labeled (output_labeled)
		
		dummy_layer_unlabeled = Lambda(lambda x: x, name=self.get_layer_name('output'))
		output_unlabeled = self.model (inputs['unlabeled'])
		output_unlabeled = dummy_layer_unlabeled (output_unlabeled)
		
		return  Model(inputs=[i for d in inputs for i in inputs[d]], outputs=[output_labeled, output_unlabeled], name=self.target_label + '_duo')
		
	def train(self, iters=50, patience=10, lambda_n=1, optimizer='adam', validation_texts=[], labeled_loss='categorical_crossentropy', unlabeled_loss=label_entropy, batch_size=32):
		print (5*'=','TRAINING (',iters,')', 5*'=')
		# set up training model
		self.model.compile(optimizer=optimizer, loss=labeled_loss, metrics=['accuracy'])				
		
		duo_model = self.build_duo_model()
		duo_model.compile(optimizer=optimizer, loss={self.get_layer_name('output'):labeled_loss, self.get_layer_name('output'): unlabeled_loss}, loss_weights={self.get_layer_name('output'):lambda_n, self.get_layer_name('output'): 1-lambda_n}, metrics={self.get_layer_name('output'): 'accuracy'})
		
		plot_model(duo_model, duo_model.name + '.png')

		# Data preprocessing [TRAINING DATA]
		X, Y = self.preproc_X(self.annotated_texts, labeled=True), self.preproc_Y(self.annotated_texts, labeled=True)
		X_unlabeled, Y_unlabeled = self.preproc_X(self.unannotated_texts, labeled=False), self.preproc_Y(self.unannotated_texts, labeled=False)
		X.update(X_unlabeled)
		Y.update(Y_unlabeled)
		
		# Data preprocessing [VALIDATION DATA]
		if validation_texts:
			Xv, Yv = self.preproc_X(validation_texts, labeled=True), self.preproc_Y(validation_texts, labeled=True)
				
		losses = {'train_loss':[],'validation_loss':[]}
		patience_counter = patience
		for i in range(1,iters+1):
			time_0, printer = time(), '----- iter '+str(i)
			# get balanced batch
			x,y = self.get_balanced_batch(X,Y, self.input_types, batch_size=batch_size)
			
			# train on batch
			duo_model.train_on_batch(x, y)
			

			loss_train = self.model.evaluate(X,Y,verbose=0)
			printer += '\ttrain_loss ' + str(loss_train)[:6]
			losses['train_loss'].append(loss_train)
			
			# calculate validation loss
			if validation_texts:
				loss_val = self.model.evaluate(Xv,Yv,verbose=0)
				printer += '\tval_loss ' + str(loss_val)[:6]
				if len(losses['validation_loss'])> 0 and losses['validation_loss'][-1] < loss_val: # overfitting?
					printer += ' +'
					patience_counter -= 1
				else:
					patience_counter = patience
					printer += ' -'
					
				losses['validation_loss'].append(loss_val)
				if patience_counter < 1:
					printer += '<'
					return
					
	
			time_spend = time() - time_0
			estimated_time = (iters - i) * time_spend
			printer += '\t\tt '+str(time_spend)[:4]+' s\tETA '+str(round(estimated_time,1))+' s'
			print(printer)

class WordEmbeddingSimilarityObjective(ExtractionModel):
	def __init__(self, texts, layer_1, layer_2, shared_vocab, unlabeled_vocab, distance_metric='cosine', model_dir='.', overlapped_sampling=False):
		self.layer_1 = layer_1
		self.layer_2 = layer_2
		self.shared_vocab = shared_vocab
		self.unlabeled_vocab = list(unlabeled_vocab)
		if not overlapped_sampling:
			self.labeled_vocab = [w for w in self.shared_vocab if not w in unlabeled_vocab]
		else:
			self.labeled_vocab = [w for w in self.shared_vocab]
			
		self.distance_metric=distance_metric
		self.target_label_vocab = [self.distance_metric]
		self.shape_1 = self.layer_1.get_weights()[0].shape
		self.shape_2 = self.layer_2.get_weights()[0].shape
		super(WordEmbeddingSimilarityObjective, self).__init__(texts, 'WSim', 'distance', self.shape_1[1], [self.layer_1, self.layer_2], [], ['word_input_1', 'word_input_2'], token_vocabulary=self.shared_vocab)
		
		
		
		word_input_1 = Input(shape=(1,), dtype='int32', name=self.get_layer_name('word_input_1'))
		word_input_2 = Input(shape=(1,), dtype='int32', name=self.get_layer_name('word_input_2'))

	
		layer_1_w1 =  self.layer_1 (word_input_1)
		activation_l1_w1 = Reshape((self.shape_1[1], )) (layer_1_w1)
		layer_1_w2 = self.layer_1 (word_input_2)
		activation_l1_w2 = Reshape((self.shape_1[1], )) (layer_1_w2)
		
		
		layer_2_w1 =  self.layer_2 (word_input_1)
		activation_l2_w1 = Reshape((self.shape_2[1], )) (layer_2_w1)
		layer_2_w2 = self.layer_2 (word_input_2)
		activation_l2_w2 = Reshape((self.shape_2[1], )) (layer_2_w2)
		
		if 	self.distance_metric == 'dot':
			distance_layer_1 = Dot(1)
			distance_layer_2 = Dot(1)
		elif self.distance_metric == 'cosine':
			distance_layer_1 = Dot(1, normalize=True)
			distance_layer_2 = Dot(1, normalize=True)
		elif self.distance_metric == 'difference':
			distance_layer_1 = Lambda(lambda x: K.sum((x[0] - x[1])**2), output_shape=(1,))
			distance_layer_2 = Lambda(lambda x: K.sum((x[0] - x[1])**2), output_shape=(1,))
		else:
			print('ERROR: no valid distance metric provided')
			exit()
			
		dist_layer_1 = distance_layer_1 ([activation_l1_w1, activation_l1_w2])
		dist_layer_2 = distance_layer_2 ([activation_l2_w1, activation_l2_w2])
		
		subtraction_layer = Lambda(lambda x: (x[0] - x[1])**2, name=self.get_layer_name('output'))
		difference_in_distance = subtraction_layer([dist_layer_1, dist_layer_2])
				
		self.model = Model(inputs=[word_input_1, word_input_2],outputs=[difference_in_distance], name='WSim')
		plot_model(self.model, to_file= model_dir + '/' + self.model.name + '.png', show_shapes=True)		
			
	def generate_candidates(self, text, sample_size=1024):
		
		return [(random.choice(self.labeled_vocab), random.choice(self.unlabeled_vocab)) for i in range(sample_size)]	
		
	def preproc_candidate_x(self, candidate_span, text, labeled=True):
		r_w1 = self.word_sequence_to_vector(word_sequence=[candidate_span[0]], max_sequence_length=1)
		r_w2 = self.word_sequence_to_vector(word_sequence=[candidate_span[1]], max_sequence_length=1)
		
		x = {self.get_layer_name('word_input_1'): np.array([r_w1]), self.get_layer_name('word_input_2'): np.array([r_w2])}
		return x			

	def get_target_labels(self, candidate, text):
		return ['OTHER'] # dummy label (as the label is not used in the loss function)	

	
class WordEmbeddingDistanceModel(ExtractionModel): # altough not really an extraction model
	def __init__(self, texts, layers, token_vocabulary, distance_metric='dot', model_dir='.'):
		self.layers = layers
		self.distance_metric=distance_metric
		self.target_label_vocab = [self.distance_metric]
		self.shape = layers[0].get_weights()[0].shape
		super(WordEmbeddingDistanceModel, self).__init__(texts, 'WD', 'distance', self.shape[1], layers, [], ['word_input'], token_vocabulary=token_vocabulary)
		
		
		word_input = Input(shape=(1,), dtype='int32', name=self.get_layer_name('word_input'))
		activations = []
		for layer in self.layers:	
			layer_shape = layer.get_weights()[0].shape
			if not layer_shape == self.shape:
				print('ERROR: Layers not of the same shape!', layers[0], self.shape, layer, layer_shape)
				exit()
			raw_activation = layer (word_input)
			activation = Reshape((self.shape[1], )) (raw_activation)
			activations.append(activation)
			
		print('ACT', len(activations))	
		if 	self.distance_metric == 'dot':
			distance_layer = Dot(1, name=self.get_layer_name('output'))
		elif self.distance_metric == 'cosine':
			distance_layer = Dot(1, normalize=True, name=self.get_layer_name('output'))
		elif self.distance_metric == 'difference':
			distance_layer = Lambda(lambda x: K.sum((x[0] - x[1])**2), name=self.get_layer_name('output'), output_shape=(1,))
			
		
		else:
			print('ERROR: no valid distance metric provided')
			exit()
			
		distance = distance_layer (activations)
		self.model = Model(inputs=[word_input],outputs=[distance], name='embedding_dist')
		plot_model(self.model, to_file= model_dir + '/' + self.model.name + '.png', show_shapes=True)		
			
	def generate_candidates(self, text):

		# return text.spans # also worked quite okay
		candidates = []
		added_tokens = set([])
		random_order = range(len(text.spans))
		np.random.shuffle(random_order)
		for i in random_order:
			span = text.spans[i]
			if not text.span_to_tokens(span)[0] in added_tokens:
				candidates.append(span)
		return candidates
		
	def preproc_candidate_x(self, candidate_span, text, labeled=True):
		r_word = self.word_sequence_to_vector(word_sequence=text.span_to_tokens(candidate_span), max_sequence_length=1)
		x = {self.get_layer_name('word_input'): np.array([r_word])}
		
		return x			

	def get_target_labels(self, candidate, text):
		return ['OTHER'] # dummy label (as the label is not used in the loss function)


	

class SkipGramModel(ExtractionModel):
	
	def __init__(self, texts, window_size=5, target_labels = None, embedding_dim=50, word_embeddings=None, word_embedding_layer=None, min_context_count=3, token_vocabulary=None, pos_vocabulary=None, model_dir = '.', pos_filter=None, left_right=False, model_name='SG'):
		self.window_size = window_size
		self.min_context_count=min_context_count
		self.left_right = left_right
		self.pos_filter =pos_filter
		# set the vocabulary are the target labels
		if target_labels==None and not pos_filter:
			target_labels = set([tok for text in texts for tok in text.vocabulary if len(text.vocabulary[tok]) >= self.min_context_count])
		elif target_labels==None and pos_filter:
			target_labels = set([tok for text in texts for i,tok in enumerate(text.vocabulary) if len(text.vocabulary[tok]) >= self.min_context_count and text.pos[i] in pos_filter])
			
			
		if left_right:
			new_target_labels = set([tok + '_left' for tok in target_labels]).union(set([tok + '_right' for tok in target_labels]))
			target_labels = new_target_labels
		label_groups = {model_name:target_labels} 
		super(SkipGramModel, self).__init__(texts, model_name, label_groups, embedding_dim, word_embeddings, [], ['r_word'], token_vocabulary=token_vocabulary, pos_vocabulary=pos_vocabulary)
		
		# Inputs
		word_input = Input(shape=(1,), dtype='int32', name=self.get_layer_name('r_word'))
		
		# Embedding
		word_embs = build_word_embedding_layer('r_word', self.token_vocab, embedding_dim, word_embeddings, word_embedding_layer)
		projection = word_embs (word_input)
		projection = Reshape((embedding_dim, )) (projection)
		
		#Output Layer
		output = Dense(len(self.target_label_vocab), activation='softmax', name=self.get_layer_name('output')) (projection)
		
		self.model= Model(inputs=[word_input], outputs=[output], name=self.target_label)
		plot_model(self.model, to_file= model_dir + '/' + self.model.name + '.png', show_shapes=True)


	def generate_candidates(self, text, max=2048):
		# return text.spans # also worked quite okay
		candidates = []
		added_tokens = set([])
		random_order = range(len(text.spans))
		random.seed(len(text.spans))
		random.shuffle(random_order)

		for i in random_order:
			span = text.spans[i]
			if not text.span_to_tokens(span)[0] in added_tokens:
				candidates.append(span)
			if len(candidates) > max:
				break
		return candidates
	
	def preproc_candidate_x(self, candidate_span, text, labeled=True):
		r_word = self.word_sequence_to_vector(word_sequence=text.span_to_tokens(candidate_span), max_sequence_length=1)
		x = {self.get_layer_name('r_word'): np.array([r_word])}
		return x				

	def get_target_labels(self, candidate, text):
		deviation = random.choice([-1, 0, 1])
		left_context = text.n_left_tokens_from_span(candidate, self.window_size + deviation)
		right_context = text.n_right_tokens_from_span(candidate, self.window_size + deviation)

		if self.pos_filter:
			left_pos_context = text.n_left_tokens_from_span(candidate, self.window_size, pos=True)
			right_pos_context = text.n_right_tokens_from_span(candidate, self.window_size, pos=True)
			left_context = [tok for i, tok in enumerate(left_context) if left_pos_context[i] in self.pos_filter]
			right_context = [tok for i, tok in enumerate(right_context) if right_pos_context[i] in self.pos_filter]
		
		if self.left_right:
			return [tok+'_left' for tok in left_context] + [tok+'_right' for tok in right_context]
		else:
			return left_context + right_context

class EventSkipGramModel(SkipGramModel):

	def __init__(self, texts, window_size=20, target_labels = None, embedding_dim=50, word_embeddings=None, word_embedding_layer=None, min_context_count=1, token_vocabulary=None, model_dir = '.', left_right=False):
		self.event_label = 'etype_EVENT'
		if target_labels==None:
			target_labels = set([text.span_to_tokens(event_span)[-1] for text in texts for event_span in text.span_annotations[self.event_label] if len(text.span_to_tokens(event_span))>0])

		super(EventSkipGramModel, self).__init__(texts=texts, window_size=window_size, target_labels = target_labels, embedding_dim=embedding_dim, word_embeddings=word_embeddings, word_embedding_layer=word_embedding_layer, min_context_count=min_context_count, token_vocabulary=token_vocabulary, pos_vocabulary=None, model_dir =model_dir, pos_filter=None, left_right=left_right, model_name='ESG')
		print('Labels', self.target_label_vocab)
		
	def preproc_candidate_x(self, candidate_span, text, labeled=True):
		words = [text.span_to_tokens(candidate_span)[-1]] if len(text.span_to_tokens(candidate_span))>0 else []
		r_word = self.word_sequence_to_vector(word_sequence=words, max_sequence_length=1)
		x = {self.get_layer_name('r_word'): np.array([r_word])}
		return x	

	def generate_candidates(self, text):
		return text.span_annotations[self.event_label]
		
	def get_target_labels(self, candidate_span, text):
		left_context = text.n_left_tokens_from_span(candidate_span, self.window_size)
		right_context = text.n_right_tokens_from_span(candidate_span, self.window_size)

		if self.left_right:
			return [tok+'_left' for tok in left_context if tok in self.target_label_vocab] + [tok+'_right' for tok in right_context if tok in self.target_label_vocab]
		else:

			return [event for event in left_context + right_context if event in self.target_label_vocab]
		
		

class SkipGramArgumentModel(ExtractionModel):
	
	def __init__(self, annotated_texts, relation_labels = None, embedding_dim=50, word_embeddings=None, word_embedding_layer=None, min_context_count=3, token_vocabulary=None, model_dir = '.'):
		self.min_context_count=min_context_count
		# set the vocabulary are the target labels
		self.relation_labels =  relation_labels
		target_labels = set([tok for text in annotated_texts for candidate in self.generate_candidates(text) for tok in text.span_to_tokens(candidate[1])])
		
		label_groups = {'SGA2':target_labels} 
		super(SkipGramArgumentModel, self).__init__(annotated_texts,'SGA2', label_groups, embedding_dim, word_embeddings, [], ['r_word'], token_vocabulary=token_vocabulary)
		
		# Inputs
		word_input = Input(shape=(1,), dtype='int32', name=self.get_layer_name('r_word'))
		
		# Embedding
		word_embs = build_word_embedding_layer('r_word', self.token_vocab, embedding_dim, word_embeddings, word_embedding_layer)
		projection = word_embs (word_input)
		projection = Reshape((embedding_dim, )) (projection)
		
		#Output Layer
		output = Dense(len(self.target_label_vocab), activation='softmax', name=self.get_layer_name('output')) (projection)
		
		self.model= Model(inputs=[word_input], outputs=[output], name='SGA2')
		plot_model(self.model, to_file= model_dir + '/' + self.model.name + '.png', show_shapes=True)


	def generate_candidates(self, text):
		# return text.spans # also worked quite okay
		candidates = []
		for label in self.relation_labels:
			if label in text.span_pair_annotations:
				for span_pair in text.span_pair_annotations[label]:
					if len(text.span_to_tokens(span_pair[0]))==1:
						candidates.append(span_pair)
					
		return candidates
	
	def preproc_candidate_x(self, candidate_span_pair, text, labeled=True):
		a1_span, a2_span = candidate_span_pair
		input_word = [random.choice(text.span_to_tokens(a1_span))]
		r_word = self.word_sequence_to_vector(word_sequence=input_word, max_sequence_length=1)
		
		x = {self.get_layer_name('r_word'): np.array([r_word])}
		return x				

	def get_target_labels(self, candidate_span_pair, text):
		a1_span, a2_span = candidate_span_pair

		return text.span_to_tokens(a2_span)
			
class EntityModel(ExtractionModel):
	""" A Model that scores / predicts entity labels (labeled sequences/spans)"""
	
	def __init__(self, annotated_texts, target_label, candidate_labels, label_groups, candidate_generator=None, context_size=10, lstm_dim=50, embedding_dim=50, word_embeddings=None, unannotated_texts=[],max_entity_length=10, word_embedding_layer=None, pos_embedding_layer=None, hidden=[], dropout=0.5, token_vocabulary=None, pos_vocabulary=None,model_dir='.', pos=None, input_dropout=None, bi=None, recurrent_unit='LSTM'):
		inp_types = ['r_left','r_E', 'r_right']
		if pos:
			inp_types += ['r_left_pos','r_center_pos', 'r_right_pos']	
		super(EntityModel, self).__init__(annotated_texts, target_label, label_groups, embedding_dim, word_embeddings, unannotated_texts, inp_types, token_vocabulary=token_vocabulary, pos_vocabulary=pos_vocabulary)
		self.max_entity_length = max_entity_length
		self.candidate_labels = candidate_labels
		if not candidate_generator:
			self.candidate_generator = SpanGenerator(self.candidate_labels)
		else:
			self.candidate_generator = candidate_generator
		self.get_candidate_generation_statistics()
		self.context_size=context_size
		self.lstm_dim=lstm_dim
		self.POS=pos

		# Inputs
		r_left_in, r_left, word_embs, r_left_pos_in, pos_embs = build_basic_lstm_encoder(self.context_size, self.token_vocab, self.lstm_dim, self.get_layer_name('r_left'), self.embedding_dim, self.word_embeddings,embedding_layer=word_embedding_layer, pos_embedding_layer=pos_embedding_layer, input_dropout=input_dropout, bi=bi, recurrent_unit=recurrent_unit)
		r_e1_in, r_e1, _, _, _ = build_basic_lstm_encoder(self.max_entity_length, self.token_vocab, self.lstm_dim, self.get_layer_name('r_E'), self.embedding_dim, self.word_embeddings, embedding_layer=word_embs, pos_embedding_layer=pos_embs, pos_vocab=self.pos_vocab, input_dropout=input_dropout, bi=bi, recurrent_unit=recurrent_unit)
		r_right_in, r_right, _, _, _ = build_basic_lstm_encoder(self.context_size, self.token_vocab, self.lstm_dim, self.get_layer_name('r_right'), self.embedding_dim, self.word_embeddings, embedding_layer=word_embs, pos_embedding_layer=pos_embs, pos_vocab=self.pos_vocab, input_dropout=input_dropout, bi=bi, recurrent_unit=recurrent_unit)
		
		# Concatenate Inputs
		r_DR = Concatenate() ([r_left, r_e1, r_right])
		
		# Hidden layers
		for h in hidden:
			r_DR = Dense(h, activation='sigmoid') (r_DR)
			
		# Dropout	
		r_DR = Dropout(dropout) (r_DR)	
		
		# Output Layer (Softmax)
		q_e_uni_softmax = Dense(len(self.target_label_vocab), activation='softmax', name=self.get_layer_name('output')) (r_DR)
			
		self.model = Model(inputs = [r_left_in, r_e1_in, r_right_in], outputs = [q_e_uni_softmax], name=self.target_label)
		plot_model(self.model, to_file= model_dir + '/' + self.model.name + '.png', show_shapes=True)			
	
	def preproc_candidate_x(self, candidate_span, text, labeled=True):
		
		left_context, E, right_context = text.n_left_tokens_from_span(candidate_span, self.context_size), text.span_to_tokens(candidate_span), text.n_right_tokens_from_span(candidate_span, self.context_size)
		r_left_x = self.word_sequence_to_vector(left_context, self.context_size)
		r_E_x = self.word_sequence_to_vector(E, self.max_entity_length)
		r_right_x = self.word_sequence_to_vector(right_context, self.context_size)
		x = {self.get_layer_name('r_left'):np.array([r_left_x]), self.get_layer_name('r_E'):np.array([r_E_x]), self.get_layer_name('r_right'):np.array([r_right_x])}

		if self.POS:
			left_context_pos, E_pos, right_context_pos = text.n_left_tokens_from_span(candidate_span, self.context_size, pos=True), text.span_to_tokens(candidate_span, pos=True), text.n_right_tokens_from_span(candidate_span, self.context_size, pos=True)
			r_left_pos_x = self.word_sequence_to_vector(left_context_pos, self.context_size, pos=True)
			r_E_pos_x = self.word_sequence_to_vector(E_pos, self.max_entity_length, pos=True)
			r_right_pos_x = self.word_sequence_to_vector(right_context_pos, self.context_size, pos=True)
			
			
			x.update({self.get_layer_name('r_left_pos'):np.array([r_left_pos_x]), self.get_layer_name('r_E_pos'):np.array([r_E_pos_x]), self.get_layer_name('r_right_pos'):np.array([r_right_pos_x])})			

		return x
		


class EntityLSTMModel(ExtractionModel):
	""" A Model that scores / predicts entity labels (labeled sequences/spans)"""
	
	def __init__(self, annotated_texts, target_label, candidate_labels, label_groups, candidate_generator=None, context_size=10, lstm_dim=50, embedding_dim=50, word_embeddings=None, unannotated_texts=[],max_entity_length=10, word_embedding_layer=None, pos_embedding_layer=None, hidden=[], dropout=0.5, token_vocabulary=None, pos_vocabulary=None,model_dir='.', pos=None, input_dropout=None, bi=None, recurrent_unit='LSTM'):
		inp_types = ['r_tokens','r_position_e1']
		if pos:
			inp_types += ['r_pos']	
		super(EntityLSTMModel, self).__init__(annotated_texts, target_label, label_groups, embedding_dim, word_embeddings, unannotated_texts, inp_types, token_vocabulary=token_vocabulary, pos_vocabulary=pos_vocabulary)
		self.max_entity_length = max_entity_length
		self.candidate_labels = candidate_labels
		if candidate_generator:
			self.candidate_generator=candidate_generator
		else:
			self.candidate_generator = SpanGenerator(self.candidate_labels)
		self.get_candidate_generation_statistics()
		self.context_size=context_size
		self.lstm_dim=lstm_dim
		self.POS=pos
		self.input_dropout = input_dropout
		self.hidden = hidden
		self.dropout=dropout
		
		for (arg_num,labels) in [(1,candidate_labels)]:
			for label in labels:
				if not '<' + label + '_'+str(arg_num)+'>' in self.pos_vocab:
					self.pos_vocab.update({'<' + label + '_'+str(arg_num)+'>':len(self.pos_vocab) })
					print('added','<' + label + '_'+str(arg_num)+'>')
				if not '</'+ label + '_'+str(arg_num)+'>' in self.pos_vocab:
					self.pos_vocab.update({'</'+ label + '_'+str(arg_num)+'>':len(self.pos_vocab) })			
					print('added','<' + label + '_'+str(arg_num)+'>')
				if not '<'+ label + '_'+str(arg_num)+'>' in self.token_vocab:
					self.token_vocab.update({'<'+ label + '_'+str(arg_num)+'>':len(self.token_vocab)})
					print('added','<' + label + '_'+str(arg_num)+'>')
				if not '</'+ label + '_'+str(arg_num)+'>' in self.token_vocab:
					self.token_vocab.update({'</' + label + '_'+str(arg_num)+'>':len(self.token_vocab) })
					print('added','<' + label + '_'+str(arg_num)+'>')

		self.reverse_token_vocab = {i:w for w,i in self.token_vocab.items()}

		self.max_sequence_length = 2 + self.max_entity_length  + 2 * self.context_size
		print('\nMax_seq_length', self.max_sequence_length)
		word_input = Input(shape=(self.max_sequence_length,), dtype='int32', name=self.get_layer_name('r_tokens'))
		position_input_e1 = Input(shape=(self.max_sequence_length,1,), dtype='float32', name=self.get_layer_name('r_position_e1'))
		if self.POS:
			print('\nNum POS:', len(self.pos_vocab) )
			
			pos_input = Input(shape=(self.max_sequence_length,), dtype='int32', name=self.get_layer_name('r_pos'))
			pos_embs = build_word_embedding_layer(self.get_layer_name('r_pos'), self.pos_vocab, 20, None, pos_embedding_layer, pos=True)		

			pos_activations = pos_embs (pos_input)
		print('\nRecurrence:', recurrent_unit, '(bi)' if bi else '')
		word_emb_layer = build_word_embedding_layer(self.get_layer_name('r_tokens'), self.token_vocab, self.embedding_dim, word_embeddings, word_embedding_layer)		
		
		word_emb_activations = word_emb_layer (word_input)
		
		if self.POS:
			total_in = Concatenate() ([word_emb_activations, position_input_e1, pos_activations])
		else:
			total_in = Concatenate() ([word_emb_activations, position_input_e1])
		
		if input_dropout:
			total_in = Dropout(input_dropout) (total_in)
		
		if recurrent_unit == 'LSTM':
			recurrent_layer = LSTM(lstm_dim)
		elif recurrent_unit == 'GRU':
			recurrent_layer = GRU(lstm_dim)
		
		if bi:
			recurrent_layer = Bidirectional(self.recurrent_layer) 
	
		r_DR = recurrent_layer (total_in)
		
		# Hidden layers
		for h in self.hidden:
			r_DR = Dense(h, activation='sigmoid') (r_DR)
			
		# Dropout	
		r_DR = Dropout(dropout) (r_DR)		
		
		# Output Layer (Softmax)
		output_layer = Dense(len(self.target_label_vocab), activation='softmax', name=self.get_layer_name('output'))
		q_ee_uni_softmax = output_layer (r_DR)
		
		if pos:
			self.model = Model(inputs = [word_input, position_input_e1, pos_input], outputs = [q_ee_uni_softmax], name=self.target_label)		
		else:		
			self.model = Model(inputs = [word_input, position_input_e1], outputs = [q_ee_uni_softmax], name=self.target_label)


	def preproc_candidate_x(self, candidate_span, text, labeled=True):


		left_context, first_entity, right_context = text.n_left_tokens_from_span(candidate_span, self.context_size)[:self.context_size] , text.span_to_tokens(candidate_span)[:self.max_entity_length],text.n_right_tokens_from_span(candidate_span, self.context_size)[:self.context_size] 

		if len(self.candidate_labels) > 0:
			selection = [l for l in text.reverse_span_annotations[candidate_span] if l in self.candidate_labels]
		else: 
			selection = []
		
		if len(selection) > 0:
			label_first = str(selection[0])
		else:
			label_first = 'SPAN'

		first_entity = ['<'+ label_first + '_1>'] + first_entity + ['</' + label_first + '_1>']

		token_string = left_context + first_entity + right_context

		if self.POS:
			left_context_pos, first_entity_pos, right_context_pos = text.n_left_tokens_from_span(candidate_span, self.context_size, pos=True)[:self.context_size] , text.span_to_tokens(candidate_span, pos=True)[:self.max_entity_length], text.n_right_tokens_from_span(candidate_span, self.context_size, pos=True)[:self.context_size] 
			
			first_entity_pos = ['<'+ label_first + '_1>'] + first_entity_pos + ['</' + label_first + '_1>']
			
			pos_string = left_context_pos + first_entity_pos + right_context_pos
		
		padding_length = self.max_sequence_length - len(token_string)
	
		position_first =   len(left_context) * [0.0] + len(first_entity) * [1.0] + len(right_context) * [0.0]
			
		

		position_e1 =  padding_length * [0.0] + position_first

		if self.POS:
			pos_input = self.word_sequence_to_vector(pos_string, self.max_sequence_length, pos=True)
		token_input = self.word_sequence_to_vector(token_string, self.max_sequence_length)

			
		if self.POS:
			x = {self.get_layer_name('r_tokens'):np.array([token_input]),self.get_layer_name('r_pos'):np.array([pos_input]),  self.get_layer_name('r_position_e1'):np.array([position_e1]).reshape((1, self.max_sequence_length, 1))}
		else:		
			x = {self.get_layer_name('r_tokens'):np.array([token_input]), self.get_layer_name('r_position_e1'):np.array([position_e1]).reshape((1, self.max_sequence_length, 1))}
		return x

class RelationLSTMModel(ExtractionModel):
	""" A Model that scores / predicts relations (labeled sequence pairs) """

	def __init__(self, annotated_texts, target_label, candidate_labels_a1, candidate_labels_a2, label_groups, context_size=10, lstm_dim=50, embedding_dim=50, word_embeddings=None, unannotated_texts=[],max_entity_length=10,max_center_length=50, word_embedding_layer=None, pos_vocabulary=None, pos_embedding_layer=None,hidden=[], dropout=0.5, token_vocabulary=None, model_dir = '.', pos=True, within_sentences=False, unidirectional_candidates=False, input_dropout=None, bi=None, recurrent_unit='LSTM'):	
		inp_types = ['r_tokens','r_position_e1','r_position_e2']
		if pos:
			inp_types += ['r_pos']
		super(RelationLSTMModel, self).__init__(annotated_texts, target_label, label_groups, embedding_dim, word_embeddings, unannotated_texts, inp_types, token_vocabulary=token_vocabulary, pos_vocabulary=pos_vocabulary)
		self.max_entity_length = max_entity_length
		self.max_center_length = max_center_length
		self.candidate_labels_a1 = candidate_labels_a1
		self.candidate_labels_a2 = candidate_labels_a2
		self.candidate_generator = SpanPairGenerator(self.candidate_labels_a1, self.candidate_labels_a2, self.max_center_length, within_sentences=within_sentences, within_paragraphs=True, left_to_right=unidirectional_candidates)
		self.get_candidate_generation_statistics()
		self.context_size=context_size
		self.lstm_dim=lstm_dim
		self.POS = pos
		self.input_dropout = input_dropout
		self.hidden = hidden
		self.dropout=dropout
	
		
		for (arg_num,labels) in [(1,candidate_labels_a1), (2,candidate_labels_a2)]:
			for label in labels:
				if not '<' + label + '_'+str(arg_num)+'>' in self.pos_vocab:
					self.pos_vocab.update({'<' + label + '_'+str(arg_num)+'>':len(self.pos_vocab) })
					print('added','<' + label + '_'+str(arg_num)+'>')
				if not '</'+ label + '_'+str(arg_num)+'>' in self.pos_vocab:
					self.pos_vocab.update({'</'+ label + '_'+str(arg_num)+'>':len(self.pos_vocab) })			
					print('added','<' + label + '_'+str(arg_num)+'>')
				if not '<'+ label + '_'+str(arg_num)+'>' in self.token_vocab:
					self.token_vocab.update({'<'+ label + '_'+str(arg_num)+'>':len(self.token_vocab)})
					print('added','<' + label + '_'+str(arg_num)+'>')
				if not '</'+ label + '_'+str(arg_num)+'>' in self.token_vocab:
					self.token_vocab.update({'</' + label + '_'+str(arg_num)+'>':len(self.token_vocab) })
					print('added','<' + label + '_'+str(arg_num)+'>')

		self.reverse_token_vocab = {i:w for w,i in self.token_vocab.items()}

		self.max_sequence_length = (2 * self.max_entity_length + self.max_center_length + 2 * self.context_size)  + 4
				
		
		print('\nMax_seq_length', self.max_sequence_length)
	
		word_input = Input(shape=(self.max_sequence_length,), dtype='int32', name=self.get_layer_name('r_tokens'))
		position_input_e1 = Input(shape=(self.max_sequence_length,1,), dtype='float32', name=self.get_layer_name('r_position_e1'))
		position_input_e2 = Input(shape=(self.max_sequence_length,1,), dtype='float32', name=self.get_layer_name('r_position_e2'))
		if self.POS:
			print('\nNum POS:', len(self.pos_vocab) )
			
			pos_input = Input(shape=(self.max_sequence_length,), dtype='int32', name=self.get_layer_name('r_pos'))
			pos_embs = build_word_embedding_layer(self.get_layer_name('r_pos'), self.pos_vocab, 20, None, pos_embedding_layer, pos=True)		

			pos_activations = pos_embs (pos_input)
		print('\nRecurrence:', recurrent_unit, '(bi)' if bi else '')
		word_emb_layer = build_word_embedding_layer(self.get_layer_name('r_tokens'), self.token_vocab, self.embedding_dim, word_embeddings, word_embedding_layer)		
		
		word_emb_activations = word_emb_layer (word_input)
		
		if self.POS:
			total_in = Concatenate() ([word_emb_activations, position_input_e1, position_input_e2, pos_activations])
		else:
			total_in = Concatenate() ([word_emb_activations, position_input_e1, position_input_e2])
		
		if input_dropout:
			total_in = Dropout(input_dropout) (total_in)
		
		if recurrent_unit == 'LSTM':
			recurrent_layer = LSTM(lstm_dim)
		elif recurrent_unit == 'GRU':
			recurrent_layer = GRU(lstm_dim)
		
		if bi:
			recurrent_layer = Bidirectional(self.recurrent_layer) 
	
		r_TL = recurrent_layer (total_in)
		
		# Hidden layers
		for h in self.hidden:
			r_TL = Dense(h, activation='sigmoid') (r_TL)
			
		# Dropout	
		r_TL = Dropout(dropout) (r_TL)		
		
		# Output Layer (Softmax)
		output_layer = Dense(len(self.target_label_vocab), activation='softmax', name=self.get_layer_name('output'))
		q_ee_uni_softmax = output_layer (r_TL)
		
		if pos:
			self.model = Model(inputs = [word_input, position_input_e1, position_input_e2, pos_input], outputs = [q_ee_uni_softmax], name=self.target_label)		
		else:		
			self.model = Model(inputs = [word_input, position_input_e1, position_input_e2], outputs = [q_ee_uni_softmax], name=self.target_label)
		plot_model(self.model, to_file= model_dir + '/' + self.model.name + '.png', show_shapes=True)		
		
	def preproc_candidate_x(self, candidate_span_pair, text, labeled=True):
		span_a1, span_a2 = candidate_span_pair
		first, second = min([span_a1,span_a2],key=lambda x:x[0]), max([span_a1,span_a2],key=lambda x:x[0])
		reverse = span_a1[0] > span_a2[0]

		left_context, first_entity, center_context, second_entity, right_context = text.n_left_tokens_from_span(first, self.context_size)[:self.context_size] , text.span_to_tokens(first)[:self.max_entity_length], text.tokens_inbetween(first, second)[:self.max_center_length], text.span_to_tokens(second)[:self.max_entity_length], text.n_right_tokens_from_span(second, self.context_size)[:self.context_size] 
		
		
		
		label_first = [l for l in text.reverse_span_annotations[first] if l in self.candidate_labels_a1+self.candidate_labels_a2][0]
		label_second = [l for l in text.reverse_span_annotations[second] if l in self.candidate_labels_a1+self.candidate_labels_a2][0]
		
		if reverse:
			first_entity = ['<'+ label_first + '_2>'] + first_entity + ['</' + label_first + '_2>']
			second_entity = ['<'+ label_second + '_1>'] + second_entity + ['</' + label_second + '_1>']
		else:
			first_entity = ['<'+ label_first + '_1>'] + first_entity + ['</' + label_first + '_1>']
			second_entity = ['<'+ label_second + '_2>'] + second_entity + ['</' + label_second + '_2>']
				
		token_string = left_context + first_entity + center_context + second_entity + right_context

		if self.POS:
			left_context_pos, first_entity_pos, center_context_pos, second_entity_pos, right_context_pos = text.n_left_tokens_from_span(first, self.context_size, pos=True)[:self.context_size] , text.span_to_tokens(first, pos=True)[:self.max_entity_length], text.tokens_inbetween(first, second, pos=True)[:self.max_center_length], text.span_to_tokens(second, pos=True)[:self.max_entity_length], text.n_right_tokens_from_span(second, self.context_size, pos=True)[:self.context_size] 
			
			if reverse:
				first_entity_pos = ['<'+ label_first + '_2>'] + first_entity_pos + ['</' + label_first + '_2>']
				second_entity_pos = ['<'+ label_second + '_1>'] + second_entity_pos + ['</' + label_second + '_1>']
			else:
				first_entity_pos = ['<'+ label_first + '_1>'] + first_entity_pos + ['</' + label_first + '_1>']
				second_entity_pos = ['<'+ label_second + '_2>'] + second_entity_pos + ['</' + label_second + '_2>']
			
			
			pos_string = left_context_pos + first_entity_pos + center_context_pos + second_entity_pos + right_context_pos
		

		padding_length = self.max_sequence_length - len(token_string)
	
		position_first =   len(left_context) * [0.0] + len(first_entity) * [1.0] + len(center_context + second_entity + right_context) * [0.0]
		position_second =   len(left_context + first_entity + center_context) * [0.0] + len(second_entity) * [1.0] + len(right_context) * [0.0]
			
		
		if reverse:
			token_string = list(reversed(token_string))
			if self.POS:
				pos_string = list(reversed(pos_string))
			position_e1 =  padding_length * [0.0] + list(reversed(position_second))
			position_e2 =  padding_length * [0.0] + list(reversed(position_first))
		else:
			position_e1 =  padding_length * [0.0] + position_first
			position_e2 =  padding_length * [0.0] + position_second

		if self.POS:
			pos_input = self.word_sequence_to_vector(pos_string, self.max_sequence_length, pos=True)
		token_input = self.word_sequence_to_vector(token_string, self.max_sequence_length)

			
		if self.POS:
			x = {self.get_layer_name('r_tokens'):np.array([token_input]),self.get_layer_name('r_pos'):np.array([pos_input]),  self.get_layer_name('r_position_e1'):np.array([position_e1]).reshape((1, self.max_sequence_length, 1)), self.get_layer_name('r_position_e2'):np.array([position_e2]).reshape((1, self.max_sequence_length, 1))}
		else:		
			x = {self.get_layer_name('r_tokens'):np.array([token_input]), self.get_layer_name('r_position_e1'):np.array([position_e1]).reshape((1, self.max_sequence_length, 1)), self.get_layer_name('r_position_e2'):np.array([position_e2]).reshape((1, self.max_sequence_length, 1))}
		return x


class LSTMAutoEncoder(RelationLSTMModel):
			
	def __init__(self, base_model, annotated_texts, model_dir='.', target_labels=None, pos_filter=None, min_context_count=3, model_name='AE', input_types = ['r_tokens', 'r_position_e1', 'r_position_e2'], max_sequence_length=None):
		
		# RETRIEVE ALL ATTRIBUTES FROM THE BASE MODEL
		self.__dict__.update(base_model.__dict__) 
		self.old_target_label_vocab = copy(self.target_label_vocab)
		self.min_context_count=min_context_count
		self.target_label = model_name
		self.annotated_texts=annotated_texts
		self.pos_filter=pos_filter
		if max_sequence_length:
			self.max_sequence_length = max_sequence_length
		
		if target_labels==None and not pos_filter:
			target_labels = set([tok for text in self.annotated_texts for tok in text.vocabulary if len(text.vocabulary[tok]) >= self.min_context_count])
		elif target_labels==None and pos_filter:
			target_labels = set([tok for text in self.annotated_texts for i,tok in enumerate(text.vocabulary) if len(text.vocabulary[tok]) >= self.min_context_count and text.pos[i] in pos_filter])

		self.label_groups = {model_name:target_labels}
		self.target_label_vocab = self.get_target_label_vocab()
		self.target_label_vocab_reverse = {index:label for label,index in self.target_label_vocab.items()}

		input_layers = []
		concat_layers = []
		if 'r_tokens' in input_types:
			word_input = Input(shape=(self.max_sequence_length,), dtype='int32', name=self.get_layer_name('r_tokens'))
			input_layers.append(word_input)
			word_emb_layer = [l for l in self.model.layers if 'word_embs' in l.name][0]	
			word_emb_activations = word_emb_layer (word_input)
			concat_layers.append(word_emb_activations)
		if 'r_position_e1' in input_types:
			p1_layer = Input(shape=(self.max_sequence_length,1,), dtype='float32', name=self.get_layer_name('r_position_e1'))
			concat_layers.append(p1_layer)
			input_layers.append(p1_layer)
		if 'r_position_e2' in input_types:
			p2_layer = Input(shape=(self.max_sequence_length,1,), dtype='float32', name=self.get_layer_name('r_position_e2'))
			concat_layers.append(p2_layer)
			input_layers.append(p2_layer)
		if self.POS:
			pos_input = Input(shape=(self.max_sequence_length,), dtype='int32', name=self.get_layer_name('r_pos'))
			pos_emb_layer = [l for l in self.model.layers if 'pos_embs' in l.name][0]
			pos_activations = pos_emb_layer (pos_input)
			concat_layers.append(pos_activations)
			input_layers.append(pos_input)

			
			
		
		total_in = Concatenate() (concat_layers)
	
		
		if self.input_dropout:
			total_in = Dropout(self.input_dropout) (total_in)
			
		print( [l.name for l in self.model.layers])	
		lstm_layer = [l for l in self.model.layers if 'lstm' in l.name][0]
		r_TL = lstm_layer (total_in)
			
		# Dropout	
		r_TL = Dropout(self.dropout) (r_TL)		
		
		# Output Layer (Softmax)
		output_layer=Dense(len(self.target_label_vocab), activation='softmax', name=self.get_layer_name('output'))
		q_ee_uni_softmax = output_layer (r_TL)


	
		self.model = Model(inputs = input_layers, outputs = [q_ee_uni_softmax], name=self.target_label)


		plot_model(self.model, to_file= model_dir + '/' + self.model.name + '.png', show_shapes=True)		

	def get_target_labels(self, candidate_span_pair, text):
		span_a1, span_a2 = candidate_span_pair
		first, second = min([span_a1,span_a2],key=lambda x:x[0]), max([span_a1,span_a2],key=lambda x:x[0])
		
		first_entity, center_context, second_entity = text.span_to_tokens(first)[:self.max_entity_length], text.tokens_inbetween(first, second)[:self.max_center_length], text.span_to_tokens(second)[:self.max_entity_length]		
		context = first_entity + center_context + second_entity

		if self.pos_filter:
			pos_context = text.span_to_tokens(first, pos=True)[:self.max_entity_length] + text.tokens_inbetween(first, second, pos=True)[:self.max_center_length] + text.span_to_tokens(second, pos=True)[:self.max_entity_length]
			context = [tok for i, tok in enumerate(pos_context) if pos_context[i] in self.pos_filter]
		
		return context


class LSTMArgument2Predictor(LSTMAutoEncoder):
	def __init__(self, base_model, annotated_texts, model_dir='.', target_labels=None, pos_filter=None, min_context_count=1):
		self.old_target_label_vocab = base_model.target_label_vocab
		if not target_labels:
			target_labels = set([tok for text in annotated_texts for candidate in self.generate_candidates(text) for tok in text.span_to_tokens(candidate[1])])
		super(LSTMArgument2Predictor, self).__init__(base_model, annotated_texts, model_dir, target_labels=target_labels, pos_filter=pos_filter, min_context_count=min_context_count, model_name='PA2', input_types=['r_tokens', 'r_position_e1', 'r_position_e2', 'r_pos'], max_sequence_length=base_model.max_entity_length)
	
	
	def preproc_candidate_x(self, candidate_span_pair, text, labeled=True):
		span_a1, span_a2 = candidate_span_pair
		a1_entity = text.span_to_tokens(span_a1)[:self.max_entity_length]
		token_input = self.word_sequence_to_vector(a1_entity, self.max_sequence_length)
		if self.POS:
			pos_input = self.word_sequence_to_vector(a1_entity, self.max_sequence_length, pos=True)

		p1_input = [0.0] * self.max_sequence_length
		p2_input = [0.0] * self.max_sequence_length
		x = {self.get_layer_name('r_tokens'):np.array([token_input]), self.get_layer_name('r_position_e1'):np.array([p1_input]).reshape((1, self.max_sequence_length, 1)), self.get_layer_name('r_position_e2'):np.array([p2_input]).reshape((1, self.max_sequence_length, 1)), self.get_layer_name('r_pos'):np.array([pos_input])}
		
		return x
	
	def generate_candidates(self, text):
		candidates = []
		for label in text.span_pair_annotations:

			if label in self.old_target_label_vocab:
				candidates += text.span_pair_annotations[label]

		print(len(candidates))
		return candidates
	
	def get_target_labels(self, candidate_span_pair, text):
		span_a1, span_a2 = candidate_span_pair
		a2_entity = text.span_to_tokens(span_a2)[:self.max_entity_length]
		return a2_entity
		
	
	
	

class RelationModel(ExtractionModel):
	""" A Model that scores / predicts relations (labeled sequence pairs) """

	def __init__(self, annotated_texts, target_label, candidate_labels_a1, candidate_labels_a2, label_groups, context_size=10, lstm_dim=50, embedding_dim=50, word_embeddings=None, unannotated_texts=[],max_entity_length=10,max_center_length=100, word_embedding_layer=None, pos_embedding_layer=None,hidden=[], dropout=0.5, token_vocabulary=None, model_dir = '.', pos=True, within_sentences=False, unidirectional_candidates=False, input_dropout=None, bi=None, recurrent_unit='LSTM'):	
		inp_types = ['r_left','r_E1', 'r_center', 'r_E2', 'r_right']
		
		if pos:
			inp_types += ['r_left_pos','r_E1_pos', 'r_center_pos', 'r_E2_pos', 'r_right_pos']	
			
		super(RelationModel, self).__init__(annotated_texts, target_label, label_groups, embedding_dim, word_embeddings, unannotated_texts, inp_types, token_vocabulary=token_vocabulary)
		self.max_entity_length = max_entity_length
		self.max_center_length = max_center_length
		self.candidate_labels_a1 = candidate_labels_a1
		self.candidate_labels_a2 = candidate_labels_a2
		self.candidate_generator = SpanPairGenerator(self.candidate_labels_a1, self.candidate_labels_a2, self.max_center_length, within_sentences=within_sentences, within_paragraphs=True, left_to_right=unidirectional_candidates)
		self.get_candidate_generation_statistics()
		self.context_size=context_size
		self.lstm_dim=lstm_dim
		self.POS = pos

			

		# Inputs
		r_left_in, r_left, word_embs, r_left_pos_in, pos_embs = build_basic_lstm_encoder(self.context_size, self.token_vocab, self.lstm_dim, self.get_layer_name('r_left'), self.embedding_dim, self.word_embeddings,embedding_layer=word_embedding_layer, pos_embedding_layer=pos_embedding_layer, pos=self.POS, pos_vocab=self.pos_vocab, input_dropout=input_dropout, bi=bi, recurrent_unit=recurrent_unit)
		r_e1_in, r_e1, _,r_e1_pos_in, _ = build_basic_lstm_encoder(self.max_entity_length, self.token_vocab, self.lstm_dim, self.get_layer_name('r_E1'), self.embedding_dim, self.word_embeddings, embedding_layer=word_embs, pos_embedding_layer=pos_embs, pos=self.POS, pos_vocab=self.pos_vocab, input_dropout=input_dropout, bi=bi, recurrent_unit=recurrent_unit)
		r_center_in, r_center, _, r_center_pos_in, _ = build_basic_lstm_encoder(self.max_center_length, self.token_vocab, self.lstm_dim, self.get_layer_name('r_center'), self.embedding_dim, self.word_embeddings, embedding_layer=word_embs, pos_embedding_layer=pos_embs, pos=self.POS, pos_vocab=self.pos_vocab, input_dropout=input_dropout, bi=bi, recurrent_unit=recurrent_unit)
		r_e2_in, r_e2, _, r_e2_pos_in, _ = build_basic_lstm_encoder(self.max_entity_length, self.token_vocab, self.lstm_dim, self.get_layer_name('r_E2'), self.embedding_dim, self.word_embeddings, embedding_layer=word_embs, pos_embedding_layer=pos_embs, pos=self.POS, pos_vocab=self.pos_vocab, input_dropout=input_dropout, bi=bi, recurrent_unit=recurrent_unit)
		r_right_in, r_right, _,r_right_pos_in, _ = build_basic_lstm_encoder(self.context_size, self.token_vocab, self.lstm_dim, self.get_layer_name('r_right'), self.embedding_dim, self.word_embeddings, embedding_layer=word_embs, pos_embedding_layer=pos_embs, pos=self.POS, pos_vocab=self.pos_vocab, input_dropout=input_dropout, bi=bi, recurrent_unit=recurrent_unit)
		
		
		# Concatenate Inputs
		r_TL = Concatenate() ([r_left, r_e1, r_center, r_e2, r_right])
		
		# Hidden layers
		for h in hidden:
			r_TL = Dense(h, activation='sigmoid') (r_TL)
			
		# Dropout	
		r_TL = Dropout(dropout) (r_TL)		
		
		# Output Layer (Softmax)
		q_ee_uni_softmax = Dense(len(self.target_label_vocab), activation='softmax', name=self.get_layer_name('output')) (r_TL)
		
		inputs = [r_left_in, r_e1_in, r_center_in, r_e2_in, r_right_in]
		if self.POS:
			print('\nNum POS:', len(self.pos_vocab), self.pos_vocab.keys()[:10])
			inputs += [r_left_pos_in, r_e1_pos_in, r_center_pos_in, r_e2_pos_in, r_right_pos_in]
		
		print('\nRecurrence:', recurrent_unit, '(bi)' if bi else '')
		self.model = Model(inputs = inputs, outputs = [q_ee_uni_softmax], name=self.target_label)
		plot_model(self.model, to_file= model_dir + '/' + self.model.name + '.png', show_shapes=True)		

	def preproc_candidate_x(self, candidate_span_pair, text, labeled=True, not_to_vector=False):
		span_a1, span_a2 = candidate_span_pair
		first, second = min([span_a1,span_a2],key=lambda x:x[0]), max([span_a1,span_a2],key=lambda x:x[0])
		reverse = span_a1[0] > span_a2[0]

		left_context, first_entity, center_context, second_entity, right_context = text.n_left_tokens_from_span(first, self.context_size)[:self.context_size] , text.span_to_tokens(first)[:self.max_entity_length], text.tokens_inbetween(first, second)[:self.max_center_length], text.span_to_tokens(second)[:self.max_entity_length], text.n_right_tokens_from_span(second, self.context_size)[:self.context_size] 
		if self.POS:
			left_context_pos, first_entity_pos, center_context_pos, second_entity_pos, right_context_pos = text.n_left_tokens_from_span(first, self.context_size, pos=True)[:self.context_size] , text.span_to_tokens(first, pos=True)[:self.max_entity_length], text.tokens_inbetween(first, second, pos=True)[:self.max_center_length], text.span_to_tokens(second, pos=True)[:self.max_entity_length], text.n_right_tokens_from_span(second, self.context_size, pos=True)[:self.context_size] 

		if reverse:

			left_context, first_entity, center_context, second_entity, right_context = list(reversed(right_context)), second_entity, list(reversed(center_context)), first_entity, list(reversed(left_context))
			if self.POS:
				left_context_pos, first_entity_pos, center_context_pos, second_entity_pos, right_context_pos= list(reversed(right_context_pos)), second_entity_pos, list(reversed(center_context_pos)), first_entity_pos, list(reversed(left_context_pos))
				
		r_left_x = self.word_sequence_to_vector(left_context, self.context_size)
		r_E1_x = self.word_sequence_to_vector(first_entity, self.max_entity_length)
		r_center_x = self.word_sequence_to_vector(center_context, self.max_center_length)
		r_E2_x = self.word_sequence_to_vector(second_entity, self.max_entity_length)
		r_right_x = self.word_sequence_to_vector(right_context, self.context_size)
		
		if self.POS:
			
			r_left_pos_x = self.word_sequence_to_vector(left_context_pos, self.context_size, pos=True)
			r_E1_pos_x = self.word_sequence_to_vector(first_entity_pos, self.max_entity_length, pos=True)
			r_center_pos_x = self.word_sequence_to_vector(center_context_pos, self.max_center_length, pos=True)
			r_E2_pos_x = self.word_sequence_to_vector(second_entity_pos, self.max_entity_length, pos=True)
			r_right_pos_x = self.word_sequence_to_vector(right_context_pos, self.context_size, pos=True)			
		
		x = {self.get_layer_name('r_left'):np.array([r_left_x]), self.get_layer_name('r_E1'):np.array([r_E1_x]), self.get_layer_name('r_center'):np.array([r_center_x]), self.get_layer_name('r_E2'):np.array([r_E2_x]), self.get_layer_name('r_right'):np.array([r_right_x])}
		
		if self.POS:
			x.update({self.get_layer_name('r_left_pos'):np.array([r_left_pos_x]), self.get_layer_name('r_E1_pos'):np.array([r_E1_pos_x]), self.get_layer_name('r_center_pos'):np.array([r_center_pos_x]), self.get_layer_name('r_E2_pos'):np.array([r_E2_pos_x]), self.get_layer_name('r_right_pos'):np.array([r_right_pos_x])})
		return x
	
