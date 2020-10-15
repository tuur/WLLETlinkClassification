from __future__ import print_function
from lib.models import EntityModel, EntityLSTMModel, RelationModel, RelationLSTMModel, SkipGramModel, EventSkipGramModel, LSTMArgument2Predictor, SkipGramArgumentModel, load_extraction_model, save_extraction_model, train_word2vec_embeddings, visualize_word_vectors, WordEmbeddingDistanceModel, WordEmbeddingSimilarityObjective, LSTMAutoEncoder
from lib.evaluation import get_evaluation, analyse_mistakes
from lib.thyme import read_thyme_documents, write_texts_to_thyme_anafora_xml
from lib.multitasking import MultitaskTrainer
from lib.data import read_text_files, Logger, transform_to_unidirectonal_relations_text, transform_to_bidirectonal_relations_text, transform_to_bidirectonal_relations
from keras import optimizers
from lib.candidates import TokenGenerator
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import argparse, os, shutil, sys, pickle, glob
from time import time
from copy import copy


parser = argparse.ArgumentParser(description='Run Experiment (on THYME)')
parser.add_argument('-train_data', type=str, default=None,
                    help='Labeled training data directory for the main task.')
parser.add_argument('-validation_data', type=str, default=None,
                    help='Labeled validation data directory.')
parser.add_argument('-test_data', type=str, default=None,
                    help='Labeled testing data directory.')
parser.add_argument('-proxy_data', type=str, default=None,
                    help='Possibly unlabeled training data directory for the proxy task.')
parser.add_argument('-save_data', type=str, default=None,
                    help='Save all data to pickle file. default:none')
parser.add_argument('-load_data', type=str, default=None,
                    help='Load all data from pickle file. default:none')
parser.add_argument('-max_train_data', type=int, default=1000000000,
                    help='maximum amount of labeled documents used for training (default:1000000)')
parser.add_argument('-max_test_data', type=int, default=1000000000,
                    help='maximum amount of labeled documents used for testing (default:1000000)')
parser.add_argument('-max_proxy_data', type=int, default=1000000000,
                    help='maximum amount of documents used for the proxy task (default:1000000)')																				
parser.add_argument('-validation_size', type=int, default=3,
                    help='Size of validation set, in instances (default:3)')
parser.add_argument('-model_dir', type=str, default='./prototype_model',
                    help='Model directory, for saving and loading the models (default: "./models")')																				
parser.add_argument('-bs', type=int, default=32,
                    help='Batch size (default:32)')
parser.add_argument('-lstm_dim', type=int, default=100,
                    help='LSTM dimension size (default:100)')
parser.add_argument('-hidden', type=str, default="",
                    help='Dimensions of hidden layers: e.g. 100-50-25 (default:none)')
parser.add_argument('-context_size', type=int, default=10,
                    help='Context size to the left and right (default:10)')
parser.add_argument('-embedding_dim', type=int, default=25,
                    help='Word embedding dimension size (default:25)')
parser.add_argument('-patience', type=int, default=20,
                    help='Patience parameter used for early stopping (default:20)')																				
parser.add_argument('-num_epochs', type=int, default=100,
                    help='Maximum number of training epochs (default:100)')	
parser.add_argument('-task', type=str, default='CR',
                    help='Specify the task: CR, DR, ES, or TS (default:CR)')
parser.add_argument('-proxy', type=str, default=None,
                    help='Specify the proxy: SG or SGLR ... (default: None)')
parser.add_argument('-task_weight', type=float, default=1.0,
                    help='Task weight in the loss function (default:1.0)')																				
parser.add_argument('-task_loss', type=str, default='categorical_crossentropy',
                    help='Task loss function: categorical_crossentropy, hinge, modified_hinge (default:categorical_crossentropy)')	
parser.add_argument('-proxy_loss', type=str, default='categorical_crossentropy',
                    help='Proxy loss function: categorical_crossentropy, hinge, modified_hinge (default:categorical_crossentropy)')	
parser.add_argument('-proxy_weight', type=float, default=1.0,
                    help='Proxy task weight in the loss function (default:1.0)')																				
parser.add_argument('-validation_interval', type=int, default=1,
                    help='Validation interval (default:1)')
parser.add_argument('-sharing_type', type=str, default='hard',
                    help='Type of parameter sharing of word embeddings: hard or soft (default:hard)')																		
parser.add_argument('-sharing_weight', type=float, default=1.0,
                    help='Soft sharing weight (default:1.0)')																		
parser.add_argument('-distance_metric', type=str, default='dot',
                    help='Distance metric used for soft sharing: dot or cosine (default:dot)')
parser.add_argument('-soft_sharing_data', type=str, default='train',
                    help='Data that should be used for the soft sharing: train or proxy (default:train)')
parser.add_argument('-to_xml', type=int, default=1,
                    help='Write predictions to xml out in the model_dir (default:None)')
parser.add_argument('-model_type', type=str, default='sequential',
                    help='Type of relation model: compositional, sequential (default:sequential)')
parser.add_argument('-lowercase', type=int, default=1,
                    help='Lowercase the text (default:1)')
parser.add_argument('-conflate_digits', type=int, default=1,
                    help='Conflate digits to default value like 12->55 and 1991->5555 (default:1)')
parser.add_argument('-within_sentences', type=int, default=0,
                    help='Generate candidates within sentences (default:0)')
parser.add_argument('-closure', type=int, default=0,
                    help='Use the transitive closure for CONTAINS and BEFORE (default:0)')
parser.add_argument('-pos', type=int, default=1,
                    help='Use Part-of-Speech Tags (default: 1)')
parser.add_argument('-transform_to_unidirectional', type=int, default=0,
                    help='Transform relation extraction to unidrectional task by adding inverse labels and changing candidate generation to left_to_right arcs only. 1 sees the problem as a 3 class classification, 2 constructs two one-vs-rest models. (default:0)')																					
parser.add_argument('-file_regex_task', type=str, default='.*xml',
                    help='Regular expression used to filter xml files for train, validation and test data (default:".*clin.*Temp.*")')																					
parser.add_argument('-file_regex_proxy', type=str, default='.*',
                    help='Regular expression used to filter xml files for the proxy data (default:".*Temp.*")')
parser.add_argument('-train', type=int, default=1,
                    help='Train the model (default:1)')	
parser.add_argument('-predict', type=int, default=1,
                    help='Predict with the model on the given test set (default:1)')	
parser.add_argument('-proxy_headstart', type=int, default=0,
                    help='First train the proxy task for n iterations before starting the main task (default:10)')	
parser.add_argument('-plot_embeddings', type=int, default=0,
                    help='Plot the word embeddings in a GUI (default:0)')	
parser.add_argument('-min_token_count', type=int, default=2,
                    help='Minimum word frequency for words to be part of the vocabulary (others are <UNK>). (default:2)')	
parser.add_argument('-input_dropout', type=float, default=0.5,
                    help='Dropout on the LSTM input (default:0.5)')
parser.add_argument('-init_with_pretrained_sg_w2v_embeddings', type=int, default=1,
                    help='Initialize word embeddings with w2v skip gram embedding.')	
parser.add_argument('-fix_embeddings_at', type=int, default=-1,
                    help='Fix the embedding layers trainability at a certain epoch number. 0 can be given if they should be fixed from the beginning. (default:-1, i.e. always trainable)')	
parser.add_argument('-fix_proxy_at', type=int, default=-1,
                    help='Fix the proxy weights at a certain point (default=-1, i.e. never)')	
parser.add_argument('-lr', type=float, default=.001,
                    help='Adam learning rate (default:0.001)')	
parser.add_argument('-bi', type=int, default=0,
                    help='Use bidirectional LSTMs (default:0)')
parser.add_argument('-recurrent_unit', type=str, default='LSTM',
                    help='Type of recurrent units to be used, GRU or LSTM (default:LSTM)')																				
parser.add_argument('-proxy_window_size', type=int, default=2,
                    help='Contextual window size of proxy (default:2)')
parser.add_argument('-w2v_window_size', type=int, default=2,
                    help='Contextual window size of w2v initialization (default:2)')
parser.add_argument('-sharing_loss', type=str, default='minimize_output',
                    help='Loss used for the sharing objectives WD or WSim (default:minimize_output, alternative: minimize_output2)')
parser.add_argument('-low_memory_mode', type=int, default=1,
                    help='When using proxies, it uses less memory, but is often slower. (default:1)')
parser.add_argument('-main_proxy', type=str, default=None,
                    help='Use a proxy objective for the main model on the training data')																				
parser.add_argument('-main_proxy_weight', type=float, default=0.1,
                    help='Weight used for the main proxy (default:0.1)')																				
parser.add_argument('-wsim_proxy_max_count', type=int, default=0,
                    help='Amount of time a word should occur in the supervised data in order to not be influenced by the wsim. Normally only words that do not occur in the supervised/training data but do occur in the unsupervised/proxy data are influenced by wsim. So (default:0)')
args = parser.parse_args()

time_0 = time()

if not (args.train_data or args.test_data or args.proxy_data or args.load_data):
	print('ERROR: Provide some sort of data...')
	exit()


if args.train:
	# Create an empty model directory
	if os.path.exists(args.model_dir):
		shutil.rmtree(args.model_dir)
	os.makedirs(args.model_dir)

	# logging and printing 
	sys.stdout = Logger(stream=sys.stdout, file_name=args.model_dir + '/log.log', log_prefix=str(args))
	sys.stderr = sys.stdout


	# Help variables
	shared_embedding_layer, data, ignore_labels, hidden = None, {}, [], [int(layer_dim) for layer_dim in args.hidden.split('-') if layer_dim != '']

	sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# Some hyperparameters
	optimizer, max_center_length,  = adam, 30
	
else:
	# logging and printing 
	sys.stdout = Logger(stream=sys.stdout, file_name=args.model_dir + '/log_pred.log', log_prefix=str(args))
	sys.stderr = sys.stdout	


if args.load_data:
	with open(args.load_data, 'rb') as f:
		print('\nLoading pickled data from', args.load_data)
		data = pickle.load(f)

if args.closure:
	closure = ['CONTAINS','BEFORE']
else:
	closure = []
# Load the data
if args.train_data:
	data['train'] = read_thyme_documents(args.train_data,regex=args.file_regex_task, pos=args.pos, closure=closure, lowercase=args.lowercase, conflate_digits=args.conflate_digits)


if args.proxy_data and args.proxy_data == args.train_data and args.file_regex_task == args.file_regex_proxy:
	data['proxy'] = data['train']
elif args.proxy_data:
	data['proxy'] = read_thyme_documents(args.proxy_data,regex=args.file_regex_proxy, pos=args.pos, closure=[], lowercase=args.lowercase, less_strict=True, conflate_digits=args.conflate_digits)[:args.max_proxy_data]		


if args.validation_data:
	data['valid'] = read_thyme_documents(args.validation_data,regex=args.file_regex_task, pos=args.pos, closure=[], lowercase=args.lowercase, conflate_digits=args.conflate_digits)[:args.validation_size]
	
elif not args.load_data or not 'valid' in data:
	data['valid'] = data['train'][-1 * args.validation_size:]
	data['train'] = data['train'][:-1 * args.validation_size]		

	
if args.test_data:
	data['test'] = read_thyme_documents(args.test_data,regex=args.file_regex_task, pos=args.pos, closure=[], lowercase=args.lowercase, conflate_digits=args.conflate_digits)
data['test'] = data['test'][:args.max_test_data]
	
	#data['proxy'] = read_text_files(args.proxy_data, regex=args.file_regex_proxy, pos=args.pos, lowercase=args.lowercase)[:args.max_proxy_data]

data['train'] = data['train'][:args.max_train_data]
data['valid'] = data['valid'][:args.validation_size]
data['test'] = data['test'][:args.max_test_data]

if sum([len(data[x]) for x in ['train','valid','test']]) == 0:
	print('ERROR: No data found...')
	print(data)
	exit()
	
print(args.proxy_data)
	
print('\n'+'\t'.join(key + ': ' + str(len(data[key])) for key in data))	

if args.save_data:
	print('\nSaving data to', args.save_data)
	with open(args.save_data, 'wb') as f:
		pickle.dump(data, f)

if args.transform_to_unidirectional:
	for text in data['train']:
		text = transform_to_unidirectonal_relations_text(text)
	for text in data['valid']:
		text = transform_to_unidirectonal_relations_text(text)
	for text in data['test']:
		text = transform_to_unidirectonal_relations_text(text)	

# get shared vocab
all_vocabs = [text.get_vocabs() for dataset in data for text in data[dataset] if not dataset=='test']
token_vocabs = {dataset:{} for dataset in data}	
for dataset in data:
	for text in data[dataset]:
		vocab_tok, vocab_pos = text.get_vocabs() 
		for w in vocab_tok:
			if not w in token_vocabs[dataset]:
				token_vocabs[dataset][w] = 0
			token_vocabs[dataset][w] += len(vocab_tok[w])

word_counts = {}
for vocab, pos_vocab in all_vocabs:
	for w in vocab:
		if not w in word_counts:
			word_counts[w] = 0
		word_counts[w] += 1
		
shared_vocab = {tok:i for i,tok in enumerate(list([w for w in word_counts if word_counts[w] >= args.min_token_count]) + ['<UNK>', '<PADDING>'])}

non_overlapping_words = {w for w in shared_vocab if w in token_vocabs['proxy'] and (not w in token_vocabs['train'] or token_vocabs['train'][w] <= args.wsim_proxy_max_count)}
print('\nNon-overlapping tokens:', len(non_overlapping_words))

#shared_vocab = {tok:i for i,tok in enumerate(list(set([w for vocabs in shared_vocabs for w in vocabs[0].keys()] + ['<UNK>', '<PADDING>'])))}
initial_shared_vocab = copy(shared_vocab)
shared_pos_vocab = {pos:i for i,pos in enumerate(list(set([w for vocabs in all_vocabs for w in vocabs[1].keys()] + ['<UNK>', '<PADDING>'])))}

if args.model_type =='sequential':
	#extend vocabulary
	if args.task == 'CR':
		labs = ['etype_EVENT','etype_TIMEX3','etype_SECTIONTIME', 'etype_DOCTIME']
	elif args.task == 'DR':
		labs = ['etype_EVENT']
	elif args.task == 'ES':
		labs = ['SPAN']
	elif args.task == 'TS':
		labs = ['SPAN']
		
	for candidate in labs:
		prefix_start, prefix_end, a1_suffix, a2_suffix = '<' + candidate, '</' + candidate, '_1>', '_2>'
		shared_vocab[prefix_start + a1_suffix] = len(shared_vocab)
		shared_pos_vocab[prefix_start + a1_suffix] = len(shared_pos_vocab)
		shared_vocab[prefix_end + a1_suffix] = len(shared_vocab)
		shared_pos_vocab[prefix_end + a1_suffix] = len(shared_pos_vocab)
		if args.task == 'CR':
			shared_pos_vocab[prefix_start + a2_suffix] = len(shared_pos_vocab)
			shared_vocab[prefix_start + a2_suffix] = len(shared_vocab)
			shared_vocab[prefix_end + a2_suffix] = len(shared_vocab)
			shared_pos_vocab[prefix_end + a2_suffix] = len(shared_pos_vocab)	
		
print('\nVocab size:', len(shared_vocab))
	

if args.init_with_pretrained_sg_w2v_embeddings and args.train:
	# TODO:
	print('\nPre-training word embeddings.')
	modified_texts = [[tok if tok in shared_vocab else '<UNK>' for tok in text.tokens] for text in data['train']+data['proxy']]
	initial_word_embedding_weights = Word2Vec(modified_texts, size=args.embedding_dim, window=args.w2v_window_size, min_count=1, sg=1)
	visualize_word_vectors(initial_word_embedding_weights, show=args.plot_embeddings)
	os.mkdir(args.model_dir + '/embs/')
	initial_word_embedding_weights.wv.save_word2vec_format(args.model_dir + '/embs/init_w2v_sg.bin', binary=True)
else:
	initial_word_embedding_weights=None
	
if args.task == 'CR':
	print('\nModel Type:', args.model_type)
	test_labels = [text.span_pair_annotations for text in data['test']]
if args.task in ['DR', 'ES', 'TS']:
	print('\nModel Type:', args.model_type)
	test_labels = [text.span_annotations for text in data['test']]


if args.train:
	multitask = MultitaskTrainer(optimizer=optimizer, model_dir=args.model_dir)
	
	# Set up the Proxy model
	
	if args.proxy == 'SG':
		print('\nProxy:', args.proxy)
		proxy_model = SkipGramModel(data['proxy'], window_size=args.proxy_window_size, embedding_dim=args.embedding_dim, word_embeddings=initial_word_embedding_weights, model_dir=args.model_dir, pos_vocabulary=shared_pos_vocab, token_vocabulary=shared_vocab)
		shared_vocab = proxy_model.token_vocab
		multitask.add_task(args.proxy ,proxy_model, args.proxy_loss, args.proxy_weight, proxy_task=True)
	elif args.proxy == 'SGLR':
		print('\nProxy:', args.proxy)
		proxy_model = SkipGramModel(data['proxy'], window_size=args.proxy_window_size, embedding_dim=args.embedding_dim, word_embeddings=initial_word_embedding_weights, model_dir=args.model_dir, pos_vocabulary=shared_pos_vocab, token_vocabulary=shared_vocab, left_right=True)
		shared_vocab = proxy_model.token_vocab
		multitask.add_task(args.proxy ,proxy_model, args.proxy_loss, args.proxy_weight, proxy_task=True)				
	elif args.proxy == 'SGV':
		print('\nProxy:', args.proxy)
		context_pos = set(['MD', 'VB', 'VBD','VBG','VBN','VBP','VBZ'])
		proxy_model = SkipGramModel(data['proxy'], window_size=args.proxy_window_size, embedding_dim=args.embedding_dim, word_embeddings=initial_word_embedding_weights, model_dir=args.model_dir, pos_vocabulary=shared_pos_vocab, token_vocabulary=shared_vocab, pos_filter=context_pos)
		shared_vocab = proxy_model.token_vocab
		multitask.add_task(args.proxy ,proxy_model, args.proxy_loss, args.proxy_weight, proxy_task=True)		
	elif args.proxy == 'SGLRV':
		print('\nProxy:', args.proxy)
		context_pos = set(['MD', 'VB', 'VBD','VBG','VBN','VBP','VBZ'])
		proxy_model = SkipGramModel(data['proxy'], window_size=args.proxy_window_size, embedding_dim=args.embedding_dim, word_embeddings=initial_word_embedding_weights, model_dir=args.model_dir, pos_vocabulary=shared_pos_vocab, token_vocabulary=shared_vocab, left_right=True, pos_filter=context_pos)
		shared_vocab = proxy_model.token_vocab
		multitask.add_task(args.proxy ,proxy_model, args.proxy_loss, args.proxy_weight, proxy_task=True)

	elif args.proxy == 'DR':
		print('\nProxy:', args.proxy)
		label_groups = {'DR':['dr_BEFORE','dr_AFTER','dr_OVERLAP', 'dr_BEFORE/OVERLAP'], 'ENTITIES':['etype_EVENT','etype_TIMEX3','etype_SECTIONTIME', 'etype_DOCTIME']} 
		if args.model_type == 'compositional':
			proxy_model = EntityModel(annotated_texts=data['proxy'], unannotated_texts=[], target_label='DR', label_groups=label_groups, context_size=args.context_size, candidate_labels=['etype_EVENT'], word_embeddings=initial_word_embedding_weights, embedding_dim=args.embedding_dim, lstm_dim=50, model_dir=args.model_dir, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
		elif args.model_type == 'sequential':
			proxy_model = EntityLSTMModel(annotated_texts=data['proxy'], unannotated_texts=[], target_label='DR', label_groups=label_groups, context_size=args.context_size, candidate_labels=['etype_EVENT'], word_embeddings=initial_word_embedding_weights, embedding_dim=args.embedding_dim, lstm_dim=50, model_dir=args.model_dir, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)

		shared_vocab = proxy_model.token_vocab
		multitask.add_task(args.proxy ,proxy_model, args.proxy_loss, args.proxy_weight, proxy_task=True)
	

	if args.proxy and args.sharing_type == 'hard':
		shared_embedding_layer = proxy_model.get_word_embedding_layers()[0]

	validation_models = []
	# Set up the Task model
	print('\nTask:', args.task)	
	if args.task == 'DR':
		label_groups = {'DR':['dr_BEFORE','dr_AFTER','dr_OVERLAP', 'dr_BEFORE/OVERLAP'], 'ENTITIES':['etype_EVENT','etype_TIMEX3','etype_SECTIONTIME', 'etype_DOCTIME']} 
		validation_models.append('DR')
		test_labels = [text.span_annotations for text in data['test']]
		if args.model_type == 'compositional':
			task_model = EntityModel(annotated_texts=data['train'], unannotated_texts=[], target_label='DR', label_groups=label_groups, context_size=args.context_size, candidate_labels=['etype_EVENT'], word_embeddings=initial_word_embedding_weights, word_embedding_layer=shared_embedding_layer, embedding_dim=args.embedding_dim, lstm_dim=args.lstm_dim, token_vocabulary=shared_vocab, model_dir=args.model_dir, hidden=hidden, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
		elif args.model_type == 'sequential':
			task_model = EntityLSTMModel(annotated_texts=data['train'], unannotated_texts=[], target_label='DR', label_groups=label_groups, context_size=args.context_size, candidate_labels=['etype_EVENT'], word_embeddings=initial_word_embedding_weights, word_embedding_layer=shared_embedding_layer, embedding_dim=args.embedding_dim, lstm_dim=args.lstm_dim, token_vocabulary=shared_vocab, model_dir=args.model_dir, hidden=hidden, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
		multitask.add_task('DR',task_model, args.task_loss, args.task_weight, main_task=True, training_delay=args.proxy_headstart if args.proxy else 0)

	elif args.task == 'ES':
		label_groups = {'ES':['etype_EVENT', 'OTHER']} 
		validation_models.append('ES')
		candidate_generator = TokenGenerator([1])
		test_labels = [text.span_annotations for text in data['test']]
		if args.model_type == 'compositional':
			task_model = EntityModel(annotated_texts=data['train'], unannotated_texts=[], target_label='ES', label_groups=label_groups, context_size=args.context_size, candidate_labels=[], candidate_generator=candidate_generator, word_embeddings=initial_word_embedding_weights, word_embedding_layer=shared_embedding_layer, embedding_dim=args.embedding_dim, lstm_dim=args.lstm_dim, token_vocabulary=shared_vocab, model_dir=args.model_dir, hidden=hidden, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
		elif args.model_type == 'sequential':
			task_model = EntityLSTMModel(annotated_texts=data['train'], unannotated_texts=[], target_label='ES', label_groups=label_groups, context_size=args.context_size, candidate_labels=[], candidate_generator=candidate_generator, word_embeddings=initial_word_embedding_weights, word_embedding_layer=shared_embedding_layer, embedding_dim=args.embedding_dim, lstm_dim=args.lstm_dim, token_vocabulary=shared_vocab, model_dir=args.model_dir, hidden=hidden, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
		multitask.add_task('ES',task_model, args.task_loss, args.task_weight, main_task=True, training_delay=args.proxy_headstart if args.proxy else 0)

	elif args.task == 'TS':
		label_groups = {'TS':['etype_TIMEX3', 'OTHER']} 
		validation_models.append('TS')
		candidate_generator = TokenGenerator([1,2,3,4,5,6])
		test_labels = [text.span_annotations for text in data['test']]
		if args.model_type == 'compositional':
			task_model = EntityModel(annotated_texts=data['train'], unannotated_texts=[], target_label='TS', label_groups=label_groups, context_size=args.context_size, candidate_labels=[], candidate_generator=candidate_generator, word_embeddings=initial_word_embedding_weights, word_embedding_layer=shared_embedding_layer, embedding_dim=args.embedding_dim, lstm_dim=args.lstm_dim, token_vocabulary=shared_vocab, model_dir=args.model_dir, hidden=hidden, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
		elif args.model_type == 'sequential':
			task_model = EntityLSTMModel(annotated_texts=data['train'], unannotated_texts=[], target_label='TS', label_groups=label_groups, context_size=args.context_size, candidate_labels=[], candidate_generator=candidate_generator, word_embeddings=initial_word_embedding_weights, word_embedding_layer=shared_embedding_layer, embedding_dim=args.embedding_dim, lstm_dim=args.lstm_dim, token_vocabulary=shared_vocab, model_dir=args.model_dir, hidden=hidden, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
		multitask.add_task('TS',task_model, args.task_loss, args.task_weight, main_task=True, training_delay=args.proxy_headstart if args.proxy else 0)

	elif args.task == 'CR':
		print('\nModel Type:', args.model_type)
		validation_models.append('CR')
		test_labels = [text.span_pair_annotations for text in data['test']]
	
		if args.model_type == 'compositional':
			task_model = RelationModel(annotated_texts=data['train'], unannotated_texts=[], target_label='CR', max_center_length=max_center_length, context_size=args.context_size, label_groups={'CR':['CONTAINS', 'OTHER']}, candidate_labels_a1=['etype_EVENT','etype_TIMEX3','etype_SECTIONTIME', 'etype_DOCTIME'], candidate_labels_a2=['etype_EVENT'], word_embedding_layer=shared_embedding_layer, word_embeddings=initial_word_embedding_weights, lstm_dim=args.lstm_dim, embedding_dim=args.embedding_dim, token_vocabulary=shared_vocab, pos_vocabulary=shared_pos_vocab, model_dir=args.model_dir, hidden=hidden, pos=args.pos, within_sentences=args.within_sentences, unidirectional_candidates=args.transform_to_unidirectional, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
		elif args.model_type == 'sequential':
			if args.transform_to_unidirectional == 1:
				task_model = RelationLSTMModel(annotated_texts=data['train'], unannotated_texts=[], target_label='CR', max_center_length=max_center_length, context_size=args.context_size, label_groups={'CR':['CONTAINS','CONTAINS_INVERSE', 'OTHER']}, candidate_labels_a1=['etype_EVENT','etype_TIMEX3','etype_SECTIONTIME', 'etype_DOCTIME'], candidate_labels_a2=['etype_EVENT'], word_embedding_layer=shared_embedding_layer, word_embeddings=initial_word_embedding_weights, lstm_dim=args.lstm_dim, embedding_dim=args.embedding_dim, token_vocabulary=shared_vocab, pos_vocabulary=shared_pos_vocab, model_dir=args.model_dir, hidden=hidden, pos=args.pos, within_sentences=args.within_sentences,unidirectional_candidates=args.transform_to_unidirectional, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)			
			else:
				task_model = RelationLSTMModel(annotated_texts=data['train'], unannotated_texts=[], target_label='CR', max_center_length=max_center_length, context_size=args.context_size, label_groups={'CR':['CONTAINS', 'OTHER']}, candidate_labels_a1=['etype_EVENT','etype_TIMEX3','etype_SECTIONTIME', 'etype_DOCTIME'], candidate_labels_a2=['etype_EVENT'], word_embedding_layer=shared_embedding_layer, word_embeddings=initial_word_embedding_weights, lstm_dim=args.lstm_dim, embedding_dim=args.embedding_dim, token_vocabulary=shared_vocab, pos_vocabulary=shared_pos_vocab, model_dir=args.model_dir, hidden=hidden, pos=args.pos, within_sentences=args.within_sentences,unidirectional_candidates=args.transform_to_unidirectional, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)


		multitask.add_task('CR',task_model, args.task_loss, args.task_weight, main_task=True, training_delay=args.proxy_headstart if args.proxy else 0)

		if args.transform_to_unidirectional == 2:
			inv_embedding_layer=task_model.get_word_embedding_layers()[0]
			inv_pos_embedding_layer=None#task_model.get_word_embedding_layers(pos=True)[0]
			inv_max_center_lenght=20
			inv_within_sentences=1
			shared_vocab = task_model.token_vocab # possibly extended
		
			if args.model_type == 'compositional':
				task_inv_model = RelationModel(annotated_texts=data['train'], unannotated_texts=[], target_label='TLinv', max_center_length=inv_max_center_lenght, context_size=args.context_size, label_groups={'TLinv':['CONTAINS_INVERSE', 'OTHER']}, candidate_labels_a1=['etype_EVENT'], candidate_labels_a2=['etype_EVENT','etype_TIMEX3','etype_SECTIONTIME', 'etype_DOCTIME'], word_embedding_layer=inv_embedding_layer, word_embeddings=initial_word_embedding_weights, pos_embedding_layer=inv_pos_embedding_layer, lstm_dim=args.lstm_dim, embedding_dim=args.embedding_dim, token_vocabulary=shared_vocab, pos_vocabulary=shared_pos_vocab, model_dir=args.model_dir, hidden=hidden, pos=args.pos, within_sentences=inv_within_sentences, unidirectional_candidates=args.transform_to_unidirectional, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
			elif args.model_type == 'sequential':
				task_inv_model = RelationLSTMModel(annotated_texts=data['train'], unannotated_texts=[], target_label='TLinv', max_center_length=inv_max_center_lenght, context_size=args.context_size, label_groups={'TLinv':['CONTAINS_INVERSE', 'OTHER']}, candidate_labels_a1=['etype_EVENT'], candidate_labels_a2=['etype_EVENT','etype_TIMEX3','etype_SECTIONTIME', 'etype_DOCTIME'], word_embedding_layer=inv_embedding_layer, word_embeddings=initial_word_embedding_weights, pos_embedding_layer=inv_pos_embedding_layer, lstm_dim=args.lstm_dim, embedding_dim=args.embedding_dim, token_vocabulary=shared_vocab, pos_vocabulary=shared_pos_vocab, model_dir=args.model_dir, hidden=hidden, pos=args.pos, within_sentences=inv_within_sentences, unidirectional_candidates=args.transform_to_unidirectional, input_dropout=args.input_dropout, bi=args.bi, recurrent_unit=args.recurrent_unit)
			multitask.add_task('TLinv',task_inv_model, args.task_loss, args.task_weight, training_delay=args.proxy_headstart if args.proxy else 0)
	else:
		print('ERROR: No valid task argument! Choose: CR, DR, ES, or TS')
		exit()			
	
	print('\nTask Loss:', args.task_loss)

	# Set up the trainer

	if args.proxy and args.sharing_type == 'WD':
		soft_sharing_layers = proxy_model.get_word_embedding_layers()+task_model.get_word_embedding_layers()
		LayerSimilarityModel = WordEmbeddingDistanceModel(texts=data['proxy'], layers=soft_sharing_layers, token_vocabulary=shared_vocab, distance_metric=args.distance_metric)
		print('\nSoft Sharing Layers:', [layer.name for layer in soft_sharing_layers])
		multitask.add_task('WD', LayerSimilarityModel, args.sharing_loss, args.sharing_weight, training_delay=args.proxy_headstart if args.proxy else 0)
	
	if args.proxy and args.sharing_type == 'WSim':
		soft_sharing_layers = proxy_model.get_word_embedding_layers()+task_model.get_word_embedding_layers()
		LayerSimilarityModel = WordEmbeddingSimilarityObjective(data['proxy'], task_model.get_word_embedding_layers()[0], proxy_model.get_word_embedding_layers()[0], initial_shared_vocab, non_overlapping_words, distance_metric=args.distance_metric)
		print('\nSoft Sharing Layers:', [layer.name for layer in soft_sharing_layers])
		multitask.add_task('WSim', LayerSimilarityModel, args.sharing_loss, args.sharing_weight, training_delay=args.proxy_headstart if args.proxy else 0)

	if args.proxy and args.sharing_type == 'WSimf':
		soft_sharing_layers = proxy_model.get_word_embedding_layers()+task_model.get_word_embedding_layers()
		LayerSimilarityModel = WordEmbeddingSimilarityObjective(data['proxy'], task_model.get_word_embedding_layers()[0], proxy_model.get_word_embedding_layers()[0], initial_shared_vocab, initial_shared_vocab, distance_metric=args.distance_metric, overlapped_sampling=True)
		print('\nSoft Sharing Layers:', [layer.name for layer in soft_sharing_layers])
		multitask.add_task('WSimf', LayerSimilarityModel, args.sharing_loss, args.sharing_weight, training_delay=args.proxy_headstart if args.proxy else 0)

	if args.main_proxy=='PA2':
		main_proxy = LSTMArgument2Predictor(task_model, annotated_texts=data['train'], model_dir=args.model_dir)
		multitask.add_task('PA2', main_proxy, args.proxy_loss, args.main_proxy_weight, proxy_task=True)

	if args.main_proxy == 'RBOW':
		main_proxy = LSTMAutoEncoder(task_model, annotated_texts=data['proxy'], model_dir=args.model_dir)
		multitask.add_task('RBOW', main_proxy, args.proxy_loss, args.main_proxy_weight, proxy_task=True)

	if args.main_proxy == 'RBOV':
		context_pos = set(['MD', 'VB', 'VBD','VBG','VBN','VBP','VBZ'])
		main_proxy = LSTMAutoEncoder(task_model, annotated_texts=data['proxy'], model_dir=args.model_dir, pos_filter=context_pos)
		multitask.add_task('RBOV', main_proxy, args.proxy_loss, args.main_proxy_weight, proxy_task=True)
		
	if args.main_proxy== 'SGA2':
		main_proxy = SkipGramArgumentModel(data['train'], relation_labels = ['CONTAINS'], embedding_dim=args.embedding_dim, word_embedding_layer=shared_embedding_layer, min_context_count=1, model_dir = args.model_dir, token_vocabulary=shared_vocab)
		multitask.add_task('SGA2', main_proxy, args.proxy_loss, args.main_proxy_weight, proxy_task=True)

	if args.main_proxy == 'ESG':
		main_proxy = EventSkipGramModel(data['train'], window_size=args.proxy_window_size, embedding_dim=args.embedding_dim, word_embeddings=initial_word_embedding_weights, model_dir=args.model_dir, token_vocabulary=shared_vocab)
		multitask.add_task('ESG' ,main_proxy, args.proxy_loss, args.proxy_weight, proxy_task=True)


	if args.main_proxy == 'ESGLR':
		main_proxy = EventSkipGramModel(data['train'], window_size=args.proxy_window_size, embedding_dim=args.embedding_dim, word_embeddings=initial_word_embedding_weights, model_dir=args.model_dir, token_vocabulary=shared_vocab, left_right=True)
		multitask.add_task('ESGLR' ,main_proxy, args.proxy_loss, args.proxy_weight, proxy_task=True)


	print('\n'+str(multitask))	
	# training

	
	if args.transform_to_unidirectional == 2:
		validation_models.append('TLinv')
	
	low_memory_models = []
	if args.low_memory_mode:
		low_memory_models = [args.proxy, 'WSimf', 'WSim', 'WD', 'RBOW', 'RBOV']
		
	multitask.train(max_epochs=args.num_epochs, patience=args.patience, validation_texts=data['valid'], validation_models=validation_models, validation_interval=args.validation_interval, batch_size=args.bs, low_memory_models=low_memory_models, fix_embeddings_at=args.fix_embeddings_at, fix_proxy_at=args.fix_proxy_at)
	
	# save the trained model
	save_extraction_model(task_model, args.model_dir + '/' + task_model.name + '_model.p')	
	task_model.write_embeddings_to_files(args.model_dir + '/embs/task/')
	if args.transform_to_unidirectional == 2:
		save_extraction_model(task_inv_model, args.model_dir + '/' + task_inv_model.name + '_model.p')	
		task_inv_model.write_embeddings_to_files(args.model_dir + '/embs/inv_task/')
	if args.proxy:
		proxy_model.write_embeddings_to_files(args.model_dir + '/embs/proxy/')


if args.predict:
	ignore_labels=['OTHER']
	#predict/evaluate test data
	print('\nPredicting:')
	task_model = load_extraction_model(args.model_dir + '/' + args.task + '_model.p')	
	if args.transform_to_unidirectional == 2:
		task_inv_model = load_extraction_model(args.model_dir + '/TLinv_model.p')	

	predictions = task_model.predict(data['test'], ignore_labels=ignore_labels)
	
	evaluation, mistakes = get_evaluation(test_labels,predictions, verbose=True, texts=data['test'], model=task_model)

	
	for label in task_model.target_label_vocab:
		if not label in ignore_labels:
			potentially_harmful, probably_harmless = analyse_mistakes(mistakes, data=data, model=task_model, label=label, file_path=args.model_dir + '/mistakes.'+label.replace('/','_')+'.txt')

	if args.transform_to_unidirectional == 2:	
		predictions2 = task_inv_model.predict(data['test'], ignore_labels=ignore_labels)		
		evaluation_inv, mistakes_inv = get_evaluation(test_labels,predictions2, verbose=True, texts=data['test'], model=task_inv_model)
		potentially_harmful_inv, probably_harmless_inv = analyse_mistakes(mistakes_inv, data=data, model=task_inv_model, label='CONTAINS_INVERSE', file_path=args.model_dir + '/mistakes.CONTAINS_INVERSE.txt')

		for i,pred in enumerate(predictions):
			pred['CONTAINS_INVERSE'] = predictions2[i]['CONTAINS_INVERSE']

		print('\n------------------ BOTH:')
		evaluation, mistakes_inv = get_evaluation(test_labels,predictions, verbose=True, labels = ['CONTAINS', 'CONTAINS_INVERSE'])

	if args.to_xml:
		print('Writing to xml')
		out_texts = []
		for text in data['test']:
			predictions = task_model.predict([text], ignore_labels=['OTHER'])[0]
			if args.transform_to_unidirectional == 2:
				predictions_inv = task_inv_model.predict([text], ignore_labels=['OTHER'])[0]
				predictions['CONTAINS_INVERSE'] = predictions_inv['CONTAINS_INVERSE']
			if args.task == 'CR':				
				text.update_annotations(span_pair_update=predictions)
				if args.transform_to_unidirectional:
					text = transform_to_bidirectonal_relations_text(text)

				
			elif args.task in ['DR', 'ES', 'TS']:
				text.update_annotations(span_update=predictions)
			out_texts.append(text)	

		if args.task in ['DR', 'TS', 'ES']:
			write_texts_to_thyme_anafora_xml(out_texts, pred_dir=args.model_dir + '/out/' + args.task, ignore_relations=True)
		else:
			write_texts_to_thyme_anafora_xml(out_texts, pred_dir=args.model_dir + '/out/' + args.task)
	
	#wordvectors = task_model.get_gensim_word_vectors()
	#selection = ['history', 'colonoscopy', 'patient','.', 'the','a', 'colon', 'cancer','end', 'is', 'december', 'right','week', '14', 'start']
	#selection = ['december', 'march', 'april',  'ct', 'scan', 'start', 'end']
	if args.transform_to_unidirectional == 2:
		harmful = potentially_harmful.union(potentially_harmful_inv)
		harmless = probably_harmless.union(probably_harmless_inv)
	else:
		harmful = potentially_harmful
		harmless = probably_harmless
	selection = [w for w,count in task_model.token_vocab.items()[:100]]
	selection = {'e':['history', 'colonoscopy', 'patient'], 't':['december', 'march', 'april']}
	selection= {'harmful':list(harmful)[:100], 'harmless':list(harmless)[:100]}
	for word_vector_file in glob.glob(args.model_dir + '/embs/*/*.bin'):
		print('\nVisualizing', word_vector_file)
		wordvectors = KeyedVectors.load_word2vec_format(word_vector_file, binary=True)
		visualize_word_vectors(wordvectors,selected_words=selection, to_file=word_vector_file[:-3] + 'png', show=args.plot_embeddings)	
		
		
#	selection = ['history', 'colonoscopy', 'patient','.', 'the','a', 'colon', 'cancer','end', 'is', 'december', 'right','week', '14', 'start']
#	visualize_word_vectors(wordvectors.values()[0],words=selection)	
	
print('\nFinished Experiment (took', int(round(time() - time_0,0)), 'seconds in total).')
