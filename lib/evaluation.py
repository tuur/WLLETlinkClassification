from __future__ import print_function
import numpy as np

# EVALUTION BASED ON SPAN AND SPAN PAIR PREDICTIONS

def get_evaluation(test_reference, predictions, texts=None, model=None, per_text=False, verbose=True, labels=None):
	print('\nEvaluating:')
	summed_metrics = {}
	mistakes = {}
	if not labels:
		labels = set([label for i in range(len(predictions)) for label in predictions[i]])
	for i in range(len(predictions)):
		
		if texts and model:
			text_metrics, mistakes[i] = get_text_metrics(test_reference[i], predictions[i], text=texts[i], model=model)
		else:
			text_metrics, _ = get_text_metrics(test_reference[i], predictions[i])
			

		for label in labels:
			if not label in summed_metrics:
				summed_metrics[label] = {}
			for k,v in text_metrics[label].items():
				if not k in summed_metrics[label]:
					summed_metrics[label][k] = 0
				summed_metrics[label][k] += len(v) if type(v) == list else v
	metrics = {}
	for label in labels:
		precision = get_precision(summed_metrics[label]['tp'], summed_metrics[label]['fp'])
		recall = get_recall(summed_metrics[label]['tp'], summed_metrics[label]['fn'])
		fmeasure = get_fmeasure(summed_metrics[label]['tp'], summed_metrics[label]['fp'], summed_metrics[label]['fn'])
		if verbose:
			print ('P',str(round(precision,3)).ljust(5), '\tR',str(round(recall,3)).ljust(5), '\tF',str(round(fmeasure,3)).ljust(5), '\t', label)

		metrics[label] = {'tp':summed_metrics[label]['tp'], 'fp':summed_metrics[label]['fp'], 'fn':summed_metrics[label]['fn'], 'precision':precision, 'recall':recall, 'fmeasure':fmeasure}
			

	TP = sum([summed_metrics[label]['tp'] for label in labels])
	FP = sum([summed_metrics[label]['fp'] for label in labels])
	FN = sum([summed_metrics[label]['fn'] for label in labels])
	PRECISION, RECALL, FMEASURE = get_precision(TP, FP), get_recall(TP, FN), get_fmeasure(TP, FP, FN)
	metrics['TOTAL'] = {'precision': PRECISION, 'recall':RECALL, 'fmeasure': FMEASURE,'tp':TP, 'fp':FP, 'fn':FN}
	

			
	if verbose:
		print('---')
		print ('P',str(round(PRECISION,3)).ljust(5), '\tR',str(round(RECALL,3)).ljust(5), '\tF',str(round(FMEASURE,3)).ljust(5), '\t<TOTAL>')
		print ('TP',TP, '\tFP',FP, '\tFN',FN, '\t<TOTAL>')
				
		
	return metrics, mistakes


def get_text_metrics(true_labels, pred_labels, text=None, model=None):
	text_metrics, mistakes = {}, {}
	
	for label in true_labels.keys() + pred_labels.keys():
		if not label in true_labels:
			true_labels[label] = []
		if not label in pred_labels:
			pred_labels[label]= []
			
		label_metrics = get_metrics_from_raw_predictions(true_labels[label],pred_labels[label])
		text_metrics[label] = label_metrics
		
		if text and model:
			mistakes[label] = {}
			mistakes[label]['fp'] = [(span_pair, model.preproc_candidate_x(span_pair, text, labeled=True)) for span_pair in label_metrics['fp']]
			mistakes[label]['fn'] = [(span_pair, model.preproc_candidate_x(span_pair, text, labeled=True)) for span_pair in label_metrics['fn']]
			mistakes[label]['tp'] = [(span_pair, model.preproc_candidate_x(span_pair, text, labeled=True)) for span_pair in label_metrics['tp']]
		
	return text_metrics, mistakes
	
def get_metrics_from_raw_predictions(true, pred):
	tp = get_tp(pred, true)
	fp = get_fp(pred, true)
	fn = get_fn(pred, true)

	precision = get_precision(len(tp),len(fp))
	recall = get_recall(len(tp), len(fn))
	fmeasure = get_fmeasure(len(tp), len(fp), len(fn))	
	return {'tp':tp, 'fp':fp, 'fn':fn, 'precision':precision, 'recall':recall, 'fmeasure':fmeasure}

def get_tp(pred,true):
	return [span for span in pred if span in true]

def get_fp(pred,true):
	return [span for span in pred if not span in true]

def get_fn(pred, true):
	return [span for span in true if not span in pred]

def get_precision(num_tp, num_fp):
	if num_tp + num_fp ==0:
		return 0.0
	return float(num_tp) / (num_tp + num_fp)
	
def get_recall(num_tp, num_fn):
	if num_tp + num_fn ==0:
		return 0.0
	return float(num_tp) / (num_tp + num_fn)

	
def get_fmeasure(num_tp, num_fp, num_fn, beta=1.0):
	precision = get_precision(num_tp, num_fp)
	recall = get_recall(num_tp, num_fn)
	if precision * recall == 0:
		return 0.0
	else:
		return (1.0+ beta*beta) * ((precision * recall) / (beta*beta 	* precision + recall))
	

def get_confusion():
	pass



def calculate_combined_inv_fmeasure(preds, true, models):
	TP, FP, FN = 0, 0, 0 
	for model in models:
		val_preds = preds[model.name]
		evaluation = calculate_fmeasure(val_preds, true, model)
		TP += evaluation['tp']
		FP += evaluation['fp']
		FN += evaluation['fn']
		print('eval', model.name, evaluation['fmeasure'])
	return 1.0 - get_fmeasure(TP, FP, FN)	


def calculate_fmeasure(preds, true, model):
	evaluation ={'tp':0, 'fp':0, 'fn':0} 
	for i,pred_vec in enumerate(preds):
		
		out_name = model.get_layer_name('output')
		true_vec = true[out_name][i]
		pred_label = model.predict_with_argmax(pred_vec)
		true_label =  model.target_label_vocab_reverse[np.argmax(true_vec)]
		correct = pred_label == true_label
		
		if correct and true_label == 'OTHER': # ignore OTHER labels
			pass
		elif correct and true_label != 'OTHER' :
			evaluation['tp']+=1
		elif not correct and true_label=='OTHER':
			evaluation['fp']+=1
		elif not correct and true_label!='OTHER':
			evaluation['fn']+=1
			if len(model.target_label_vocab) > 2 :
				evaluation['fp']+=1	
	fmeasure = 	get_fmeasure(evaluation['tp'], evaluation['fp'], evaluation['fn'])	
	evaluation.update({'fmeasure':fmeasure})
	return evaluation

def calculate_thresholded_inv_fmeasure(preds, true, model, thresholds=[0.4, 0.5, 0.6]):	
	
	max_f, best_threshold = 0, 0.5
	for threshold_value in thresholds:
		evaluation ={'tp':0, 'fp':0, 'fn':0} 
		for i,pred_vec in enumerate(preds):
		
			out_name = model.get_layer_name('output')
			true_vec = true[out_name][i]
			true_label =  model.target_label_vocab_reverse[np.argmax(true_vec)]


			# prediction with threshold
			pred_label = model.predict_with_threshold(pred_vec, threshold_value)
			
			correct = pred_label == true_label

			if  correct and true_label == 'OTHER': # ignore OTHER labels
				pass
			elif correct and true_label != 'OTHER' :
				evaluation['tp']+=1
			elif not correct and true_label=='OTHER':
				evaluation['fp']+=1
			elif not correct and true_label!='OTHER':
				evaluation['fn']+=1
				if len(model.target_label_vocab) > 2 :
					evaluation['fp']+=1	
						
		fmeasure = 	get_fmeasure(evaluation['tp'], evaluation['fp'], evaluation['fn'])	
		if fmeasure > max_f:
			max_f = fmeasure
			best_threshold = threshold_value
			
	return 1.0 - max_f, best_threshold

def analyse_mistakes(mistakes, data, model, label, file_path=None):
	texts = data['test']
	counts_per_word = {category:{} for category in ['tp', 'fp', 'fn']}

	word_freqs = {dataset:{} for dataset in data}
	for dataset in data:
		for vocab in [text.vocabulary for text in data[dataset]]:
			for w in vocab:
				if not w in word_freqs[dataset]:
					word_freqs[dataset][w] = 0
				word_freqs[dataset][w] += len(vocab[w])
		
	error_analysis = 'ERROR ANALYSIS\n'
	
	potentially_harmful = set([w for w in word_freqs['proxy'] if not w in word_freqs['train']])
	harmless = set([])
	
	for text_index in mistakes:
		error_analysis += '\n' + str(texts[text_index].id) + ' ' + str({category:len(mistakes[text_index][label][category]) for category in mistakes[text_index][label]}) + '\n'

		for category in mistakes[text_index][label]:
			for example_span_pair, example_input in mistakes[text_index][label][category]:
				if type(example_span_pair[0]) != tuple:
					error_analysis += '\n\n>>>' + category + ' ' + str(category) + ' ' + str(texts[text_index].id) + ' ' + str(texts[text_index].span_to_tokens(example_span_pair)) + '\n'					
				else:	
					error_analysis += '\n\n>>>' + category + ' ' + str(category) + ' ' + str(texts[text_index].id) + ' ' + str(texts[text_index].span_to_tokens(example_span_pair[0])) + ' ' + str(texts[text_index].span_to_tokens(example_span_pair[1])) + '\n'
					
			

				for input_type, word_indices in example_input.items():
					if not input_type in counts_per_word[category] and not 'pos' in input_type:
						counts_per_word[category][input_type] = {}
				
					if not 'pos' in input_type:
						word_indices = word_indices[0]
						words = [model.reverse_token_vocab[w_index] for w_index in word_indices if not w_index == model.token_vocab[model.padding_token]]
						word_strings = [w + '_' + str(word_freqs['train'][w]) if w in word_freqs['train'] else (w if '>' == w[-1]  else (w + '_' + str(word_freqs['proxy'][w]) + '*' if w in word_freqs['proxy'] else w + '_unk')) for w in words]

						error_analysis += str(input_type) + ':\t' + ' '.join([w if w !='\n' else '<newline>' for w in word_strings])
						for word in words:
							if not word in counts_per_word[category][input_type]:
								counts_per_word[category][input_type][word] = 0
							counts_per_word[category][input_type][word] += float(1.0 / len(mistakes[text_index][label][category]))
	
	if file_path:
		with open(file_path, 'w') as f:
			f.write(error_analysis)
	else:
		print(error_analysis)
					
	for category in [ 'tp']:
		print('\nCATEGORY', category)
		for input_type in counts_per_word[category]:
			print(input_type, '\t', sorted(counts_per_word[category][input_type].items(), key=lambda(x,y):y, reverse=True)[:10])
			for w in counts_per_word[category][input_type]:
				if w in potentially_harmful:
					potentially_harmful.remove(w)
				else:
					harmless.add(w)


	for category in [ 'fp', 'fn']:
		print('\nCATEGORY', category)
		for input_type in counts_per_word[category]:
			print(input_type, '\t', sorted(counts_per_word[category][input_type].items(), key=lambda(x,y):y, reverse=True)[:10])
			
	return potentially_harmful, harmless

		
