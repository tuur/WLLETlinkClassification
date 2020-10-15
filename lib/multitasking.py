from __future__ import print_function
from keras.models import Model
from keras.utils import plot_model
from time import time
import numpy as np
import random, pickle
import collections
import matplotlib.pyplot as plt
import losses, keras
from evaluation import get_evaluation
keras.losses.modified_hinge = losses.modified_hinge
keras.losses.minimize_output = losses.minimize_output
keras.losses.label_entropy = losses.label_entropy
keras.losses.modified_hinge2 = losses.modified_hinge2
keras.losses.binary_crossentropy = losses.binary_crossentropy
keras.losses.minimize_output2 = losses.minimize_output2
keras.losses.maximize_output = losses.maximize_output
keras.losses.mixed_loss = losses.mixed_loss
random.seed(1)


from evaluation import calculate_combined_inv_fmeasure, calculate_thresholded_inv_fmeasure


class MultitaskTrainer(object):
	
	
	def __init__(self, optimizer='adam', mode='joint', soft_sharing_layers=None, model_dir='.'):
		self.models = {}
		self.losses = {}
		self.loss_weights = {}
		self.mode = mode
		self.optimizer=optimizer
		self.soft_sharing_layers = None
		self.model_dir = model_dir
		self.main_task = None
		self.proxy_tasks = set([])
		self.trainer_model = None
		self.training_delays = {0:[model for model in self.models]}
		self.started = []

		self.indices_per_label = {}

	def __str__(self):
		return 'Multitask: '+str(self.loss_weights)

	def add_task(self, task_name, task_model, task_loss, task_weight, main_task=False, proxy_task=False, training_delay=0):
		self.losses.update({task_name:task_loss})
		self.models.update({task_name:task_model})
		self.loss_weights.update({task_name:task_weight})
		if not training_delay in self.training_delays:
			self.training_delays[training_delay] = []
		self.training_delays[training_delay].append(task_name)
		if proxy_task:
			self.proxy_tasks.add(task_name)
		
		if main_task:
			if self.main_task:
				print('WARNING: changing main task', self.main_task, 'to', main_task, '!')
			self.main_task = task_name
			
			
	def build_training_model(self, epoch_num=0):
		inputs, outputs = [], []
		loss_functions,loss_weights = {},{}
		loss_dict = {'label_entropy':keras.losses.label_entropy, 'mixed_loss':keras.losses.mixed_loss, 'maximize_output':keras.losses.maximize_output, 'minimize_output':keras.losses.minimize_output, 'minimize_output2': keras.losses.minimize_output2, 'hinge':keras.losses.hinge, 'modified_hinge':keras.losses.modified_hinge, 'categorical_crossentropy':keras.losses.categorical_crossentropy, 'modified_hinge2':keras.losses.modified_hinge2, 'binary_crossentropy':keras.losses.binary_crossentropy}
		if epoch_num in self.training_delays:
			self.started += self.training_delays[epoch_num]
		
		print('\nActive models', self.started)
		for model_name, model in self.models.items():
			model.model.compile(optimizer=self.optimizer, loss=self.losses[model_name])
			inputs += model.model.inputs
			outputs.append(model.model.output)
			output_layer_name = model.model.output.name.split('/')[0]
			loss_functions[output_layer_name] = self.losses[model_name]

			if model_name in self.started:
				loss_weights[output_layer_name] = self.loss_weights[model_name]				
			else:
				loss_weights[output_layer_name] = 0.0

			
		trainer_model =	Model(inputs=inputs, outputs=outputs, name='trainer_model')
		plot_model(trainer_model, self.model_dir + '/' + 'multitask.png', show_shapes=True)
		trainer_model.compile(optimizer=self.optimizer, loss={model:loss_dict[loss_string] for model,loss_string in loss_functions.items()}, loss_weights=loss_weights)
		return trainer_model


	def shuffle_data(self, X,Y):
		for (model_name, model) in self.models.items():
			input_names = [str(inp.name.split(':')[0]) for inp in model.model.inputs]
			output_name = model.model.output.name.split('/')[0]
			index_array = np.arange(X[input_names[0]].shape[0])
			
			np.random.shuffle(index_array)
			for input_name in input_names:
				X[input_name] = X[input_name][index_array]

			Y[output_name] = Y[output_name][index_array]
		return X,Y

	def get_next_batch(self, X, Y, batch_size=32, batch_counter = 0, main_model_name = 'TL'):
		x, y, used_batch_size = {},{}, batch_size 
		input_names = [str(inp.name.split(':')[0]) for (model_name, model) in self.models.items() for inp in model.model.inputs]

		max_batch_size = min([X[input_name].shape[0] for input_name in input_names])
		if batch_size > max_batch_size:
			print('WARNING: batch size', batch_size, 'too big, using', max_batch_size, 'instead!')
			used_batch_size = max_batch_size - 1
		
		for (model_name, model) in self.models.items():
			output_name = model.model.output.name.split('/')[0]
			input_names = [str(inp.name.split(':')[0]) for inp in model.model.inputs]

			dataset_size = X[input_names[0]].shape[0]
			if dataset_size==0:
				return 0,0, None
			start = (batch_counter * batch_size) % dataset_size
			end = ((batch_counter * batch_size) + used_batch_size) % dataset_size
			if model_name == main_model_name:
				epoch_index = int((batch_counter * batch_size) / dataset_size)

			if end < start:
				selected_indices = range(start, dataset_size)
				selected_indices += range(used_batch_size - len(selected_indices))
			else:
				selected_indices = range(start,end)
			
			for input_name in input_names:

				x[input_name] = X[input_name][selected_indices,:]

			y[output_name] = Y[output_name][selected_indices,:]	
		
		
		return x,y, epoch_index

	def get_max_batch(self, X, Y):
		x, y = {},{} 
		input_names = [str(inp.name.split(':')[0]) for (model_name, model) in self.models.items() for inp in model.model.inputs]
		max_batch_size = min([X[input_name].shape[0] for input_name in input_names])
		for (model_name, model) in self.models.items():
			output_name = model.model.output.name.split('/')[0]
			input_names = [str(inp.name.split(':')[0]) for inp in model.model.inputs]
			selected_indices = range(max_batch_size)
			for input_name in input_names:

				x[input_name] = X[input_name][selected_indices,:]

			y[output_name] = Y[output_name][selected_indices,:]			
		return x, y, max_batch_size

	
	def train(self, patience=10, validation_texts=[], validation_models=[], batch_size=32,validation_interval=1, verbose=False, low_memory_models=[], max_epochs=10, num_docs_low_memory=2, fix_embeddings_at=None, fix_proxy_at=None):
				
		print('\nPreprocessing:')
		preproc_time = time()
		saved_weights=False
		X, Y = {}, {}
		for model_name, model in self.models.items():
			if not model_name in low_memory_models:
				X_m, Y_m = model.preproc_X(model.annotated_texts, labeled=True), model.preproc_Y(model.annotated_texts, labeled=True)				
				X.update(X_m)
				Y.update(Y_m)
			else:
				selected_text = model.annotated_texts[0 % len(model.annotated_texts)]
				X_lmm, Y_lmm = model.preproc_X([selected_text], labeled=True), model.preproc_Y([selected_text], labeled=True)
				X.update(X_lmm)
				Y.update(Y_lmm)	

		# Data preprocessing [VALIDATION DATA]
		if validation_texts:
			
			Xv, Yv = {}, {}
			for model_name, model in self.models.items():
				Xv_m, Yv_m = model.preproc_X(validation_texts, labeled=True), model.preproc_Y(validation_texts, labeled=True)
				Xv.update(Xv_m)
				Yv.update(Yv_m)
			xv, yv, val_size = self.get_max_batch(Xv,Yv)
		
		print('finished, took', int(round(time()-preproc_time,0)), 'seconds')
		
		print('\nTraining delays', self.training_delays)
		self.trainer_model = self.build_training_model()
		
		print('\nFixing all word embeddings at epoch', fix_embeddings_at)

		if fix_proxy_at:
			print('\nFixing proxy weights at epoch', fix_proxy_at)

		fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(10, 10), dpi=300)
		metrics, metric_iters, metric_names, metric_names_training = [], [], self.trainer_model.metrics_names + ['inv_f1'], self.trainer_model.metrics_names
		val_metrics,val_metric_iters, min_val_loss = [], [], 1000000000
		patience_counters, min_val_loss = {i:patience for i in range(len(metric_names))}, {i:100000 for i in range(len(metric_names))}


		main_task = self.main_task
		main_model_inputs  = [str(inp.name.split(':')[0]) for inp in self.models[main_task].model.inputs]
		dataset_size = X[main_model_inputs[0]].shape[0]
		if len(validation_models)==0:
			validation_models=[main_task]
			
		iters = max_epochs * ((dataset_size / batch_size)+1)
		time_0, go, epoch_num, new_epoch_num, best_overall_threshold, after_number_of_batches, embeddings_fixed, proxy_fixed = time(), True, 0, 0, None, None, False, False

		# ================================ TRAINING ================================

		print('\nTraining ( task_data_size:',dataset_size, 'batch_size:', batch_size, 'val_size:', val_size, 'max_num_epochs:', max_epochs, 'patience:', patience, ')\n')
		for i in range(1,iters+1):
			if not go:
				break
			time_i, printer = time(), 'b-' +str(i) + ' ('+ str((batch_size*i)/1000) + 'k)'
			time_prof = time()

			# preprocessing a text from models that should be treated with low memory
			for lmm_model_name in low_memory_models:
				if lmm_model_name in self.models and lmm_model_name in self.started:
					lmm_model = self.models[lmm_model_name]

					selected_texts = lmm_model.annotated_texts[(num_docs_low_memory*i) % len(lmm_model.annotated_texts):(num_docs_low_memory*i+num_docs_low_memory)% len(lmm_model.annotated_texts)] # takes the num_docs_low_memory next texts
					X_lmm, Y_lmm = lmm_model.preproc_X(selected_texts, labeled=True), lmm_model.preproc_Y(selected_texts, labeled=True)
					X.update(X_lmm)
					Y.update(Y_lmm)		
			if verbose:			
				print('I preproc LMMM', time()-time_prof)
				time_prof = time()
			
			# get balanced batch
			if verbose:
				print('getting batch')

			x, y, new_epoch_num = self.get_next_batch(X, Y, batch_size=batch_size, batch_counter=i, main_model_name=main_task)		
			
			
			if new_epoch_num == None:
				print('no data present... skipped epoch')
				continue

			
			if fix_embeddings_at == epoch_num and not embeddings_fixed:
				print('>>>Fixing all word embeddings<<<\n')
				for model_name, model in self.models.items():
					for embedding_layer in model.get_word_embedding_layers():
						embedding_layer.trainable = False
				embeddings_fixed=True

			if fix_proxy_at == epoch_num and not proxy_fixed:
				print('>>>Fixing proxy weights<<<\n')
				for model_name, model in self.models.items():
					if model_name in self.proxy_tasks:
						for layer in model.model.layers:
							layer.trainable = False
						self.started.remove(model_name)
				proxy_fixed=True
			
			
			time_prof = time()
			
			#  train on batch 
			if verbose:
				print('updating')
			batch_metrics = self.trainer_model.train_on_batch(x, y) 
			if verbose:
				print('IV (trainonbatch)', time()-time_prof)
				time_prof = time()

			if not isinstance(batch_metrics, collections.Iterable):
				batch_metrics=[batch_metrics]

			metrics.append(batch_metrics)
			metric_iters.append(i)
			for metric_index, metric_name in enumerate(metric_names_training):
				printer += '\t' + metric_name[:5]  + ': ' + str(round(batch_metrics[metric_index],6)).ljust(8)


			
			total_time_spend = time() - time_0
			time_spend_i = time() - time_i
			estimated_time = (iters - i) * (float(total_time_spend) / i)
			printer += '\tt '+str(time_spend_i)[:4]+' s\tETA '+str(int(round(estimated_time/60,0)))+' m'
			print(printer)	

			if  new_epoch_num != epoch_num:
				time_prof = time()
				printer = 'VALIDATION:'
				print('\n--- EPOCH', epoch_num, '---')
				if verbose:
					print('shuffling data')
				X, Y = self.shuffle_data(X, Y)
				if verbose:
					print('V (shuffle)', time()-time_prof)
					time_prof = time()
				if validation_texts and epoch_num%validation_interval == 0:
					if verbose:
						print('calculating validation metrics')
						
					current_validation_metrics = self.trainer_model.evaluate(xv,yv, verbose=0)
					if verbose:
						print('VI (eval_metrics)', time()-time_prof)
						time_prof = time()	
					
					val_preds = {model_name:self.models[model_name].model.predict(xv) for model_name in validation_models}
					if verbose:
						print('VII (eval preds)', time()-time_prof)
						time_prof = time()	

			
					if self.models[main_task].prediction_threshold:

						inv_fmeasure_val, best_threshold = calculate_thresholded_inv_fmeasure(val_preds, yv, self.models[main_task], thresholds=[.5])
						printer += '\tbest thr:' + str(best_overall_threshold) 
						
					else:
						
						inv_fmeasure_val = calculate_combined_inv_fmeasure(val_preds, yv, [self.models[model_name] for model_name in validation_models])
					if verbose:		
						print('VII (inv f)', time()-time_prof)
						time_prof = time()
					
					if not isinstance(current_validation_metrics, collections.Iterable):
						current_validation_metrics = [current_validation_metrics]
					current_validation_metrics += [inv_fmeasure_val]	
					for metric_index, metric_name in enumerate(metric_names):
						metric_value = current_validation_metrics[metric_index]
						printer += '\tval_' + metric_name[:6] + ': ' + str(round(metric_value,4)).ljust(6)
					
						if len(val_metrics) > 0 and min_val_loss[metric_index] < metric_value and self.main_task in self.started:
							printer += ' (+' + str(round(metric_value-min_val_loss[metric_index],3)) + ')'
							patience_counters[metric_index]-= 1
							printer += '*'
						elif self.main_task in self.started:
							patience_counters[metric_index] = patience
							printer += ' -'	
							min_val_loss[metric_index] = metric_value
						
					val_metrics.append(current_validation_metrics)
					val_metric_iters.append(i)
				
				
				if  patience_counters[metric_names.index('inv_f1')] == patience:
					if self.models[main_task].prediction_threshold:
						best_overall_threshold = best_threshold
					after_number_of_batches = i
					if verbose:
						print('saving checkpoint')
					self.trainer_model.save_weights(self.model_dir + '/weights_checkpoint.h5')
					saved_weights = epoch_num
				
					
				if  patience_counters[metric_names.index('inv_f1')] < 1:
					printer += '\nno patience left...'
					printer += 'chose model from epoch ' + str(saved_weights)
					go = False	

				if epoch_num!=new_epoch_num and new_epoch_num in self.training_delays:
					self.trainer_model = self.build_training_model(epoch_num=new_epoch_num)

				if verbose:
					print('VIII (saving checkpoint and printing)', time()-time_prof)
					time_prof = time()	
				
				epoch_num = new_epoch_num	
				print(printer + '\n')	

		print('finished, took', int(round(time() - time_0,0)), 'seconds')
		print('\nWriting loss plot')
		for metric_index, metric_name in enumerate(metric_names_training):
			ax.plot(metric_iters, [metric_step[metric_index] for metric_step in metrics], label=metric_name)
			ax.legend(loc=1)	
		for metric_index, metric_name in enumerate(metric_names):
			ax.plot(val_metric_iters, [metric_step[metric_index] for metric_step in val_metrics], label='val_'+metric_name, linestyle='--')
			ax.legend(loc=1)	
			
		if after_number_of_batches:
			ax.axvline(after_number_of_batches, color='k', linestyle='--')
		ax.grid()	
		fig.savefig(self.model_dir + '/loss.png')   # save the figure to file
		
		with open(self.model_dir + 'loss_trajectory.p', 'wb') as f:
			pickle.dump([metric_iters, metrics, val_metric_iters, val_metrics, metric_names, metric_names], f)

		if best_overall_threshold:
			self.models[main_task].prediction_threshold = best_overall_threshold
			print('best prediction threshold (used):', best_overall_threshold)
		if saved_weights and not go:
			print('loading model from checkpoint',saved_weights)
			self.trainer_model.load_weights(self.model_dir + '/weights_checkpoint.h5')
