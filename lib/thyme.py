from __future__ import print_function
from data import Text
import xml.etree.ElementTree as ET 
from xml.dom import minidom
import glob, re, shutil, os,io

def read_thyme_documents(folder, regex='.*Temp.*', max_documents=1000000, closure=['CONTAINS','BEFORE'], lowercase=False, ctakes_out_dir=False, pos=True, less_strict=False, verbose=False, conflate_digits=True):
		if verbose:
			print('\nReading THYME documents from', folder)
		documents = []
		for i,subfolder_path in enumerate(glob.glob(folder + '/*')):
				subfolder_name = subfolder_path.split('/')[-1]

				text_file, annotations_file = None, None
				for file_path in glob.glob(subfolder_path + '/*'):
					file_name = file_path.split('/')[-1]
					if file_name == subfolder_name:
						text_file = file_path
					if re.search(regex, file_name):
						annotations_file = file_path
				if not text_file and less_strict:
						text_file=subfolder_path 
				if (text_file and annotations_file) or (less_strict and text_file):	
					if verbose:
						print(text_file, annotations_file)
					with io.open(text_file, 'r',encoding="UTF8") as fin:
						text = fin.read()
					if annotations_file:
						id = annotations_file.split('/')[-1].split('.')[0]
						span_annotations, span_pair_annotations = read_thyme_anafora_xml(text_file, annotations_file, verbose=verbose)
					elif less_strict:
						id = None
						span_annotations, span_pair_annotations = {}, {}
					
					doc = Text(text, span_annotations, span_pair_annotations, id, pos=pos, transitive_closure=closure, lowercase=lowercase, conflate_digits=conflate_digits)
					if 'etype_EVENT' in doc.span_annotations and verbose:
						print('EVENT:', len(doc.span_annotations['etype_EVENT']))
					if 'etype_TIMEX3' in doc.span_annotations and verbose:
						print('TIMEX3:', len(doc.span_annotations['etype_TIMEX3']))
					if 'CONTAINS' in doc.span_pair_annotations and verbose:
						print('CONTAINS:', len(doc.span_pair_annotations['CONTAINS']))
					if verbose:		
						print('POS:', len(doc.pos))
					documents.append(doc)
					if max_documents and i >= max_documents:
						return documents
				else:
					if verbose:
						print('warning: no annotations or text for',subfolder_name, '(therefore skipped)')
		return documents



def write_texts_to_thyme_anafora_xml(texts, pred_dir= 'out/', ignore_relations=False):
	print('\nWriting anafora xml to', pred_dir)
	if os.path.exists(pred_dir):
		shutil.rmtree(pred_dir)
	os.makedirs(pred_dir)	

	for text in texts:
		print(text.id)
		target_dir = pred_dir + '/' + text.id
		target_txt_file = pred_dir + '/' + text.id + '/' + text.id 
		target_xml_file = pred_dir + '/' + text.id + '/' + text.id + '.Temporal-Relation.system.completed.xml'
		os.makedirs(target_dir)
		
		# writing text file
		with io.open(target_txt_file, 'w',encoding="UTF8") as f:
			f.write(text.text)
			
		# constructing xml
		doc_xml = ET.Element('data')
		doc_xml_annotations = ET.SubElement(doc_xml,'annotations')	
		entity_counter=0
		span_to_id = {}
		if 'etype_EVENT' in text.span_annotations:
			for event_span in text.span_annotations['etype_EVENT']:
				entity_counter+=1
				labels = {l.split('_')[0]:l.split('_')[1] for l in text.reverse_span_annotations[event_span]}

				entity = ET.SubElement(doc_xml_annotations, 'entity')
				e_id, e_span, e_type, e_parentsType, e_props  = ET.SubElement(entity, 'id'), ET.SubElement(entity, 'span'), ET.SubElement(entity, 'type'),ET.SubElement(entity, 'parentsType'),ET.SubElement(entity, 'properties')
				e_id.text, e_type.text, e_span.text, e_parentsType.text = str(entity_counter)+'@e@'+text.id+'@system', labels['etype'], str(event_span[0])+','+str(event_span[1]), 'TemporalEntities'
				span_to_id[event_span] = e_id.text
				# event properties	
				e_dctr, e_Type, e_degree, e_polarity, e_modality, e_aspect, e_permanence = ET.SubElement(e_props, 'DocTimeRel'), ET.SubElement(e_props, 'Type'), ET.SubElement(e_props, 'Degree'), ET.SubElement(e_props, 'Polarity'), ET.SubElement(e_props, 'ContextualModality'), ET.SubElement(e_props, 'ContextualAspect'), ET.SubElement(e_props, 'Permanence')
				for label_type in ['dr', 'subtype', 'degree', 'polarity', 'modality', 'aspect', 'permanence']:
					if not label_type in labels:
						labels[label_type]='NONE'
				e_dctr.text, e_Type.text, e_degree.text, e_polarity.text, e_modality.text, e_aspect.text, e_permanence.text = labels['dr'], labels['subtype'], labels['degree'], labels['polarity'], labels['modality'], labels['aspect'], labels['permanence']				

				
		if 'etype_TIMEX3' in text.span_annotations:
			for event_span in text.span_annotations['etype_TIMEX3']:
				entity_counter+=1
				labels = {l.split('_')[0]:l.split('_')[1] for l in text.reverse_span_annotations[event_span]}
				entity = ET.SubElement(doc_xml_annotations, 'entity')
				e_id, e_span, e_type, e_parentsType, e_props  = ET.SubElement(entity, 'id'), ET.SubElement(entity, 'span'), ET.SubElement(entity, 'type'),ET.SubElement(entity, 'parentsType'),ET.SubElement(entity, 'properties')
				e_id.text, e_type.text, e_span.text, e_parentsType.text = str(entity_counter)+'@e@'+text.id+'@system', labels['etype'], str(event_span[0])+','+str(event_span[1]), 'TemporalEntities'
				span_to_id[event_span] = e_id.text
				e_class = ET.SubElement(e_props, 'Class')
				for label_type in ['eclass']:
					if not label_type in labels:
						labels[label_type]='NONE'
				e_class.text = labels['eclass']

		if 'etype_SECTIONTIME' in text.span_annotations:
			for event_span in text.span_annotations['etype_SECTIONTIME']:
				entity_counter+=1
				labels = {l.split('_')[0]:l.split('_')[1] for l in text.reverse_span_annotations[event_span]}
				entity = ET.SubElement(doc_xml_annotations, 'entity')
				e_id, e_span, e_type, e_parentsType, e_props  = ET.SubElement(entity, 'id'), ET.SubElement(entity, 'span'), ET.SubElement(entity, 'type'),ET.SubElement(entity, 'parentsType'),ET.SubElement(entity, 'properties')
				e_id.text, e_type.text, e_span.text, e_parentsType.text = str(entity_counter)+'@e@'+text.id+'@system', labels['etype'], str(event_span[0])+','+str(event_span[1]), 'TemporalEntities'
				span_to_id[event_span] = e_id.text
				
		if 'etype_DOCTIME' in text.span_annotations:
			for event_span in text.span_annotations['etype_DOCTIME']:
				entity_counter+=1
				labels = {l.split('_')[0]:l.split('_')[1] for l in text.reverse_span_annotations[event_span]}
				entity = ET.SubElement(doc_xml_annotations, 'entity')
				e_id, e_span, e_type, e_parentsType, e_props  = ET.SubElement(entity, 'id'), ET.SubElement(entity, 'span'), ET.SubElement(entity, 'type'),ET.SubElement(entity, 'parentsType'),ET.SubElement(entity, 'properties')
				e_id.text, e_type.text, e_span.text, e_parentsType.text = str(entity_counter)+'@e@'+text.id+'@system', labels['etype'], str(event_span[0])+','+str(event_span[1]), 'TemporalEntities'
				span_to_id[event_span] = e_id.text
		
		if not ignore_relations:	
			rel_counter = 0
			for rel_label in text.span_pair_annotations:
				for span_pair in  text.span_pair_annotations[rel_label]:
					rel_counter += 1
					span_a1, span_a2 = span_pair
					rel = ET.SubElement(doc_xml_annotations, 'relation')
					r_id, r_type, r_parentsType, r_props = ET.SubElement(rel, 'id'), ET.SubElement(rel, 'type'), ET.SubElement(rel, 'parentsType'), ET.SubElement(rel, 'properties')
					r_id.text , r_parentsType.text = str(rel_counter)+'@r@'+text.id+'@system', 'TemporalRelations'
				
					if rel_label in ['CONTINUES','TERMINATES', 'INITIATES']:
						r_type.text = 'ALINK'
					else:
						r_type.text = 'TLINK'
				
					r_source, r_Type, r_target = ET.SubElement(r_props, 'Source'), ET.SubElement(r_props, 'Type'), ET.SubElement(r_props, 'Target')
					r_source.text, r_Type.text, r_target.text = span_to_id[span_a1], rel_label, span_to_id[span_a2]
				

		doc_xml_string = minidom.parseString(ET.tostring(doc_xml).replace('\n','').replace('\t', '')).toprettyxml(indent = "\t", newl='\n\n')
		with io.open(target_xml_file, 'w',encoding="UTF8") as f:
			f.write(doc_xml_string)

def read_thyme_anafora_xml(text, anafora_xml_file, verbose=False):
		try:
			tree = ET.parse(anafora_xml_file)
		except:
			print('ERROR: could not read', anafora_xml_file, ' due to an xml syntax error')
			return {}, {}
		
		root = tree.getroot()
		span_annotations = {}
		span_pair_annotations = {}
		entity_id_to_span = {}
		
		# Reading Entities
		for e in root.iter('entity'):
			e_type, e_id, span, doctimerel, e_subtype, e_degree, e_polarity, e_ContextualModality,e_Class, e_Aspect,e_Permanence = None, None, None, None, None, None, None, None, None, None, None
			for child in e.getchildren():
				if child.tag == 'id':
					e_id = child.text
				if child.tag == 'span':
					spans = [(int(s1),int(s2)) for (s1,s2) in  [s.split(',') for s in child.text.split(';')]]
					span=spans[0]
					entity_id_to_span[e_id] = span

				if child.tag == 'type':
					e_type = child.text
				if child.tag == 'properties':
					
					for doctime_child in child.iter('DocTimeRel'):
						doctimerel = doctime_child.text
					for e_subtype_child in child.iter('Type'):
						e_subtype = e_subtype_child.text
					for e_degree_child in child.iter('Degree'):
						e_degree = e_degree_child.text
					for e_polarity_child in child.iter('Polarity'):
						e_polarity = e_polarity_child.text
					for e_ContextualModality_child in child.iter('ContextualModality'):
						e_ContextualModality = e_ContextualModality_child.text						
					for e_Aspect_child in child.iter('ContextualAspect'):
						e_Aspect = e_Aspect_child.text	
					for e_Class_child in child.iter('Class'):
						e_Class = e_Class_child.text	
					for e_Permanence_child in child.iter('Permanence'):
						e_Permanence = e_Permanence_child.text

			for label_type, label in [('etype',e_type), ('dr',doctimerel), ('subtype',e_subtype), ('degree',e_degree), ('polarity',e_polarity), ('modality',e_ContextualModality), ('eclass',e_Class), ('aspect',e_Aspect), ('permanence',e_Permanence)]:
				if label:
					lab = label_type + '_' + label
					if not lab in span_annotations:
						span_annotations[lab] = []
					span_annotations[lab].append(span)
			
		# Reading Relations
		for r in root.iter('relation'):
			source, target, relation = None, None, None
			for child in r.getchildren():
				if child.tag == 'properties':
					for properties_child in child:
						if properties_child.tag == 'Source':
							if properties_child.text in entity_id_to_span:
								source = entity_id_to_span[properties_child.text]
						if properties_child.tag == 'Target':
							if properties_child.text in entity_id_to_span:
								target = entity_id_to_span[properties_child.text]
						if properties_child.tag == 'Type':
							relation = properties_child.text
			if not relation in span_pair_annotations:
				span_pair_annotations[relation] = []
			span_pair_annotations[relation].append((source,target))

		if verbose:
			print('=====')
			e_labs = sorted([(l, len(span_annotations[l])) for l in span_annotations],key=lambda (x,y):y, reverse=True)
			for label,num in e_labs:
				print('E', label, num)
			print('-----')
		
			ee_labs = sorted([(l, len(span_pair_annotations[l])) for l in span_pair_annotations],key=lambda (x,y):y, reverse=True)
			
			for label,num in ee_labs:
				print('R', label, num)

		return span_annotations, span_pair_annotations	
		
