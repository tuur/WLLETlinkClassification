from __future__ import print_function


class TokenGenerator(object):
	
	def __init__(self, lengths=[1]):
		self.lengths = lengths
		
	def generate_candidates(self, text):
		candidates = []
		for length in self.lengths:
			for i in range(0, len(text.spans) - length + 1):
				
				start = text.spans[i][0]
				end = text.spans[i + length - 1][-1]
				candidates.append((start, end))
		return candidates		


class SpanGenerator(object):
	
	
	def __init__(self, generator_labels):
		# span labels used to generate candidates
		self.generator_labels = generator_labels
	
	def generate_candidates(self, text):
		candidates = []
		for generator_label in self.generator_labels:
			if generator_label in text.span_annotations:
				candidates += text.span_annotations[generator_label]
		return candidates
		
class SpanPairGenerator(object):
	
	def __init__(self, generator_labels_a1, generator_labels_a2, max_token_distance=30, within_sentences=True, within_paragraphs=True, left_to_right=False):
		# span labels used to generate candidate arguments
		self.candidate_generator_a1 = SpanGenerator(generator_labels_a1)
		self.candidate_generator_a2 = SpanGenerator(generator_labels_a2)
		self.max_token_distance=max_token_distance
		self.within_sentences=within_sentences
		self.within_paragraphs=within_paragraphs
		self.left_to_right=left_to_right
		
	def generate_candidates(self, text):
		
		candidates = []
		for candidate_span_a1 in self.candidate_generator_a1.generate_candidates(text):
			for candidate_span_a2 in self.candidate_generator_a2.generate_candidates(text):	
				if self.left_to_right and candidate_span_a1[0] > candidate_span_a2[0]:
					continue

				
				if not(self.max_token_distance >= text.token_distance(candidate_span_a1, candidate_span_a2) > 0):
					continue
				if self.within_sentences and not(text.spans_lie_within_one_sentence(candidate_span_a1, candidate_span_a2)):
					continue
				if self.within_paragraphs and not(text.spans_lie_within_one_paragraph(candidate_span_a1, candidate_span_a2)):
					continue
				candidates.append((candidate_span_a1,candidate_span_a2))				
		return candidates	
		
