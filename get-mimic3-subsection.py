# Gets a subsection of MIMICIII notes by their ROW ID
import csv, sys, os, shutil

row_ids_txt = 'mimic3-notes.txt'
csv_file = sys.argv[1]
out_folder = 'mimic3-subsection-out'
out_info_file = 'mimic3-subsection-info.csv'

if os.path.exists(out_folder):
	shutil.rmtree(out_folder)
os.makedirs(out_folder)	

print 'reading', row_ids_txt
with open(row_ids_txt, 'r') as f:
	row_ids = set()
	for line in f.readlines():
		row_id = line.strip()
		row_ids.add(row_id)

print 'reading',csv_file
info_file = open(out_info_file, 'w')
with open(csv_file, 'rb') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	for row in csv_reader:
		ROW_ID, SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, STORETIME, CATEGORY, DESCRIPTION, CGID, ISERROR, TEXT = row
		if ROW_ID in row_ids:
			with open(out_folder + '/' + ROW_ID ,'w') as txtout:
				print 'writing', out_folder + '/' + ROW_ID
				txtout.write(TEXT)
				info_file.write(','.join([ROW_ID, SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, STORETIME, CATEGORY, DESCRIPTION, CGID, ISERROR]) + '\n')
				row_ids.remove(ROW_ID)
				

		if len(row_ids) == 0:
			break


info_file.close()	

