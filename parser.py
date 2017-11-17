
import re
import sys, os


def pull_adverbs(f, o, useBracks):

	sdelim = '['
	edelim = ']'
	if not useBracks:
		sdelim = '('
		edelim = ')'

	all_lines = f.readlines()
	condensed_lines = []
	prev_line = ""
	for line in all_lines:
		if line in ['\n', '\r\n']:

			condensed_lines.append(prev_line)
			prev_line = ""
		else:
			prev_line += str(' ' + line.strip())

	counts = 0
	for line in condensed_lines:
		if len(line) <= 1:
			continue

		# l = line
		# l =  line.strip()
		# l = l.replace('\r', '').replace('\n', ' ')

		patt = '\[.*?\]' if useBracks else '\(.*?\)'
		matches = re.findall(patt, line)

		if matches and (len(matches[0].split()) == 1  ):
			counts += 1
			o.write(str( '[' + matches[0][1:-1] + ']' + " " + line.strip().replace('\r', '').replace('\n', '').replace( matches[0], '') + "\n"))
		
		# if sdelim in l and l.index(sdelim) > 0:
		# 	full_line =  line[l.index(sdelim) -1:]
		# 	words = full_line.split()

		# 	if not words:
		# 		continue



		# 	if words[0].startswith(sdelim) and words[0].endswith(edelim):
		# 		o.write('[' + str(words[0][1:-1] + ']' + " " + full_line[full_line.index(edelim)+1:].replace('\r', '').replace('\n', '')) + "\n")
		# 		# o.write(str(words[0] + " " + l +"\n"))
		# 		counts += 1

	print f, counts
	f.close()

def count_adverbs(contents):
    patt = '\[(.*?)\]'
    matches = re.findall(patt, contents)
    c = 0
    for m in matches:
        if len(m.split()) == 1:
            c += 1
    return c


if __name__ == '__main__':

	adverbs = open('adverbs.txt', 'w')

	for author in os.listdir('glibs'):
		for play in os.listdir(os.path.join('glibs', author )):
			bracks = True
			if (play.endswith('paren')):
				bracks = False

			pin = os.path.join('glibs', author, play)
			f = open(pin)
			pull_adverbs(f, adverbs, bracks)


	adverbs.close()

    # if len(sys.argv) != 2:
    #     print('Usage: python check_adverbs.py <file>')
    #     sys.exit(0)

    # all_contents = ''
    # for line in open(sys.argv[1], 'r'):
    #     all_contents += line
    # print(count_adverbs(all_contents))

	# earnest = open('iobe.txt')
	# earnest_out = open('iobe-adverbs.txt', 'w')
	# earnest_speakers = ['Jack', 'Lady Bracknell', 'Algernon', 'Cecily', 'Gwendolen', 'Miss Prism', 'Lane', 'Merriman', 'Chasuble']


	# lwf = open('lwf.txt')
	# lwf_out = open('lwf-adverbs.txt', 'w')
	# lwf_speakers = ['Lord Windermere','Mr. George Alexander','Lord Darlington','Mr. Nutcombe Gould','Lord Augustus Lorton','Mr. H. H. Vincent','Mr. Cecil Graham','Mr. Ben Webster','Mr. Dumby','Mr. Vane-Tempest','Mr. Hopper','Mr. Alfred Holles','Parker','Mr. V. Sansbury','Lady Windermere','Miss Lily Hanbury','The Duchess of Berwick','Miss Fanny Coleman','Lady Agatha Carlisle','Miss Laura Graves','Lady Plymdale','Miss Granville','Lady Jedburgh','Miss B. Page','Lady Stutfield','Miss Madge Girdlestone','Mrs. Cowper-Cowper','Miss A. de Winton','Mrs. Erlynne','Miss Marion Terry','Rosalie','Miss Winifred Dolan']


	# pyg = open('pyg.txt')
	# pyg_out =  open('pyg-adverbs.txt', 'w')
	# pyg_speakers = ['THE MOTHER', 'THE DAUGHTER', 'LIZA', 'DOOLITTLE', 'FREDDY', 'THE FLOWER GIRL', 'THE GENTLEMAN', 'THE BYSTANDER', 'THE NOTETAKER', 'THE SARCASTIC ONE']
	
	# pull_adverbs(earnest, earnest_out, earnest_speakers)

	# pull_adverbs(lwf, lwf_out, lwf_speakers)

	# pull_adverbs(pyg, pyg_out, pyg_speakers)
