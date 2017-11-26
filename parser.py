import re
import os
from os.path import join
from nltk.stem import SnowballStemmer


ILLEGAL_ADVERBS = ['trademark/copyright',
                   'wwwgutenbergnet',
                   'wwwgutenbergorg',
                   '801',
                   '^y',
                   'rc',
                   'lc']
STEMMER = SnowballStemmer('english')


def pull_adverbs(input_file, output_fp, brackets=True):
    with open(input_file) as f:
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

            patt = '\[.*?\]' if brackets else '\(.*?\)'
            matches = re.findall(patt, line)

            for m in matches:
                line = line.replace(m, '')

            for adverb in matches:
                # Clean line
                line = line.strip()
                line = re.sub('[\r\n_]', '', line)
                character = re.match('([A-Z]{2,}\s?\.?\s?){1,}', line)
                if character:
                    line = line[character.end(0):]
                line = line.lstrip('.- ')

                # Clean adverb
                adverb = adverb[1:-1]
                adverb = re.sub('[_\.]', '', adverb)
                adverb = adverb.lower()
                adverb = adverb.replace('\xc3\xaf', 'i')
                adverb = adverb.replace('\xc3\xab', 'e')

                # Line validation
                valid = len(line) > 1
                valid &= bool(line.split())

                # Adverb validation
                valid &= len(adverb.split()) == 1
                valid &= len(adverb) > 1
                valid &= adverb not in ILLEGAL_ADVERBS

                if valid:
                    counts += 1
                    stemmed_adverb = STEMMER.stem(adverb)
                    output_fp.write('%s\t%s\n' % (stemmed_adverb.encode('utf-8'), line))
                    break

        print input_file, counts


if __name__ == '__main__':
    with open('adverbs.txt', 'w') as adverbs_fp:
        for author in os.listdir('glibs'):
            for play in os.listdir(os.path.join('glibs', author)):
                input_filename = join('glibs', author, play)
                if play.endswith('paren'):
                    pull_adverbs(input_filename, adverbs_fp, brackets=False)
                else:
                    pull_adverbs(input_filename, adverbs_fp)
