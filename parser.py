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
                   'lc',
                   'oc',
                   'v/o',
                   'v0',
                   'vfx-i',
                   'vfx-p',
                   'vo',
                   'voice',
                   'subtitle',
                   'overlap',
                   'os',
                   'offscreen',
                   'off-screen',
                   'off',
                   'cont',
                   'cont\'d',
                   'con\'t',
                   'contd',
                   'contid',
                   'continued', 'continuing',
                   '0s',
                   '1966',
                   '1974',
                   'viet/subtitle', 'subtitled',
                   'tv',
                   'o/'
                   'hick!', 'nina', 'salmon', 'subtitles',
                   'b\%w', 'etc',
                   '1947', '1949', '1954', '1955', '1965', '1968', '1972', '1973', 'a1', '2', '3', '40', '7/19/2012', '10', 'conttnttfd'
                   ]
STEMMER = SnowballStemmer('english')


def clean_line(line, brackets):
    patt = '\[.*?\]' if brackets else '\(.*?\)'
    matches = re.findall(patt, line)

    for m in matches:
        line = line.replace(m, '')

    line = line.strip()
    line = re.sub('[\r\n_]', '', line)
    character = re.match('([A-Z]{2,}\s?\.?\s?){1,}', line)
    if character:
        line = line[character.end(0):]
    line = line.lstrip('.- \t')
    return line


def pull_adverbs(input_file, output_fp, brackets=True, full_context=False):
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
        for i, line in enumerate(condensed_lines):
            if len(line) <= 1:
                continue

            patt = '\[.*?\]' if brackets else '\(.*?\)'
            matches = re.findall(patt, line)

            for adverb in matches:
                # Clean line
                cleaned_line = clean_line(line, brackets)

                # Clean adverb
                adverb = adverb[1:-1]
                adverb = re.sub('[_\.]', '', adverb)
                adverb = adverb.lower()
                adverb = adverb.replace('\xc3\xaf', 'i')
                adverb = adverb.replace('\xc3\xab', 'e')

                # Line validation
                valid = len(cleaned_line) > 1
                valid &= bool(cleaned_line.split())

                # Adverb validation
                valid &= len(adverb.split()) == 1
                valid &= len(adverb) > 1
                valid &= adverb not in ILLEGAL_ADVERBS

                if valid:
                    counts += 1
                    stemmed_adverb = STEMMER.stem(adverb).encode('utf-8')
                    if full_context:
                        prev_line = clean_line(condensed_lines[i-1], brackets) if i > 0 else ''
                        next_line = clean_line(condensed_lines[i+1], brackets) if i + 1 < len(condensed_lines) else ''
                        output_fp.write('%s\t%s\t%s\t%s\n' % (stemmed_adverb, prev_line, cleaned_line, next_line))
                    else:
                        output_fp.write('%s\t%s\n' % (stemmed_adverb, cleaned_line))
                    break
        if counts > 0:
            print input_file, counts


def edit_movie(input_file, output_file):
    out = open(output_file, 'w')

    with open(input_file, 'r') as infile:
        for line in infile:
            stripped_line = line.strip().replace('\t', ' ')
            if stripped_line.isupper():
                out.write('\r\n\n')
                out.write(stripped_line)
            else:
                out.write(' ')
                out.write(stripped_line)
            
    out.close()

if __name__ == '__main__':
    with open('adverbs.txt', 'w') as adverbs_fp:
        for author in os.listdir('glibs'):
            for play in os.listdir(os.path.join('glibs', author)):
                input_filename = join('glibs', author, play)
                if play.endswith('paren'):
                    pull_adverbs(input_filename, adverbs_fp, brackets=False, full_context=False)
                else:
                    pull_adverbs(input_filename, adverbs_fp, full_context=False)

        for file in os.listdir('dialogs-edited'):
            os.remove(join('dialogs-edited', file))
        for genre in os.listdir('dialogs'):
            for movie in os.listdir(os.path.join('dialogs', genre)):
                input_filename = join('dialogs', genre, movie)
                output_filename = join('dialogs-edited', movie)
                if movie in os.listdir('dialogs-edited'):
                    continue
                edit_movie(input_filename, output_filename)
                pull_adverbs(output_filename, adverbs_fp, brackets=False, full_context=False)
                












