import re
import os


def pull_adverbs(pin, o, useBracks):
    with open(pin) as f:
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

            patt = '\[.*?\]' if useBracks else '\(.*?\)'
            matches = re.findall(patt, line)

            for match in matches:
                line = line.replace(match, '')
                line = line.strip()
                line = re.sub('[\r\n]', '', line)
                match = re.sub('[_\.]', '', match)
                match = match.lower()
                if len(match.split()) == 1 and len(match) > 3:
                    counts += 1
                    o.write('[%s]\t%s\n' % (match[1:-1], line))

        print pin, counts


if __name__ == '__main__':

    with open('adverbs.txt', 'w') as adverbs:
        for author in os.listdir('glibs'):
            for play in os.listdir(os.path.join('glibs', author)):
                bracks = True
                if (play.endswith('paren')):
                    bracks = False

                pin = os.path.join('glibs', author, play)
                pull_adverbs(pin, adverbs, bracks)
