# -*- coding: utf-8 -*-

def to2bio(tags):
    new_tags = []
    prev_pos = '$'
    for i, tag in enumerate(tags):
        cur_tag = tag[0] # using initial character
        if cur_tag == 'O':
            new_tags.append('O'+cur_tag[1:])
            cur_pos = 'O'
        else:
            # current tag is subjective tag, i.e., cur_pos is T
            cur_pos = cur_tag
            if cur_pos == prev_pos:
                # prev_pos is T
                new_tags.append('I'+tag[1:])
            else:
                # prev_pos is O
                new_tags.append('B'+tag[1:])
        prev_pos = cur_pos
    return new_tags

def bio2bieos(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1][0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B', 'S'))
        elif tag[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1][0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I', 'E'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def bieos2span(tags, tp='-AP'): # tp = '', '-AP', or '-OP'
    spans = []
    beg, end = -1, -1
    for i, tag in enumerate(tags):
        if tag == 'S'+tp:
            # start position and end position are kept same for the singleton
            spans.append((i, i))
        elif tag == 'B'+tp:
            beg = i
        elif tag == 'E'+tp:
            end = i
            if end > beg:
                # only valid chunk is acceptable
                spans.append((beg, end))
    return spans

def find_span_with_end(pos, text, tags, tp='-AP'):
    if tags[pos] in ('B'+tp, 'O'+tp):
        span = [pos] * 2
    else:
        temp_span = []
        temp_span.append(pos)
        found_beg = 0
        while tags[pos] == 'I'+tp:
            pos -= 1
            if pos < 0:
                found_beg = 1
                temp_span.append(pos)
                break
            if tags[pos] == 'B'+tp:
                found_beg = 1
                temp_span.append(pos)
                break
        if not found_beg: temp_span.append(pos)
        span = list(reversed(temp_span))
    return span