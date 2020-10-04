

def prepro(filename):
    print(filename)
    fin = open(filename, 'r', encoding='utf-8')
    fout = open(filename.split('.')[0]+'.pair', 'w', encoding='utf-8')
    triplet_cnt = 0
    ap_cnt = 0
    op_cnt = 0
    instance_cnt = 0
    ap_overlap_instance_cnt = 0
    op_overlap_instance_cnt = 0
    ap_overlap_triplet_cnt = 0
    op_overlap_triplet_cnt = 0
    for line in fin:
        instance_cnt += 1
        text, ap, op = line.strip().split('####')
        fout.write(text+'\n')
        ap_seq = [item.split('=')[1] for item in ap.split()]+['O']
        op_seq = [item.split('=')[1] for item in op.split()]+['O']
        
        ap_chunks = []
        op_chunks = []

        ap_start_idx, ap_end_idx = -1, -1
        prev_ap_tag = 'O'
        for i, ap_tag in enumerate(ap_seq):
            if ap_tag != 'O':
                if prev_ap_tag == 'O':
                    ap_start_idx = i
                prev_ap_tag = ap_tag
            elif prev_ap_tag != 'O':
                ap_end_idx = i-1
                ap_chunk = (ap_start_idx, ap_end_idx)
                ap_level = len(prev_ap_tag.split('-')[0])
                ap_polarity = prev_ap_tag.split('-')[1]
                ap_chunks.append([ap_level, ap_chunk, ap_polarity])
                ap_start_idx, ap_end_idx = -1, -1
                prev_ap_tag = 'O'

        op_start_idx, op_end_idx = -1, -1
        prev_op_tag = 'O'
        for i, op_tag in enumerate(op_seq):
            if op_tag != 'O':
                if prev_op_tag == 'O':
                    op_start_idx = i
                prev_op_tag = op_tag
            elif prev_op_tag != 'O':
                op_end_idx = i-1
                op_chunk = (op_start_idx, op_end_idx)
                op_level = len(prev_op_tag)
                op_chunks.append([op_level, op_chunk])
                op_start_idx, op_end_idx = -1, -1
                prev_op_tag = 'O'

        triplets = []
        ap_overlap = 0
        op_overlap = 0
        for ap_chunk in ap_chunks:
            ap_level = ap_chunk[0]
            found = 0
            while ap_level != 0:
                for op_chunk in op_chunks:
                    if op_chunk[0] == ap_level:
                        found += 1
                        triplets.append(str([ap_chunk[1], op_chunk[1], ap_chunk[2]]))
                if found:
                    if found > 1:
                        ap_overlap += 1
                    break
                else:
                    op_overlap += 1
                    ap_level -= 1
        fout.write(';'.join(triplets)+'\n')
        triplet_cnt += len(triplets)
        ap_cnt += len(ap_chunks)
        op_cnt += len(op_chunks) 
        if ap_overlap: 
            ap_overlap_instance_cnt += 1
            ap_overlap_triplet_cnt += ap_overlap
        if op_overlap: 
            op_overlap_instance_cnt += 1
            op_overlap_triplet_cnt += op_overlap
    print(f'total instance count: {instance_cnt}')
    print(f'aspect overlap instance count: {ap_overlap_instance_cnt}')
    print(f'opinion overlap instance count: {op_overlap_instance_cnt}')
    print(f'aspect overlap triplet count: {ap_overlap_triplet_cnt}')
    print(f'opinion overlap triplet count: {op_overlap_triplet_cnt}')
    print(f'aspect count: {ap_cnt}')
    print(f'opinion count: {op_cnt}')
    print(f'triplet count: {triplet_cnt}')
    fin.close()
    fout.close()

if __name__ == '__main__':
    prepro('14lap/train.txt')
    prepro('14lap/test.txt')
    prepro('14lap/valid.txt')
    print('################################')
    prepro('14rest/train.txt')
    prepro('14rest/test.txt')
    prepro('14rest/valid.txt')
    print('################################')
    prepro('15rest/train.txt')
    prepro('15rest/test.txt')
    prepro('15rest/valid.txt')
    print('################################')
    prepro('16rest/train.txt')
    prepro('16rest/test.txt')
    prepro('16rest/valid.txt')


'''
################################
14rest/train.txt
total instance count: 1300
aspect overlap instance count: 247
opinion overlap instance count: 187
aspect overlap triplet count: 276
opinion overlap triplet count: 302
aspect count: 2077
opinion count: 2145
triplet count: 2409
14rest/test.txt
total instance count: 496
aspect overlap instance count: 116
opinion overlap instance count: 77
aspect overlap triplet count: 151
opinion overlap triplet count: 238
aspect count: 849
opinion count: 862
triplet count: 1014
14rest/valid.txt
total instance count: 323
aspect overlap instance count: 50
opinion overlap instance count: 42
aspect overlap triplet count: 58
opinion overlap triplet count: 89
aspect count: 530
opinion count: 524
triplet count: 590
################################
15rest/train.txt
total instance count: 593
aspect overlap instance count: 114
opinion overlap instance count: 37
aspect overlap triplet count: 119
opinion overlap triplet count: 70
aspect count: 834
opinion count: 923
triplet count: 977
15rest/test.txt
total instance count: 318
aspect overlap instance count: 46
opinion overlap instance count: 22
aspect overlap triplet count: 47
opinion overlap triplet count: 24
aspect count: 426
opinion count: 455
triplet count: 479
15rest/valid.txt
total instance count: 148
aspect overlap instance count: 30
opinion overlap instance count: 12
aspect overlap triplet count: 33
opinion overlap triplet count: 29
aspect count: 225
opinion count: 238
triplet count: 260
################################
16rest/train.txt
total instance count: 842
aspect overlap instance count: 151
opinion overlap instance count: 57
aspect overlap triplet count: 157
opinion overlap triplet count: 99
aspect count: 1183
opinion count: 1289
triplet count: 1370
16rest/test.txt
total instance count: 320
aspect overlap instance count: 54
opinion overlap instance count: 23
aspect overlap triplet count: 56
opinion overlap triplet count: 64
aspect count: 444
opinion count: 465
triplet count: 507
16rest/valid.txt
total instance count: 210
aspect overlap instance count: 38
opinion overlap instance count: 14
aspect overlap triplet count: 41
opinion overlap triplet count: 20
aspect count: 291
opinion count: 316
triplet count: 334
################################
14lap/train.txt
total instance count: 920
aspect overlap instance count: 133
opinion overlap instance count: 130
aspect overlap triplet count: 151
opinion overlap triplet count: 214
aspect count: 1283
opinion count: 1265
triplet count: 1451
14lap/test.txt
total instance count: 339
aspect overlap instance count: 59
opinion overlap instance count: 44
aspect overlap triplet count: 64
opinion overlap triplet count: 76
aspect count: 475
opinion count: 490
triplet count: 552
14lap/valid.txt
total instance count: 228
aspect overlap instance count: 49
opinion overlap instance count: 31
aspect overlap triplet count: 56
opinion overlap triplet count: 45
aspect count: 317
opinion count: 337
triplet count: 380
'''