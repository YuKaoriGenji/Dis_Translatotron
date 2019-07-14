import os
import re
import ReadingData

fr_path='./WCE-SLT-LIG/TXT/SRC/src-ref-all.fr'
en_path='./WCE-SLT-LIG/TXT/TGT/tgt-ref-all.en'
sent_path='./sentences.txt'

en_fd=open(en_path,'r')
fr_fd=open(fr_path,'r')

sent_fd=open(sent_path,'a')

#en_lines=en_fd.readlines()
#fr_lines=fr_fd.readlines()

#print(fr_lines)
'''
for (num,value) in enumerate(fr_fd):
    print("num:",num,"value:",value)
fr_fd.seek(0)
for (num,value) in enumerate(fr_fd):
    print("num:",num)
'''
n=1
number=0
for tran_text in ReadingData.tran_texts:
    for num,value in enumerate(fr_fd):
        if tran_text in value:
            number=num
            #print(n,':',num)
            #print('# Origin:',tran_text)
            #print('# Found:', value)
            break
    for num,value in enumerate(en_fd):
        if num==number:
            print(n,':',value)
            sent_fd.write(value)
    n+=1
    fr_fd.seek(0)
    en_fd.seek(0)
en_fd.close()
fr_fd.close()
sent_fd.close()
