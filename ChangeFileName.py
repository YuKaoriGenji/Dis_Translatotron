import os
path='./wav_transcription_tst/' #changable

f=os.listdir(path)
f.sort()
n=0
nw=0
nt=0
s=str(n+1)
s=s.zfill(5)
for i in f:
    name=f[nw+nt].split('.')
    oldname=path+f[nw+nt]
    if name[1]=='wav':
        s=str(nw+1)
        s=s.zfill(5)
        newname=path+'French_'+s+'.'+name[1]
        nw+=1
        print(name[1],oldname,'------------------->',newname)
        os.rename(oldname,newname)
    else:
        s=str(nt+1)
        s=s.zfill(5)
        newname=path+'French_'+s+'.'+name[1]
        nt+=1
        print(name[1],oldname,'------------------->',newname)
        os.rename(oldname,newname)

