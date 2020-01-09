#usage:liuqi@node01[interHetNumGet] time python3 dreepConta_v5.py > log_v5
import sys,os,shutil,glob,re
import pandas as pd
from multiprocessing import Pool
from functools import reduce
from Bio import SeqIO
from collections import Counter
from collections import OrderedDict

conta32=pd.read_table("/home/liuqi/work/4generations/result/ssp/dreep_ssp_3class/sampleDistributePlot/contaDel32",sep=r"\s+",usecols=[1],header=None)

conta_fam=[1,3,4,7,13]
conta_sample=["11A_GGM_BL","16B_GM_BL","30C_MO_BL","32B_GM_BL","32D_DA_BS"]

phy=pd.read_table("/home/liuqi/work/4generations/result/mtDNAtree.build13.REF.txt",sep="\t",header=None,index_col=None,names=['index','haplo','mut'])
def getNum(s):
 #if math.isnan(s)!=True:
 if isinstance(s,str)==True:
  s=s.split(" ")
  num=len(s)
 else:
  num=0;s=""
 return s,num
#phy[['mutList','num']]=phy.apply((lambda s:getNum(s['mut'])),axis=1)#getNum(s['mut']) should have same columns as [['mutList','num']]
phy['mutList'],phy['num']=zip(*phy['mut'].map(getNum))
phy=phy.sort_values(by=['num'],axis=0)#0 index 1=columns
phy['haplo_index']=phy['haplo']+"_"+phy['index'].astype(str)

d_pos_haplo={}
for index,row in phy.iterrows():
 haplo=row['haplo'];mut_list=row['mutList'];index_number=row['index'];haplo_ix=haplo+"_"+str(index_number)
 if len(d_pos_haplo.keys())==0:
  d_pos_haplo={k:[haplo_ix] for k in mut_list}
 else:
  for pos in mut_list:
   if pos not in d_pos_haplo.keys():
    d_pos_haplo[pos]=[haplo_ix]
   else:
    d_pos_haplo[pos].append(haplo_ix)
#d_pos_haplo:
#{'73T': ['U3b0_1', 'U3b0_3'], '73': ['U3b2_3123', "U4'9_3125", 'U3b1b_3122'], '150': ['U3b2_3123', 'U3b1b_3122'], '99': ['U3b0_3'], '195': ["U4'9_3125"], '263': ["U4'9_3125", 'U3b1b_3122'], '750': ['U3b1b_3122'], '1438': ['U3b1b_3122']}

l_haplo=[]
for haplo_list in d_pos_haplo.values():
 l_haplo.extend(haplo_list)
haplo_pos_num=dict(Counter(l_haplo))
haplo_pos_num=dict(sorted(haplo_pos_num.items(),key=lambda kv:kv[1],reverse=True))
#print(haplo_pos_num)
haplo_pos_num_sort=OrderedDict(sorted(haplo_pos_num.items(),key=lambda kv:kv[1],reverse=True))
#print(haplo_pos_num_sort)

rCRS=SeqIO.read("/home/liuqi/work/reference/mito_38.fa","fasta")
print(len(rCRS.seq))
#drcrs=dict(zip(list(range(1,16570)),rCRS.seq))

lPosDelete=list(range(302,317))+list(range(513,527))+list(range(566,574))+list(range(16181,16195))+[3106,3107,16519]
enddir="/home/liuqi/work/4generations/result/ssp/dreep_ssp_3class/allHet"

def phyHetGet(pos,ref,alt):#make transation not have alt: 73A/G=73 73A/T=73T
 if ref=='A' or ref=='G':
  if alt != 'T' and alt != 'C':
   sub=str(pos)
  else:
   sub=str(pos)+alt
 else:
  if alt != 'A' and alt != 'G':
   sub=str(pos)
  else:
   sub=str(pos)+alt
 return(sub)

def proAllHet(fil):
 pro_method=fil.split("/")[-1].split(".")[0]#aftDel_emp
 df=pd.read_table(fil,sep="\t",header=None,index_col=None,\
 usecols=[0,1,9,10,11,12,13,14,17,18,19],names=['sample','pos','cov','maf','maa','mac','mia','mic','maff','mafr','amia'])
 df=df[df['pos'].map(lambda x:x not in lPosDelete)]
 #df[0]=df[0].str.replace("_pois.log","")
 df_tmp=df['sample'].str.split("_",expand=True)
 df['sample']=df_tmp[0]+"_"+df_tmp[1]+"_"+df_tmp[2]
 df.index=df['sample']
 df.index.names=['index']#change index name to avoid warning
 df['het']=df['pos'].astype(str)+df['mia']
 df['Het']=df['sample']+":"+df['het']
 d={pro_method:df}
 df['phy']=df.apply(lambda row:phyHetGet(row['pos'],rCRS[row['pos']-1],row['mia']),axis=1)
 return pro_method,df
 #df1=pd.DataFrame(df.groupby('sample')['het'].apply("|".join))
 #df1.rename(columns=({'het':pro_method})) 

allhets=glob.glob("/home/liuqi/work/4generations/result/ssp/dreep_ssp_3class/allHet/*.allhet")
p=Pool(9)
multi_res=[p.apply_async(proAllHet,args=(fil,)) for fil in allhets]
#print(type(multi_res[0]))#<class 'multiprocessing.pool.ApplyResult'>
#dict_all=[dict(zip(res.get()[0],res.get()[1])) for res in multi_res]#List Comprehensions
dict_all={k.get()[0]:k.get()[1] for k in multi_res}#dictionary comprehension

#print(dict_all)
#print(dict_all[0].keys())
def interOrUninEmpPois(df1,df2):
 inter12=pd.merge(df1,df2,on=['sample','het'],how='inner',suffixes=('_emp', '_pois'))#intersect
 #union12=pd.Index(s1).union(pd.Index(s2))#index
 union12=pd.concat([df1[['sample','het','phy']],df2[['sample','het','phy']]],ignore_index=True).drop_duplicates().reset_index(drop=True)
 return inter12,union12

def inter(df1,df2):
 inter12=pd.merge(df1[['sample','het']],df2[['sample','het']],how='inner')#intersect
 return inter12
def union(df1,df2):
 union12=pd.concat([df1[['sample','het']],df2[['sample','het']]],ignore_index=True).drop_duplicates().reset_index(drop=True)
 return union12

def mergeEmpPois(syb):
 inter,union=interOrUninEmpPois(dict_all[syb+'_emp'],dict_all[syb+'_pois'])
 print(syb,inter.shape[0],union.shape[0])#emp pois merge in inter or union
 return inter,union

syb_mem=[];d5type={}#merge the aftDel blbl bsbs blbs bsbl 's emp and pois with inter and union
for pro_type in dict_all:#aftDel_emp is pro_type
 name=pro_type.split("_");syb=name[0]#aftDel
 if syb not in syb_mem:
  d5type[syb]={}
  d5type[syb]['inter'],d5type[syb]['union']=mergeEmpPois(syb)
  syb_mem.append(syb)

print("All","inter",d5type['aftDel']['inter'].shape[0],"union",d5type['aftDel']['union'].shape[0])
print("Same","inter",d5type['blbl']['inter'].shape[0]+d5type['bsbs']['inter'].shape[0],\
	     "union",d5type['blbl']['union'].shape[0]+d5type['bsbs']['union'].shape[0])
print("Diff","inter",d5type['blbs']['inter'].shape[0]+d5type['bsbl']['inter'].shape[0],\
	     "union",d5type['blbs']['union'].shape[0]+d5type['bsbl']['union'].shape[0])

all_inter=d5type['aftDel']['inter'].copy();all_inter.index=all_inter['sample']

def addFam(name):#8D_DA_BL
 l_name=name.split("_");gener=l_name[1];tisu=l_name[2]
 fam=int(re.findall(r'\d+',l_name[0])[0])#8
 if gener=="GGM":
  y=4
 elif gener=="GM":
  y=3
 elif gener=="MO":
  y=2
 elif gener=="DA":
  y=1
 return fam,gener,tisu,y

all_inter['fam'],all_inter['gener'],all_inter['tisu'],all_inter['y']=zip(*all_inter.apply(lambda row:addFam(row['sample']),axis=1))

all_inter=all_inter[~all_inter['sample'].isin(conta32[1])]#del conta 32
all_inter.index.names=['index']
#print(all_inter)
all_inter=all_inter[all_inter['fam'].map(lambda x:x not in conta_fam)].reset_index()
all_inter=all_inter[all_inter['sample'].map(lambda x:x not in conta_sample)].reset_index()
all_inter.drop(["level_0","index"],axis=1,inplace=True)
sample_have_het=all_inter['sample'].unique().size
print("******%d samples in %d families have %d het in %d pos"%(sample_have_het,all_inter['fam'].unique().size,all_inter.shape[0],all_inter['pos_emp'].unique().size))
all_inter=all_inter.sort_values(['fam','pos_emp','y'],ascending=[True,True,False])
all_inter = all_inter[all_inter.columns.drop(list(all_inter.filter(regex='_pois')))]
all_inter.to_csv("InterHetPos_del44_mafcpr_0315.xls",sep="\t",index=False,header=True,float_format="%g")
sample_het_num=pd.DataFrame(all_inter['sample'].value_counts())
sample_het_num.to_csv("AllSampleHetNum130_mafcpr_0315.xls",sep="\t",index=True,header=False)

#all_inter['pos_emp']=all_inter['pos_emp'].astype(str)
#all_inter.groupby('fam')['pos_emp'].apply("|".join)
fam_pos_samples=all_inter.groupby(['fam','pos_emp'])['sample'].apply(list)#[1][199] is [1A_GGM_BL, 1B_GM_BL, 1D_DA_BL]
fam_pos_hettype=all_inter.groupby(['fam','pos_emp'])['het'].apply(set).apply(list)#6	93	[93A]
fam_pos_hettype_num=fam_pos_hettype.apply(len)#fam_pos_hettype[20][13395]->20   13395      2 have 2 type {'13395G', '13395A'}
fam_pos_major=all_inter.groupby(['fam','pos_emp'])['maa_emp'].apply(set).apply(list)#20 13395    [G, A]

all130=pd.read_table("/home/liuqi/work/4generations/result/ssp/dreep_ssp_3class/interHetNumGet/Sample142/process_result/sample130",names=['sample'])
df174=all130.copy()

#conta32=pd.read_table("/home/liuqi/work/4generations/result/ssp/dreep_ssp_3class/sampleDistributePlot/contaDel32",sep=r"\s+",usecols=[1],header=None)

#############
#############dreep_contav8 not need this line,we should cal all mut maf of the sample in this family by the het in InterHet142
#############
#df174=df174[~df174[0].isin(conta32[1])]#del 32 sample#should be named as df142

df174['fam'],df174['gener'],df174['tisu'],df174['y']=zip(*df174.apply(lambda row:addFam(row["sample"]),axis=1))
df174=df174.sort_values(['fam','y'],ascending=[True,False])
df174_num=df174.groupby('fam')['gener'].apply(len)#fam sample_num #all 46 fams
df174_sample=df174.groupby('fam')["sample"].apply(list)#fam sample_constitution 

fam_all={}
for fam_ix in all_inter['fam'].unique():#fam ID 9
 fam_all[fam_ix]={}
 for ix,ve in fam_pos_samples[fam_ix].items():#199	[1A_GGM_BL, 1B_GM_BL, 1D_DA_BL]#fam_pos_samples is <class 'pandas.core.series.Series'>
  pos_ix=ix#199
  if fam_pos_hettype_num[fam_ix][pos_ix]==1:
   if len(ve) < df174_num[fam_ix]:#1     4
    sampleNo=set(df174_sample[fam_ix])-set(ve)
    fam_all[fam_ix][pos_ix]={};fam_all[fam_ix][pos_ix]['no_sample']=sampleNo
    fam_all[fam_ix][pos_ix]['hettype']=fam_pos_hettype[fam_ix][pos_ix]#51   14767      [14767T]
  else:#fam:20 pos:13395
   fam_all[fam_ix][pos_ix]={};fam_all[fam_ix][pos_ix]['no_sample']=df174_sample[fam_ix]
   fam_all[fam_ix][pos_ix]['hettype']=fam_pos_hettype[fam_ix][pos_ix]
#print(fam_all)  

SspDir="/home/liuqi/work/4generations/result/ssp/dreep_ssp_3class/ssp_AfterDel/"
AllSSP=glob.glob("/home/liuqi/work/4generations/result/ssp/dreep_ssp_3class/ssp_AfterDel/*.ssp")

def sspRate(f):
 name=f.split("/")[-1].split(".")[0]
 df=pd.read_table(f,"\t",names=[str(i) for i in list(range(19))])
 df=df.fillna("")
 #fc,sc,oc=
 def extractCount(s):
  if s!="":
   lm=re.findall(r"\(([0-9-]+)\)",s);m=lm[0];ul=m.split("-");u,l=int(ul[0]),int(ul[1])
   return [u,l]
  else:
   return [0,0]
 df=df.copy()
 fcn=df.iloc[:,9].apply(lambda x: extractCount(x))
 df.insert(loc=19,column='FCN',value=fcn)
 df[['F1','F2']] = pd.DataFrame(df['FCN'].values.tolist(), index= df.index)
 scn=df.iloc[:,10].apply(lambda x: extractCount(x))
 df.insert(loc=22,column='SCN',value=scn)
 df[['S1','S2']] = pd.DataFrame(df['SCN'].values.tolist(), index= df.index)
 rate1=df['S1']/(df['F1']+df['S1']);rate2=df['S2']/(df['F2']+df['S2'])
 maf=(df['S1']+df['S2'])/(df['F1']+df['S1']+df['F2']+df['S2'])
 maaf=(df['F1']+df['F2'])/(df['F1']+df['S1']+df['F2']+df['S2'])#major allele frequency
 maf34=1-(maf+maaf)
 df.insert(loc=9,column='MAF1',value=rate1)
 df.insert(loc=10,column='MAF2',value=rate2)
 df.insert(loc=11,column='MAF',value=maf)
 df.insert(loc=12,column='MAAF',value=maaf)#major allele frequency
 df.insert(loc=13,column='MAF34',value=maf34)#the 3,4 allele frequency total
 del df['FCN'];del df['SCN']
 df["MAC"]=df["F1"]+df["F2"]#major count
 df["MIC"]=df["S1"]+df["S2"]#minor count
 df["C34"]=df["2"]-df["MAC"]-df["MIC"]#last 3rd 4th allele count
 df_gene=df[['0','1','2','3','4',"MAC","MIC",'F1','F2','S1','S2','MAF1','MAF2','MAF','MAAF','C34','MAF34']]#0-4:pos,ref,total_count,alt,sec_gene
 df_gene=df_gene.copy()
 df_gene.rename(columns={'0':'Pos','1':'Ref','2':'Count','3':'Alt','4':'Sec'},inplace=True)
 df_gene=df_gene[df_gene['Pos'].map(lambda x:x not in lPosDelete)].reset_index()#delete pos
 df_gene.drop('index',axis=1,inplace=True)
 df_gene['Phy']=df_gene.apply(lambda row:phyHetGet(row['Pos'],rCRS[row['Pos']-1],row['Sec']),axis=1)#phyHetGet(pos,ref,alt)
 def subHeterGet(df,maf,count):
  sub=df[df['Ref']!=df['Alt']]#;sub=sub.copy();sub['Sub']=sub['Pos'].astype(str)+sub['Alt']
  sub=sub[(sub['Alt']!="")&(sub['Count']>3)]
  #print(sub)
  sub=sub.copy();sub['Sub']=sub['Pos'].astype(str)+sub['Alt']
  heter=df[(df['MAF1']>=maf)&(df['MAF2']>=maf)&(df['S1']>=count)&(df['S2']>=count)]
  heter=heter.copy();heter['Het']=heter['Pos'].astype(str)+heter['Sec']
  return sub,heter
 Sub,Heter=subHeterGet(df_gene,0.01,3)
 Sub=Sub.copy()
 Sub['Phy_sub']=Sub.apply(lambda row:phyHetGet(row['Pos'],rCRS[row['Pos']-1],row['Alt']),axis=1)
 return name,df_gene#name,Sub['Sub'],Heter['Het'],Heter['Phy'],Sub['Phy_sub']

p0=Pool(20)
multres=[p0.apply_async(sspRate,args=(ssp,))for ssp in AllSSP]
sspRate_all={res.get()[0]:res.get()[1]for res in multres}#sub

def pos_info_maf(major_allele,minor_allele,pos_info,allele_i):
 if major_allele == allele_i:
  ve=pos_info['MAAF'].values[0]
 elif minor_allele == allele_i:
  ve=pos_info['MAF'].values[0]
 else:
  ve=pos_info['MAF34'].values[0]
 return ve

def noSamplePos(sample_list,pos,allele_list):
 d={}
 for sample_i in sample_list:
  sample_df=sspRate_all[sample_i]
  pos_info=sample_df[sample_df['Pos']==pos]#has the next 4 features 
  major_allele=pos_info['Alt'].values[0]
  #major_count =pos_info["MAC"].values[0]
  minor_allele=pos_info['Sec'].values[0]
  #minor_count =pos_info["MIC"].values[0]
  d[sample_i]={}
  for allele_i in allele_list:
   d[sample_i][allele_i]={}
   d[sample_i][allele_i]=pos_info_maf(major_allele,minor_allele,pos_info,allele_i)
 return d
 
def proPosDict(d):#d is fam_all store fam(10) pos(153) sampleNo([alist of sample should be extracted])
 d_no={}
 for fam in d:
  d_no[fam]={}
  for pos in d[fam]:
   no_sample=d[fam][pos]['no_sample']
   hettype=d[fam][pos]['hettype']
   if len(hettype)==1:
    allele_list=re.findall(r'[A-Z]',hettype[0])#list type
   else:
    allele_list=re.findall(r'[A-Z]',hettype[0])
    allele_list.append(re.findall(r'[A-Z]',hettype[1])[0])
   d_no[fam][pos]=noSamplePos(no_sample,pos,allele_list)
 return d_no

def getCount(pos_info,allele_should_get):#to get allele's count
 if pos_info['Alt'].values[0]==allele_should_get:
  allele_count =pos_info["MAC"].values[0]
 elif pos_info['Sec'].values[0]==allele_should_get:
  allele_count =pos_info["MIC"].values[0]
 else:
  allele_count =pos_info["C34"].values[0]
 return allele_count 

def pro_all_inter_all_not(all_inter_i,all_not):#all_not--> fam pos sample allele_i maf
 for fam in all_not:#6
  for pos in all_not[fam]:#93
   all_samples=all_not[fam][pos]#no sample
   for sample in all_samples:
    sample_df=sspRate_all[sample]##############from here to get the count!!!!!!!!!!!!!!!!
    pos_info=sample_df[sample_df['Pos']==pos]
    total_count =pos_info['Count'].values[0]
    minor_allele=pos_info['Sec'].values[0]

    for allele_i in all_samples[sample]:#get the count!
     count2=getCount(pos_info,allele_i)#second allele's count
     if fam_pos_major[fam][pos][0]!=allele_i:#solve 20 13395 two alleles
      major_should_get=fam_pos_major[fam][pos][0]
      count1=getCount(pos_info,major_should_get)
     else:
      major_should_get=fam_pos_major[fam][pos][1]
      count1=getCount(pos_info,major_should_get)
     if sample =="20A_GGM_BL":
      print(pos_info[["Count","Alt","MAC","Sec","MIC","C34"]])
      print(major_should_get,count1,allele_i,count2)
     maf_i=all_samples[sample][allele_i]#cant be "%.3f"%all_samples[sample][allele_i] or make 17 float
     all_inter_i=all_inter_i.append({'sample':sample,'pos_emp':pos,"cov_emp":total_count,"maa_emp":major_should_get,"mac_emp":count1,"mia_emp":allele_i,"mic_emp":count2,'maf_emp':maf_i,'mia_emp':allele_i,'fam':fam},ignore_index=True)#append to the all_inter df
 return all_inter_i 

if __name__=="__main__":
 all_not=proPosDict(fam_all)#get the rescue maf dict(fam pos sample allele_i)
 all_inter_i=all_inter[['sample','pos_emp',"cov_emp","maa_emp","mac_emp","mia_emp","mic_emp",'maf_emp','mia_emp','fam']]
 all_inter_i=pro_all_inter_all_not(all_inter_i,all_not)
 all_inter_i=all_inter_i.copy()
 all_inter_i['Fam'],all_inter_i['gener'],all_inter_i['tisu'],all_inter_i['y']=zip(*all_inter_i.apply(lambda row:addFam(row['sample']),axis=1))
 all_inter_i=all_inter_i.sort_values(['Fam','pos_emp','y'],ascending=[True,True,False])
 all_inter_i=all_inter_i.round({'maf_emp':3})#not work
 #all_inter_i['maf_emp']=all_inter_i['maf_emp'].astype(str)
 all_inter_i=all_inter_i.drop_duplicates()
 all_inter_i.to_csv("The130_SharePosIn_130_cprmaf_0315.xls",sep="\t",header=True,index=False,float_format="%g")#"%.3f" also not work
