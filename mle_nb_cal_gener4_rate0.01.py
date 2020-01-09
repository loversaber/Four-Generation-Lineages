import pandas as pd
import math,argparse
import numpy as np
import numba as nb
from numba import prange,cfunc
from functools import partial
from scipy.integrate import quad
from scipy.misc import factorial
from multiprocessing import Pool
import time
import scipy.integrate as si
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable

import warnings
warnings.filterwarnings('ignore')

@nb.jit
def A1(mf,x,n):
 C=math.lgamma(n+1)-math.lgamma(x+1)-math.lgamma(n-x+1)
 A1_1=math.log(mf)*x
 A1_2=math.log(1-mf)*(n-x)
 item=C+A1_1+A1_2
 A1_result=np.exp(1)**item
 return A1_result

@nb.vectorize(["float64(float64,float64,float64,float64)"])
def formula_6(mt_i,mobs,mn,epsilon):
 item3_A2=np.array([math.exp(1)**(math.log(1-epsilon)*i+math.log(epsilon)*(mt_i-i)+\
           math.log(epsilon)*(mobs-i)+math.log(1-epsilon)*(mn-mt_i-mobs+i)+\
           math.lgamma(mt_i+1)-math.lgamma(mt_i-i+1)-math.lgamma(i+1)+\
           math.lgamma(mn-mt_i+1)-math.lgamma(mobs-i+1)-math.lgamma(mn-mt_i-mobs+i+1)) for i in prange(0,min(mt_i,mobs)+1) if mn-mt_i-mobs+i>=0])
 return item3_A2.sum()

@nb.vectorize(["float64(float64,float64,float64)"])
def A2_numerator_denominator(mf,mt_i,mn):
 item1_A2=A1(mf,mt_i,mn)
 item2_A2=quad(A1,0,1,args=(mt_i,mn))
 return item1_A2/item2_A2[0] 

@nb.jit
def integrate_A(mf,x,n,mobs,mn,epsilon):
 #######A1
 A1_result=A1(mf,x,n)
 #######A2
 mt_i_list=np.arange(mn+1)
 A2_result=np.dot(formula_6(mt_i_list,mobs,mn,epsilon),A2_numerator_denominator(mf,mt_i_list,mn))
 A1_A2=A1_result*A2_result
 return A1_A2

@nb.jit
def do_integrate(func,x,n,mobs,mn,epsilon):
 return quad(func,0,1,args=(x,n,mobs,mn,epsilon))[0]

@nb.jit
def B_B2(ct_i,xf,nf,cn):
 C=math.lgamma(cn+1)-math.lgamma(ct_i+1)-math.lgamma(cn-ct_i+1)
 B1_1=math.log(xf/nf)*ct_i
 B1_2=math.log(1-xf/nf)*(cn-ct_i)
 item=C+B1_1+B1_2
 B2_result=math.exp(1)**item
 return B2_result

@nb.jit
def B_B3(xf,x,n,nf):
 cc2=nf-n+x
 if cc2-xf >= 0:
  C_B3=math.lgamma(nf-n+1)-math.lgamma(xf-x+1)-math.lgamma(nf-n-xf+x+1)
  f8_1=math.lgamma(xf)-math.lgamma(x)
  f8_2=math.lgamma(nf-xf)-math.lgamma(n-x)
  f8_3=math.lgamma(n)-math.lgamma(nf)
  item_B3=C_B3+f8_1+f8_2+f8_3
  B3_result=math.exp(1)**item_B3
 return B3_result

@nb.jit
def B2_B3(ct_i,nf,cn,x,n):
 xf_list=list(range(x,nf))
 result=np.dot(B_B2(ct_i,xf_list,nf,cn),B_B3(xf_list,x,n,nf))
 return result 

@nb.vectorize(["float64(float64,float64,float64,float64,float64)"])
def B2_B3_v(ct_i,nf,cn,x,n):
 result=0
 for xf_list in range(x,nf):
  result+=B_B2(ct_i,xf_list,nf,cn)*B_B3(xf_list,x,n,nf)
 return result

@nb.vectorize(["float64(float64,float64,float64,float64,float64,float64,float64)"])
def formula_B_v(ct_i,cobs,x,n,cn,epsilon,nf):
 return np.dot(formula_6(ct_i,cobs,cn,epsilon),B2_B3_v(ct_i,nf,cn,x,n))#.sum()#np.sum and sum() is not make any sense;because there is just one number

@nb.jit
def formula_1(n,mobs,mn,epsilon,cobs,cn,nf):#xf1 is x_i;xf2 is nf ? nf=1000 epsilon=0.0008
 sigma_n=0
 xf2=nf
 for x in prange(1,n):#simple is (1,n)for formula 8
  A=do_integrate(integrate_A,x,n,mobs,mn,epsilon)
  B=np.sum(formula_B_v(list(range(cn+1)),cobs,x,n,cn,epsilon,nf))
  item_0=A*B
  print(n,x,B,A)
  sigma_n+=item_0
 return sigma_n

def get_mother_child(df4,m,c):
 df_m_c=df4[[m,c]].rename(columns={m:"mother",c:"child"})
 return df_m_c

def generate_data():
 df=pd.read_table("/home/liuqi/work/cal_mle/rate0.01/The130_SharePosIn_130_cprmaf_0315_2.xls",sep="\t",header=0)
 df["ALL_MIC"]=df[["cov_emp","mic_emp"]].values.tolist()
 df4fisher=pd.pivot_table(df,values='ALL_MIC',index=['fam','pos_emp'],columns='gener',aggfunc="first").reset_index()[["fam","pos_emp","GGM","GM","MO","DA"]]
 df4=df4fisher.dropna().reset_index(drop=True)
 df_GGM_GM=get_mother_child(df4,"GGM","GM")
 df_GM_MO=get_mother_child(df4,"GM","MO")
 df_MO_DA=get_mother_child(df4,"MO","DA")
 df64=pd.concat([df_GGM_GM,df_GM_MO,df_MO_DA]).reset_index(drop=True)
 return df64

@nb.jit
def cal_mother_child(row,n,epsilon,nf):
 m=row["mother"]
 c=row["child"]
 mobs,mn=m[1],m[0]
 cobs,cn=c[1],c[0]
 L=formula_1(n,mobs,mn,epsilon,cobs,cn,nf)#formula_1(n,mobs,mn,epsilon,cobs,cn,nf)
 if L==0:
  L_log=-math.inf
 else:
  L_log=math.log(L,1.005)
  L_log10=math.log10(L)
 return L,L_log,L_log10

def parallel_df(df,nthreads,func):
 df_split=np.array_split(df,nthreads)
 pool=Pool(nthreads)
 df_end=pd.concat(pool.map(func,df_split))
 pool.close()
 pool.join()
 return df_end

def cal_df_L(df_split,Nx,epsilon,nf):
 df123=df_split.copy()
 L_syb="L_"+str(Nx)
 L_Log_syb="L_Log_"+str(Nx)
 L_Log_syb2="L_Log10_"+str(Nx)
 df123[L_syb],df123[L_Log_syb],df123[L_Log_syb2]=zip(*df123.apply(lambda row:cal_mother_child(row,Nx,epsilon,nf),axis=1))
 return df123

def End_parallel_cal(df_data,Nx,epsilon,nf,nthreads):
 L_syb="L_"+str(Nx)
 L_Log_syb="L_Log_"+str(Nx)
 L_Log_syb2="L_Log10_"+str(Nx)
 cal_df_L_Parametered=partial(cal_df_L,Nx=Nx,epsilon=epsilon,nf=nf)
 df_data_L=parallel_df(df_data,nthreads,cal_df_L_Parametered)
 multiple_L=1
 for l in df_data_L[L_syb]:
  multiple_L*=l
 #multiple_L=float("{0:.4e}".format(multiple_L))
 print(Nx)
 print(df_data_L)
 Likelihood_value=df_data_L[L_Log_syb].sum()
 Likelihood_value2=df_data_L[L_Log_syb2].sum()
 print("%d\t%.4e\t%.4f\t%.6f"%(Nx,multiple_L,Likelihood_value,Likelihood_value2))
 multiple_L=float("{0:.10e}".format(multiple_L))
 Likelihood_value=float("{0:.10f}".format(Likelihood_value))
 Likelihood_value2=float("{0:.10f}".format(Likelihood_value2))
 return Nx,multiple_L,Likelihood_value,Likelihood_value2

##############filter subset pos based on the max(cn,mn)
def split_list(row):
 m=row["mother"]
 c=row["child"]
 mobs,mn=m[1],m[0]
 cobs,cn=c[1],c[0]
 max_n=max(mn,cn)
 mmaf=mobs/mn
 cmaf=cobs/cn
 return mobs,mn,cobs,cn,max_n,mmaf,cmaf

def filter_df(df_data):
 df=df_data.copy()
 df["mobs"],df["mn"],df["cobs"],df["cn"],df["max_n"],df["m_maf"],df["c_maf"]=zip(*df.apply(lambda row:split_list(row),axis=1))
 return df

if __name__=="__main__":
 parser=argparse.ArgumentParser(description="Based LMK 2016 Paper to Calculate Nbe")
 #Nx should be start from in 2 because formula 8
 parser.add_argument("-Nx1",help="Range of Bottleneck Size,Start in Nx1,[1,50] in the Paper.default=1",default=1,type=int)
 parser.add_argument("-Nx2",help="Range of Bottleneck Size,End in Nx2,[1,50] in the Paper.default=20",default=20,type=int)
 parser.add_argument("-nthreads",help="Paralleled Process the Heteroplasmy DataFrame.default=20",default=20,type=int)
 parser.add_argument("opt",help="Output the Likelihood Values in Specific Nx to a File")
 args=parser.parse_args()

 epsilon=0.0006
 nf=1000
 df_data0=generate_data()
 df_data1=filter_df(df_data0)
 #df_data1=df_data11.iloc[128:]#MO_DA
 #df_data=df_data1[(df_data1["mobs"]>=10)&(df_data1["cobs"]>=10)&(df_data1["max_n"]<1000)]#15 poses
 df_data=df_data1[(df_data1["m_maf"]>=0.01)&(df_data1["c_maf"]>=0.01)]#125 pos rate 0.01
 #df_data=df_data1.head(4)
 df_end=pd.DataFrame(columns=("Nx","L","L_Log","L_Log10"))
 #print(test_L) 
 for Nx in np.arange(args.Nx1,args.Nx2+1):#Nx should start at 1
  start = time.time()
  nx,mul_L,Likelihood_value,Likelihood_value_log10_sum=End_parallel_cal(df_data,Nx,epsilon,nf,args.nthreads)
  end = time.time()
  print("Elapsed time of %d = %s" % (Nx,end - start))

  df_end=df_end.append({"Nx":nx,"L":mul_L,"L_Log":Likelihood_value,"L_Log10":Likelihood_value_log10_sum},ignore_index=True)
  df_end.to_csv(args.opt,sep="\t",header=True,index=False,mode="w+")#float_format="%.2f"
 #df_end.sort_values(by='Nx',ascending=True,inplace=True)
 #df_end.to_csv("END_Likelihood_nb_v4.5_"+str(args.Nx1)+"_"+str(args.Nx2)+".xls",sep="\t",index=False)
