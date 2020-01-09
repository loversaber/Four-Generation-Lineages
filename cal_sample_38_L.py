import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import numpy as np
import argparse
#input the first dataframe file
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from matplotlib.legend import Legend

def sample_i(fil):
 df=pd.read_csv(fil,"\t",header=0)
 df=df.copy()
 df["syb"]=df["fam"].astype(str)+"_"+df["pos_emp"].astype(str)+"_"+df["trans"]
 df12=df[df["trans"]=="GGM_GM"]
 df23=df[df["trans"]=="GM_MO"]
 df34=df[df["trans"]=="MO_DA"]
 def sample_one_pos_one_fam(df):
  df_sample=df.groupby("fam",group_keys=False).apply(lambda df:df.sample(1))
  return df_sample
 df12_i=sample_one_pos_one_fam(df12)
 df23_i=sample_one_pos_one_fam(df23)
 df34_i=sample_one_pos_one_fam(df34)
 df_38_i=pd.concat([df12_i,df23_i,df34_i]).reset_index(drop=True)
 syb_i=list(df_38_i["syb"].values)
 return df_38_i,syb_i

def sample_i_j(time,fil1,n1,n2,filpfx):
 n_i=fil1.split(".")[0].split("_")[1]
 df_38_i,syb_i=sample_i(fil1)
 L_value_n_i=df_38_i["L_Log10_"+n_i].sum()
 print(n_i,L_value_n_i)
 n_best=int(n_i)
 L_value_best=L_value_n_i
 for i in range(n1,n2+1):
  fil_j=filpfx+str(i)+".xls"
  df_j=pd.read_csv(fil_j,"\t",header=0)
  df_j["syb"]=df_j["fam"].astype(str)+"_"+df_j["pos_emp"].astype(str)+"_"+df_j["trans"]
  L_value_j=df_j[df_j["syb"].isin(syb_i)]["L_Log10_"+str(i)].sum()
  print(i,L_value_j)
  if L_value_j>L_value_best:
   L_value_best=L_value_j
   n_best=i
 print("$$$$$$$$$$$$$$$$ Best Nx:%d L_value:%.4f"%(int(n_best),L_value_best))
 return n_best,L_value_best

def plot_list(L_value_list,optpdf,color,alpha):
 class TextHandlerB(HandlerBase):
  def create_artists(self, legend, text ,xdescent, ydescent,width, height, fontsize, trans):
   tx = Text(width/2.,height/2, text, fontsize=fontsize,ha="center", va="center", fontweight="normal")
   return [tx]
 Legend.update_default_handler_map({str : TextHandlerB()})
 f=plt.figure(figsize=(8,4))
 (mu, sigma) = norm.fit(L_value_list)
 pvalue=norm.pdf(48,mu,sigma)
 pvalue="%.2e"%pvalue
 ax=sns.distplot(pd.Series(L_value_list),bins=np.arange(25,36,1),fit=norm,kde=False,\
 kde_kws={"label":"Smooth",'linewidth':1.5,"linestyle":"--","color":"K"},\
 fit_kws={'label': 'Normal','linewidth':1.5, "color": "r"},hist_kws={'linewidth':0.05,"color": color,"edgecolor":"black","alpha":alpha,"align":"mid","rwidth":1},label=str(pvalue))#,ax=ax)
 plt.axvline(48,color='r')
 legend_i=ax.get_legend_handles_labels()
 smooth_normal_handles=legend_i[0][:1]
 smooth_normal_handles.append("Pvalue ")
 plt.xticks(np.arange(min(L_value_list)-2,55,1))#max(L_value_list)+3,1))
 plt.legend(smooth_normal_handles,legend_i[1],loc='upper right',framealpha=0)
 plt.ylabel("Probability",fontsize=10,fontweight='semibold')
 plt.xlabel("Bottleneck Size,n",fontsize=10,fontweight='semibold')
 plt.title("The distribution of n",fontsize=12,fontweight='bold')
 plt.tight_layout()
 plt.savefig(optpdf,dpi=200)

if __name__=="__main__":
 parser=argparse.ArgumentParser(description="Sampling 38 Pos Get The Nx Distribution")
 parser.add_argument("-nspl",help="Sampling times. default=20",default=20,type=int)
 parser.add_argument("-filpfx",help="DF file prefix. default is rate2_",default="rate2_",type=str)
 parser.add_argument("-n1",help="Nx 1",type=int)
 parser.add_argument("-n2",help="Nx 2",type=int)
 parser.add_argument("fil1",help="One 100 Pos File")
 parser.add_argument("optpdf",help="Output distribution to PDF file")
 args=parser.parse_args()
 L_value_list=[]
 time=1
 while time <= args.nspl:
  print("#"*40+" Sampling Time:%d"%time)
  n_best,L_value_best=sample_i_j(time,args.fil1,args.n1,args.n2,args.filpfx)
  L_value_list.append(n_best)
  time+=1
 print(L_value_list)
 alpha=1
 color="xkcd:blue"
 plot_list(L_value_list,args.optpdf,color,alpha)
