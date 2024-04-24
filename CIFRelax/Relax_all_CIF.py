from __future__ import annotations

import os
import warnings
from pathlib import Path
import scipy.constants as cns
import numpy as np
import pandas as pd
import math
import time
import urllib.request
import html
from collections import Counter
import shutil

from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core.periodic_table import Species

import matgl
from matgl.ext.ase import Relaxer

warnings.filterwarnings("ignore")

pd.options.display.max_rows = 4000

os.environ["MPRESTER_MUTE_PROGRESS_BARS"] = "true"
mpr = MPRester("j1QsDSZwFP7jVWZM6XVIo6STg198M5DW")

# Load the pre-trained M3GNet Potential
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
# create the M3GNet Relaxer
relaxer = Relaxer(potential=pot)

def print_time(t):
    print(f'Time Taken: {int(t//60)}:'+f'{t%60:0.2f}'.zfill(5))

def get_parser(file):
    try:
        parser = CifParser(file)
    except:
        with open(file,'r+') as f:
            r=f.read()
            r=r.replace('?','nan')
            f.write(r)
        parser = CifParser(file)
    return parser

def print_data(struct):
    struct_dict={
        'a'     :np.round(struct.lattice.a,2),
        'b'     :np.round(struct.lattice.b,2),
        'c'     :np.round(struct.lattice.c,2),
        'alpha' :np.round(struct.lattice.alpha,2),
        'beta'  :np.round(struct.lattice.beta,2),
        'gamma' :np.round(struct.lattice.gamma,2),
        'Space Group'  :struct.get_space_group_info(symprec=0.01)[0]
        }
    print(struct_dict)
    return

def make_prop_dict(struct,val_keys):
    val_dict={val:np.nan for val in val_keys}
    if len(struct)>0:
        try:
            comp=Composition(Counter([el for el in struct.species]))
            formula=str(comp.element_composition)
            mass=float(comp.weight/cns.N_A)
        except:
            formula=np.nan
            mass=np.nan
        vol=struct.lattice.volume*10**(-24)
        struct_dict={
            'name'  :formula,
            'a'     :struct.lattice.a,
            'b'     :struct.lattice.b,
            'c'     :struct.lattice.c,
            'alpha' :struct.lattice.alpha,
            'beta'  :struct.lattice.beta,
            'gamma' :struct.lattice.gamma,
            'Space Group'  :struct.get_space_group_info(symprec=0.01)[0],
            'density'  : mass/vol
        }
        entry={val:struct_dict[val] for val in val_keys}
        return entry
    else:
        return val_dict

def make_error_dict(structs,val_keys=[]):
    rem_keys=['Space Group','density','name']
    entry={val:err_func(structs['test'][val],structs['pred'][val]) for val in val_keys}
    entry.update({'RMSD':np.sqrt(np.sum([entry[x]**2 for x in val_keys if x not in rem_keys])/(len(val_keys)-1))})
    for id,res_type in enumerate(['Original','Relaxed']):
        fold=['test','pred'][id]
        try:
            new_sg=fix_sg(structs,fold)
            entry.update({f'Space Group {res_type}':new_sg})
        except:
            entry.update({f'Space Group {res_type}':structs[fold]['Space Group']})
        entry.update({f'density {res_type}':structs[fold]['density']})
    return entry

def fix_sg(structs,fold):
    sg=structs[fold]['Space Group'].split('_')
    new_sg=''
    for part in range(len(sg)-1):
        new_sg+=sg[part][1:-1]
        new_sg+=f'${sg[part][-1]}_{sg[part+1][0]}$'
    new_sg=sg[0][0] + new_sg + sg[-1][1:]
    sg=new_sg.split('-')
    new_sg=sg[0]
    for part in range(1,len(sg)):
        new_sg+='$\\bar{'+sg[part][0]+'}$'+sg[part][1:]
    return new_sg

def err_func(theo,pred):
    if isinstance(theo, str):
        return theo==pred
    else:
        return (theo-pred)/pred


def get_results(datafilename,errfilename,root):
    val_keys=['name','a','b','c','alpha','beta','gamma','Space Group','density']
    all_files=[file for file in Path(f"{p}").glob("**/*.cif") if 'tmp' not in str(file)]
    full_dict={}
    for file in all_files:
        full_folder={}
        full_folder['test']='/'.join(str(file).split('/')[:-1])[len(str(root)):]
        for subtype in ['fix','pred']:
            full_folder[subtype]=str(root)+f'/tmp/{subtype}'+str(full_folder['test'])
        full_folder['test']=root+full_folder['test']
        key=file.stem
        File_key=f'/{key}.cif'
        structs={}
        for full in full_folder.keys():
            f=full_folder[full]+File_key
            try:
                parser = get_parser(f)
                structure=parser.get_structures()[0]
            except:
                structure={}
            structs[full]=make_prop_dict(structure,val_keys)
        structs["error"]=make_error_dict(structs,val_keys=val_keys)            
        full_dict[key]=structs

    Meta=pd.read_csv(f'{p}/results/Meta_results.csv',index_col='ID').T.to_dict()
    
    Meta_keys=Meta[[j for j in Meta.keys()][0]].keys()

    
    for key in full_dict.keys():
        full_dict[key]['Meta']=dict([])
        if int(key) in Meta.keys():
            MetaData=Meta[int(key)]
            for sub_key in Meta_keys:
                full_dict[key]['Meta'][sub_key]=MetaData[sub_key]
                if sub_key=='atoms':
                    if not MetaData[sub_key]:
                        full_dict[key]['Meta'][sub_key]=np.nan
                        full_dict[key]['Meta']['num_atoms']=np.nan
                        full_dict[key]['Meta']['total_electrons']=np.nan
                    atoms=Composition(eval(MetaData[sub_key]))
                    full_dict[key]['Meta'][sub_key]=atoms.as_dict()
                    full_dict[key]['Meta']['num_atoms']=atoms.num_atoms
                    full_dict[key]['Meta']['total_electrons']=atoms.total_electrons
        else:
            for sub_key in Meta_keys:
                full_dict[key]['Meta'][sub_key]=np.nan
        full_dict[key]['Meta']['URL']=f'https://www.crystallography.net/cod/{key}.html'
        name=full_dict[key]['test']['name']
        if not pd.isnull(name):
            name=name.split(' ')
        else:
            f = urllib.request.urlopen(full_dict[key]['Meta']['URL'])
            myfile = f.read().decode("utf-8")
            for name_type in [
                                'th>Formula</th'
                                ]:
                if name_type in myfile:
                    name=myfile.split(name_type)[1].split('</td>')[0].split('<td>')[1]
                    name=html.unescape(name)
                    if 'Form' in name_type:
                        name=name.split(' ')
        for id,el in enumerate(name):
            i=0
            for symb in el:
                if symb.isdigit():
                    i=el.index(symb)
                    break
            if i!=0:
                if el[i:]=='1':
                    el=el[:i]
                else:
                    el=el[:i]+'_'+'{'+el[i:]+'}'
            name[id]=el
        name='$\\rm '+''.join(name)+' $'
        full_dict[key]['Meta']['name']=name
    error_dict={key:full_dict[key]['error']|full_dict[key]['Meta'] for key in full_dict.keys()}
    pd.DataFrame(error_dict).T.rename_axis('ID').to_csv(errfilename)
    pd.DataFrame(full_dict).T.rename_axis('ID').to_csv(datafilename)

##########################################################
##########################################################
##########################################################

Results_name='Relaxation_results'
p='.'
selected=False

while not selected:
    PathDict={}
    i=0
    PathDict[i]=p
    for dirpath in Path(p).glob('*/'):
        if dirpath.is_dir():
            dirp=str(dirpath).split('/')[-1]
            if dirp not in ['tmp','results']:
                i+=1
                PathDict[i]=dirpath
    if i==0:
        break
    [print(key,PathDict[key]) for key in PathDict.keys()]
    print('-----',flush=True)
    selection=int(input('Choose folder:'))
    if selection==0:
        selected=True
    else:
        p=str(PathDict[selection])



all_files=[file for file in Path(f"{p}").glob("**/*.cif") if 'tmp' not in str(file)]
files_left=len(all_files)-1
start_time = time.time()
dirs=[p+folder for folder in ['/tmp/fix','/tmp/pred','/results']]
for dir in dirs:
    if not Path(dir).exists():
            Path(dir).mkdir(parents=True)

C_res=dict([])
for file in all_files:
    full_folder={}
    full_folder['test']='/'.join(str(file).split('/')[:-1])[len(str(p)):]
    for subtype in ['fix','pred']:
        full_folder[subtype]=str(p)+f'/tmp/{subtype}/'+str(full_folder['test'])
        if not Path(full_folder[subtype]).exists():
            Path(full_folder[subtype]).mkdir(parents=True)
        full_folder[subtype]=str(Path(full_folder[subtype]).absolute())
    Suffix=''
    key=file.stem
    File_key=f'/{key}.cif'
    print(
        f"{'ID:':>10}{key:>10}",
        f"{'Running:':>10}{len(all_files)-files_left:>6}/{len(all_files):<3}",
        "--",
        sep="\n"
    )
    
    loop_start=time.time()
    

    try:
        parser = get_parser(str(file))
        structure = parser.get_structures()[0]
        print_data(structure)
        CifWriter(structure).write_file(full_folder['fix']+File_key)
        parser = get_parser(full_folder['fix']+File_key)
        structure = parser.get_structures()[0]
        print_data(structure)
        # structure = relaxer.relax(structure)["final_structure"]
        print_data(structure)
        CifWriter(structure,symprec=0.01).write_file(full_folder['pred']+File_key)
        print_time(time.time()-loop_start)
        C_res[key]={
            'ID'    :key,
            'time'  :time.time()-loop_start,
            'folder':str(full_folder['pred']),
            'atoms' :Composition(Counter([el for el in structure.species])).element_composition.as_dict()
                        }
    except:
        print('failed')
        print_time(time.time()-loop_start)
        C_res[key]={
            'ID'    :key,
            'time'  :np.nan,
            'folder':str(full_folder['pred']),
            'atoms' :''
                    }
        
    files_left-=1
    
    pd.DataFrame.from_dict(C_res).T.to_csv(f'{p}/results/Meta_results.csv',index=False)


print(f'\nDone Relaxing!')
print_time(time.time()-start_time)

get_results(f'{p}/results/Database.csv',f'{p}/results/Error_results.csv',p)

print(f'\nDatabase Created!')

delete=input('Delete temporary files (y/n)?')
if delete not in ['y','n']:
    delete=input('Delete temporary files (y/n)?')

if delete=='n':
    Path(f'{p}/tmp/').rename(f'{p}/Output/')
elif delete=='y':
    shutil.rmtree(f'{p}/tmp/')