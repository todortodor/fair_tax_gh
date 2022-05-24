#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:09:43 2022

@author: simonl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import os 

y = 2018
year = str(y)
dir_num = 2
carb_cost_l = pd.read_csv('results/'+year+'_'+str(dir_num)+'/runs').carb_cost

cons_tax = {}
iot_tax = {}
output_tax = {}
va_tax = {}
co2_prod_tax = {}
price_index_tax = {}
utility_tax = {}

cons_fair_tax = {}
iot_fair_tax = {}
output_fair_tax = {}
va_fair_tax = {}
co2_prod_fair_tax = {}
price_index_fair_tax = {}
contrib_fair_tax = {}

os.chdir('../tax_model_gh/')
import treatment_funcs_agri_ind_fe as t_tax
print('loading baseline')
cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b = t_tax.load_baseline(year)
sh = t_tax.shares(cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b)
dir_num=8
path = 'results/'+year+'_'+str(dir_num)
print('loading tax')
runs_tax = pd.read_csv(path+'/runs')
for carb_cost in tqdm(carb_cost_l):
    run = runs_tax.iloc[np.argmin(np.abs(runs_tax['carb_cost'] - carb_cost))]
    cons_tax[carb_cost], iot_tax[carb_cost], output_tax[carb_cost], \
        va_tax[carb_cost], co2_prod_tax[carb_cost] , price_index_tax[carb_cost], utility_tax[carb_cost]\
            = t_tax.sol_from_loaded_data(carb_cost, run, cons_b, iot_b, output_b, va_b,co2_prod_b, sh)

os.chdir('../fair_tax_gh/')
import treatment_funcs_fair_tax_agri_ind_fe as t_fair_tax
dir_num = 2
path = 'results/'+year+'_'+str(dir_num)
print('loading fair tax')
runs_fair_tax = pd.read_csv(path+'/runs')
for carb_cost in tqdm(carb_cost_l):   
    run = runs_fair_tax.iloc[np.argmin(np.abs(runs_fair_tax['carb_cost'] - carb_cost))]
    cons_fair_tax[carb_cost], iot_fair_tax[carb_cost], output_fair_tax[carb_cost], \
        va_fair_tax[carb_cost], co2_prod_fair_tax[carb_cost] , price_index_fair_tax[carb_cost], contrib_fair_tax[carb_cost]\
            = t_fair_tax.sol_from_loaded_data(carb_cost, run, cons_b, iot_b, output_b, va_b,co2_prod_b, sh)

#%% plot contributions for $100 tax

carb_cost = 1e-4
if carb_cost not in contrib_fair_tax.keys():
    print('This tax run doesnt exist')
    print('Runs are for carb costs of :')
    print(list(contrib_fair_tax.keys()))
    
labor = pd.read_csv('data/World bank/labor_force/labor.csv').set_index('country')[year]
cont = contrib_fair_tax[carb_cost]
cont = cont.join(labor).rename(columns={str(y):'labor'})
cont['contrib_per_worker_in_dollars'] = cont['new'] / cont['labor']*1e6
cont = cont.sort_values('contrib_per_worker_in_dollars')

fig, ax = plt.subplots(figsize=(12,8))

ax.bar(x = cont.index, height = cont.contrib_per_worker_in_dollars, label ='Per worker contribution for fair tax')
ax.set_xticklabels([''])
ax.bar_label(ax.containers[0],
              labels=cont.index,
              rotation=90,
              label_type = 'edge',
              padding=4,
              fontsize=12,
              zorder=10)

ax.legend(loc='lower right',fontsize = 20)
plt.margins(x=0.01)

plt.show()

#%% total contribution as function of tax

contributions = []
for carb_cost in carb_cost_l:
    cont = contrib_fair_tax[carb_cost]
    contributions.append(cont[cont.new > 0].sum())

fig, ax = plt.subplots(figsize=(12,8))

ax.plot(carb_cost_l,contributions)

plt.show()

#%% utility as function of tax

# labor = pd.read_csv('data/World bank/labor_force/labor.csv').set_index('country')[year]

# fig, ax = plt.subplots(figsize=(12,8))

# ax1 = ax.twinx()

# ax.plot(carb_cost_l*1e6,runs_fair_tax.utility,label='Fair tax utility')
# ax.plot(carb_cost_l*1e6,
#         [utility_tax[c].mean() for c in utility_tax],
#         label='Unfair tax utility')
# ax.plot(carb_cost_l*1e6,
#         [(utility_tax[c]*cons_fair_tax[c].groupby(level=2).sum().value).sum()/cons_fair_tax[c].sum().value for c in utility_tax],
#         label='Unfair tax utility weighted by consumption')
# ax.plot(carb_cost_l*1e6,
#         [(utility_tax[c]*labor).sum()/labor.sum() for c in utility_tax],
#         label='Unfair tax utility weighted by labor')

# ax1.plot(carb_cost_l*1e6,
#          [co2_prod_tax[c].new.sum() for c in co2_prod_tax],
#          label='Unfair tax emissions')
# ax1.plot(carb_cost_l*1e6,
#          [co2_prod_fair_tax[c].new.sum() for c in co2_prod_fair_tax],
#          label='Fair tax emissions')

# ax.legend()
# ax1.legend()

# plt.show()

#%% utility changes on average and weighted average

# utilities = pd.concat([sol_tax.utility*100, sol_fair_tax.utility*100],axis=1)
# utilities.columns = ['new_tax','new_fair_tax']
# utilities = utilities.sort_values('new_tax')

# labor = pd.read_csv('data/World bank/labor_force/labor.csv').set_index('country')[year]

# labor_weighted_fair_tax = (utilities.new_tax*labor).sum()/labor.sum()

# fig, ax = plt.subplots(figsize = (18,12))

# ax.bar(x = utilities.index, height = utilities.new_tax-100, label ='Unfair tax welfare changes')
# ax.set_xticklabels([''])
# ax.bar_label(ax.containers[0],
#              labels=utilities.index,
#              rotation=90,
#               label_type = 'edge',
#               padding=4,
#               fontsize=15,
#               zorder=10)
# ax.hlines(xmin=0,xmax=len(utilities)-1
#           ,y=utilities.new_tax.mean()-100,color='r', label ='Unfair tax welfare average change')
# ax.plot(utilities.new_fair_tax-100, color = sns.color_palette()[1], label ='Fair tax welfare changes')
# ax.legend(loc='lower right',fontsize = 20)

# plt.show()

# fig, ax = plt.subplots(figsize = (18,12))

# ax.bar(x = utilities.index, height = utilities.new_tax-100, label ='Unfair tax welfare changes')
# ax.set_xticklabels([''])
# ax.bar_label(ax.containers[0],
#              labels=utilities.index,
#              rotation=90,
#               label_type = 'edge',
#               padding=4,
#               fontsize=15,
#               zorder=10)
# ax.hlines(xmin=0,xmax=len(utilities)-1
#           ,y=labor_weighted_fair_tax-100,color='r', label ='Labor weighted unfair tax welfare average change')
# ax.plot(utilities.new_fair_tax-100, color = sns.color_palette()[1], label ='Fair tax welfare changes')
# ax.legend(loc='lower right',fontsize = 20)

# plt.show()

#%% average positive and negative contribution

# contrib = (sol_fair_tax.contrib*1e6).join(labor)
# contrib.columns = ['contribution','labor']

# pos_contri = contrib[contrib.contribution > 0]
# neg_contri = contrib[contrib.contribution <= 0]

# av_pos_contrib = (pos_contri.contribution).sum()/pos_contri.labor.sum()
# av_neg_contrib = (neg_contri.contribution).sum()/neg_contri.labor.sum()

#%% GDP evolution

# gdps = pd.concat([sol_tax.va.groupby(level=0).sum(),sol_fair_tax.va.groupby(level=0).sum().new,sol_fair_tax.va.groupby(level=0).sum().new+sol_fair_tax.contrib.contribution],axis=1)
# gdps.columns = ['value','new_tax','new_fair_tax','new_fair_tax_with_contrib']
# for c in ['new_tax','new_fair_tax','new_fair_tax_with_contrib']:
#     gdps[c+'_change']= gdps[c]-gdps['value']

# gdps.sort_values('value')['new_fair_tax_with_contrib_change'].plot(kind='bar',figsize = (18,12), title= 'GDP change taking contribution into account')
# gdps.sort_values('value')['new_fair_tax_change'].plot(kind='bar',figsize = (18,12), title= 'GDP change under fair tax')
# gdps.sort_values('value')['new_tax_change'].plot(kind='bar',figsize = (18,12), title= 'GDP change under unfair tax')
