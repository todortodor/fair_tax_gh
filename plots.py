#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 07:55:15 2022

@author: simonl
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from adjustText import adjust_text
from tqdm import tqdm
from labellines import labelLines
# import treatment_funcs as t
import treatment_funcs_fair_tax_agri_ind_fe as t

#%% Set seaborn parameters
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

#%% Baseline 2018 - range of carbon costs plots

print('Setting parameters for run')

y = 2018
year = str(y)
dir_num = 1
path = 'results/'+year+'_'+str(dir_num)

#%% Efficacy of carbon tax

print('Plotting efficacy of carbon tax in terms of emissions reduction')

runs = pd.read_csv(path+'/runs')
runs['emissions'] = runs['emissions']/1e3
runs['carb_cost'] = runs['carb_cost']*1e6
runs_low_carb_cost = runs[runs['carb_cost'] <= 1e3]

fig, ax1 = plt.subplots(figsize=(12,8))
color = 'tab:green'

ax1.set_xlabel('Carbon tax (dollar / ton of CO2)',size = 30)
ax1.set_xlim((runs_low_carb_cost.carb_cost).min(),(runs_low_carb_cost.carb_cost).max())
ax1.tick_params(axis='x', labelsize = 20)

ax1.set_ylabel('Global emissions (Gt)', color=color,size = 30)
ax1.plot((runs_low_carb_cost.carb_cost),(runs_low_carb_cost.emissions), color=color,lw=5)
ax1.tick_params(axis='y', labelsize = 20)

y_100 = runs_low_carb_cost.iloc[np.argmin(np.abs(runs_low_carb_cost.carb_cost-100))].emissions
y_0 = runs_low_carb_cost.iloc[0].emissions

ax1.vlines(x=100,
           ymin=0,
           ymax=y_100,
           lw=3,
           ls = '--',
           color = color)

ax1.hlines(y=y_100,
           xmin=0,
           xmax=100,
           lw=3,
           ls = '--',
           color = color)

ax1.annotate('100',xy=(100,0), xytext=(-20,-20), textcoords='offset points',color=color)
ax1.annotate(str(y_100.round(1)),
             xy=(0,y_100),
             xytext=(-37,-10),
             textcoords='offset points',color=color)

ax1.annotate(str(y_0.round(1)),
              xy=(0,y_0),
              xytext=(-37,-6),
              textcoords='offset points',color=color)

ax1.annotate("$100/Ton tax would reduce emissions by "+str(((y_0-y_100)*100/y_0).round(1))+"%",
            xy=(100, y_100), xycoords='data',
            xytext=(100+5, y_100+4),
            textcoords='data',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3",color= 'black'),
            bbox=dict(boxstyle="round", fc="w")
            )

ax1.margins(y=0)

ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
ax1.set_yticklabels(['0', '5', '10', '15', '20', '25', '30', '35', '40', '45'])

fig.tight_layout()

plt.show()

#%% Load data for effect on GDP and welfare and Price indexes and Share output traded

print('Loading welfare, GDP cost, price index changes, share output traded changes corresponding to a carbon tax')

runs = pd.read_csv(path+'/runs')
# Baseline data
cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b = t.load_baseline(year)
sh = t.shares(cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b)

sector_list = iot_b.index.get_level_values(1).drop_duplicates().to_list()
S = len(sector_list)
country_list = iot_b.index.get_level_values(0).drop_duplicates().to_list()
C = len(country_list)

cons_traded_unit = cons_b.reset_index()[['row_country','row_sector','col_country']]
cons_traded_unit['value'] = 1
cons_traded_unit.loc[cons_traded_unit['row_country'] == cons_traded_unit['col_country'] , ['value','new']] = 0
cons_traded_unit_np = cons_traded_unit.value.to_numpy()
iot_traded_unit = iot_b.reset_index()[['row_country','row_sector','col_country','col_sector']]
iot_traded_unit['value'] = 1
iot_traded_unit.loc[iot_traded_unit['row_country'] == iot_traded_unit['col_country'] , ['value','new']] = 0
iot_traded_unit_np = iot_traded_unit.value.to_numpy()



traded_new = []
traded_share_new = []
gross_output_new = []
gdp_new = []
utility = []
emissions = []
gdp = []
# dist = []
countries = country_list
price_index_l = {}

for country in countries:
    price_index_l[country] = []

carb_cost_l = np.linspace(0, 2e-4, 21)

for carb_cost in tqdm(carb_cost_l):
    run = runs.iloc[np.argmin(np.abs(runs['carb_cost'] - carb_cost))]
    utility.append(run.utility)
    emissions.append(run.emissions)

    sigma = run.sigma
    eta = run.eta
    num = run.num
    carb_cost = run.carb_cost

    # res = pd.read_csv(run.path).set_index(['country','sector'])

    cons, iot, output, va, co2_prod, price_index = t.sol_from_loaded_data(carb_cost, run, cons_b, iot_b, output_b, va_b,
                                                                          co2_prod_b, sh)

    gross_output_new.append(cons.new.to_numpy().sum() + iot.new.to_numpy().sum())
    traded_new.append(
        (cons.new.to_numpy() * cons_traded_unit_np).sum() + (iot.new.to_numpy() * iot_traded_unit_np).sum())
    traded_share_new.append(traded_new[-1] / gross_output_new[-1])
    gdp_new.append(va.new.sum())
    gdp.append(va.value.sum())
    for country in countries:
        price_index_l[country].append(price_index[country_list.index(country)])
        
#%% Plot

print('Plotting welfare and GDP cost corresponding to a carbon tax')


fig, ax = plt.subplots(2,2,figsize=(12,8))

color = 'g'

# Upper left - Emissions
ax[0,0].plot(np.array(carb_cost_l)*1e6,np.array(emissions)/1e3,lw=4,color=color)
ax[0,0].legend(['Global emissions (Gt)'])
ax[0,0].set_xlabel('')
ax[0,0].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[0,0].set_xlim(0,200)

y_100 = np.array(emissions)[np.argmin(np.abs(np.array(carb_cost_l)*1e6-100))]/1e3
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[0,0].vlines(x=100,
            ymin=0,
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[0,0].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[0,0].margins(y=0)

ax[0,0].annotate(str(y_100.round(1)),
             xy=(0,y_100),
             xytext=(-37,-5),
             textcoords='offset points',color=color)

ax[0,0].set_ylim(0,np.array(emissions).max()/1e3+0.5)

ax[0,0].set_yticks([0, 10, 20, 30, 40])
ax[0,0].set_yticklabels(['0','10','20','30','40'])

# Upper right - GDP
gdp_covid = np.array(gdp_new).max()*0.955
tax_covid = carb_cost_l[np.argmin(np.abs(np.array(gdp_new) - gdp_covid))]

ax[0,1].plot(np.array(carb_cost_l)*1e6,np.array(gdp_new)/1e6,lw=4)
ax[0,1].set_xlabel('')
ax[0,1].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[0,1].set_xlim(0,200)
ax[0,1].hlines(y=gdp_covid/1e6,linestyle=":",xmin=0,xmax=tax_covid*1e6,lw=3)
ax[0,1].legend(['GDP (thousand billion dollars)','GDP drop due to covid (scaled)'])
# ax[0,1].vlines(x=e_max,ymax=(GDP+DISU).max(), ymin=0, color=color, linestyle=":",lw=3)

y_100 = np.array(gdp_new)[np.argmin(np.abs(np.array(carb_cost_l)*1e6-100))]/1e6
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[0,1].vlines(x=100,
            ymin=np.array(gdp_new).min()/1e6,
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[0,1].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[0,1].margins(y=0)

ax[0,1].annotate(str(y_100.round(1)),
              xy=(0,y_100),
              xytext=(-37,-5),
              textcoords='offset points',color=color)

ax[0,1].set_ylim(np.array(gdp_new).min()/1e6,np.array(gdp_new).max()/1e6+0.5)

# Bottom left - Welfare
ax[1,0].plot(np.array(carb_cost_l)*1e6,utility,lw=4,color='r')
ax[1,0].legend(['Welfare from consumption'])
ax[1,0].set_xlabel('Carbon tax ($/ton of CO2)')
ax[1,0].set_xlim(0,200)
# ax[1,0].set_ylim(min(utility),1.001)

y_100 = np.array(utility)[np.argmin(np.abs(np.array(carb_cost_l)*1e6-100))]
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[1,0].vlines(x=100,
            ymin=np.array(utility).min(),
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[1,0].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[1,0].margins(y=0)

ax[1,0].set_ylim(np.array(utility).min(),1.005)

ax[1,0].annotate(str(y_100.round(3)),
              xy=(0,y_100),
              xytext=(-50,-10),
              textcoords='offset points',color=color)

ax[1,1].plot(np.array(carb_cost_l)*1e6,np.array(gdp_new)/gdp_new[0],lw=4)
ax[1,1].plot(np.array(carb_cost_l)*1e6,utility,lw=4,color='r')
ax[1,1].plot(np.array(carb_cost_l)*1e6,np.array(emissions)/emissions[0],lw=4,color='g')
ax[1,1].legend(['GDP','Welfare','Emissions'])
ax[1,1].set_xlabel('Carbon tax (dollar/ton of CO2)')
ax[1,1].set_xlim(0,200)

plt.tight_layout()
plt.show()

#%% Effect on output share traded

print('Plotting share of output traded')

fig, ax1 = plt.subplots(figsize=(12,8))
color = 'tab:blue'

ax1.set_xlabel('Carbon tax (dollar / ton of CO2)',size = 30)
ax1.set_xlim(0,1000)
ax1.tick_params(axis='x', labelsize = 20)

ax1.set_ylabel('Share of output traded (%)', color=color,size = 30)
ax1.plot(np.array(carb_cost_l)*1e6,np.array(traded_share_new)*100, color=color,lw=5)
ax1.tick_params(axis='y', labelsize = 20)

# y_100 = (np.array(traded_share_new)/traded_share_new[0])[np.argmin(np.abs(np.array(carb_cost_l)*1e6 -100))]

# ax[1,0].vlines(x=100,
#             ymin=0.995,
#             ymax=y_100,
#             lw=3,
#             ls = '--',
#             color = color)
#
# ax[1,0].hlines(y=y_100,
#             xmin=0,
#             xmax=100,
#             lw=3,
#             ls = '--',
#             color = color)

ax1.margins(y=0)

# ax1.set_ylim((np.array(traded_share_new)*100).min()-0.05, (np.array(traded_share_new)*100).max() + 0.05)
ax1.set_ylim(12,15)

# ax[1,0].annotate(str(y_100.round(3)),
#               xy=(0,y_100),
#               xytext=(-50,-5),
#               textcoords='offset points',color=color)
#
# ax[1,0].set_yticks([1.00,1.01,1.02,1.03])
# ax[1,0].set_yticklabels(['', '1.01', '1.02', '1.03'])

plt.tight_layout
plt.show()

#%% Utility and contributions

y = 2018
year = str(y)
dir_num = 1
carb_cost = 1e-4

sol = t.sol(y,dir_num,carb_cost)
labor = pd.read_csv('data/World bank/labor_force/labor.csv').set_index('country')[year]
contrib = sol.contrib

contrib['labor'] = labor
contrib['contrib_per_worker'] = contrib['contribution'] / contrib['labor']*1e6

contrib.contribution.sort_values().plot(kind = 'bar', figsize = (18,12), title = 'Contribution ($)')
contrib.contrib_per_worker.sort_values().plot(kind = 'bar', figsize = (18,12), title = 'Contribution ($)')
