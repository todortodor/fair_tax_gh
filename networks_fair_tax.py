#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:21:25 2022

@author: simonl
"""

import pandas as pd
import numpy as np
# import treatment_funcs_agri as t
import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# from scipy.spatial.distance import pdist
# from sklearn import metrics
import seaborn as sns
from tqdm import tqdm
# from adjustText import adjust_text
# import networkx as nx
# import plotly.graph_objects as go
# import plotly.io as pio
# import geopandas as gpd
# pio.renderers.default='browser'
import treatment_funcs_fair_tax_agri_ind_fe as t

sns.set()
sns.set_context('talk')
sns.set_style("whitegrid")

#%% load solution

# carb_cost = 1e-4

sol_all = {}
dir_num = 1
carb_cost = 1e-4
# carb_cost = {}


dollar_adjustment = pd.read_csv('data/dollar_adjustment.csv',sep=';',decimal=',').set_index('year')

# for y in range(1995,2019):
for y in range(2018,2019):
    print(y)
    year = str(y)
    # sol_all[y] = t.sol(y,dir_num,dollar_adjustment.loc[y]['dollar_adjusted']*carb_cost)
    sol_all[y] = t.sol(y,dir_num,carb_cost)
    # path = '/Users/simonl/Documents/taff/tax_model/results/'+year+'_'+str(dir_num)
    # runs = pd.read_csv(path+'/runs')
    # run = runs.iloc[np.argmin(np.abs(runs.emissions-runs.iloc[0].emissions*0.7))]
    # print(run.carb_cost)    
    # carb_cost[y] = run.carb_cost
    # sol_all[y] = t.sol_emissions(y,dir_num,runs.iloc[0].emissions*0.7)

# for y in range(1995,2019):
#     print(y)
#     sol_all[y] = t.sol(y,dir_num,carb_cost)

sector_list = sol_all[y].iot.index.get_level_values(1).drop_duplicates().to_list()
S = len(sector_list)
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()
C = len(country_list)

#%% compute traded

traded = {}
tot = {}

for y in sol_all.keys():
    print(y)
    # y = 2018

    # cons = sol_all[y].cons
    # iot = sol_all[y].iot
    
    # cons_traded = cons.reset_index()
    # cons_traded.loc[cons_traded['row_country'] == cons_traded['col_country'] , ['value','new']] = 0
    # cons_traded = cons_traded.set_index(['row_country', 'row_sector', 'col_country'])
    # cons_traded = pd.concat({'cons':cons_traded}, names=['col_sector']).reorder_levels([1,2,3,0])
    
    # iot_traded = iot.reset_index()
    # iot_traded.loc[iot_traded['row_country'] == iot_traded['col_country'] , ['value','new']] = 0
    # iot_traded = iot_traded.set_index(['row_country', 'row_sector', 'col_country','col_sector'])
    
    # traded[y] = pd.concat([iot_traded,cons_traded])
    tot[y] = pd.concat([sol_all[y].iot,pd.concat({'cons':sol_all[y].cons}, names=['col_sector']).reorder_levels([1,2,3,0])])
    temp = tot[y].reset_index()
    traded[y] = temp[temp['row_country'] != temp['col_country']].set_index(['row_country', 'row_sector', 'col_country','col_sector'])
    del temp
    

# c_c.to_csv('/Users/simonl/Documents/taff/tax_model/graphs/regions3.csv')


#%% by regions with sectors

# c_c = traded[y].groupby(level=[0,2]).sum().reset_index().copy()
# c_c = traded[y].groupby(level=[0,1,2]).sum().reset_index().copy()
c_c = tot[y].groupby(level=[0,1,2]).sum().reset_index().copy()

world = pd.read_csv('/Users/simonl/Documents/taff/datas/country_coord/countries_codes_and_coordinates.csv')[['Alpha-3 code','Latitude (average)', 'Longitude (average)']]
world.columns = ['country','latitude','longitude']
for column in world.columns: 
    world[column] = world[column].str.replace('"','')
    world[column] = world[column].str.replace(' ','')
    # world[column] = world[column].astype('str')
world['country'] = world['country'].astype('string')
world.drop_duplicates('country',inplace=True)
# world['country'] = world['country'].to_string()
world['latitude'] = pd.to_numeric(world['latitude'])
world['longitude'] = pd.to_numeric(world['longitude'])
# world.set_index('country',inplace=True)

sector_map = pd.read_csv('/Users/simonl/Documents/taff/datas/OECD/industry_labels_after_agg_expl.csv',sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_map['row_sector'] = sector_list
sector_map = sector_map.reset_index().set_index('row_sector')

country_map = pd.read_csv('/Users/simonl/Documents/taff/datas/OECD/country_continent.csv',sep=';')

country_map['region_1'] = country_map.region_1.str.split().apply(reversed).apply(' '.join)

c_c['row_country'] = c_c.row_country.replace(country_map.set_index('country')['region_1'])
c_c['col_country'] = c_c.col_country.replace(country_map.set_index('country')['region_1'])

c_c = c_c.replace('TWN','Asia Eastern')
c_c = c_c.replace('ROW','Africa')
c_c = c_c.replace('Africa Northern','Africa')
c_c = c_c.replace('Africa Southern','Africa')
c_c = c_c.replace('America Central','America Northern')
c_c = c_c.replace('America Central','America Northern')
c_c = c_c.replace('Europe Southern','Europe Western')
c_c = c_c.replace('Asia Central','Asia Eastern')
c_c = c_c.replace('Zealand New and Australia','Asia South-Eastern')
c_c = c_c.replace('Asia South-eastern','Asia South-Eastern')
c_c = c_c.replace('Africa Southern','Africa')
c_c = c_c.replace('Asia Western','Middle East')
# c_c = c_c.replace('Zealand New and Australia','Oceania')

# c_c = c_c.groupby(['row_country','col_country']).sum()
c_c['row_sector'] = c_c.row_sector.replace(sector_map['industry'])
c_c = c_c.groupby(['row_country','row_sector','col_country']).sum()
c_c = c_c.rename_axis(['Exporter','Sector','Importer'])

c_c['value'] = c_c['value'].round(2)
c_c['new'] = c_c['new'].round(2)

c_c.columns = ['Trade flow (Mio$)','New trade flow (Mio$)']

c_c['Trade flow change (Mio$)'] = c_c['New trade flow (Mio$)'] - c_c['Trade flow (Mio$)']
# c_c['Trade flow change (%)'] = (c_c['Trade flow (Mio$)']/c_c['New trade flow (Mio$)'] - 1)*100

c_c = c_c.reset_index().pivot(index=['Exporter','Importer'],columns='Sector', values = ['Trade flow (Mio$)','New trade flow (Mio$)','Trade flow change (Mio$)'])
# test.rename_axis(['Trade flow', 'Trade flow change (Mio$)', 'Trade flow change (%)'],axis=1)

# test = 
# pd.concat([test.groupby(axis=1,level=0).sum(),test],axis=1,names=['Total',''])
c_c = pd.concat({' Total':c_c.groupby(axis=1,level=0).sum()},axis=1).reorder_levels([1,0],axis=1).join(c_c)

c_c[pd.MultiIndex.from_product([['Trade flow change (%)'],c_c.columns.get_level_values(1).drop_duplicates()])] = (c_c['New trade flow (Mio$)']/c_c['Trade flow (Mio$)']- 1)*100
c_c = c_c.drop(['New trade flow (Mio$)'],axis=1)
c_c = c_c.reorder_levels([1,0],axis=1).sort_index(axis=1)
c_c.rename_axis(['Sector','Quantity'],axis=1,inplace=True)

def treat_data_countries(data,columns,new_name,country_map):
    for j,column in enumerate(columns):
        data[column] = data[column].replace(country_map.set_index('country')['region_1'])
        data = data.replace('TWN','Asia Eastern')
        data = data.replace('ROW','Africa')
        data = data.replace('Africa Northern','Africa')
        data = data.replace('Africa Southern','Africa')
        data = data.replace('America Central','America Northern')
        data = data.replace('America Central','America Northern')
        data = data.replace('Europe Southern','Europe Western')
        data = data.replace('Asia Central','Asia Eastern')
        data = data.replace('Zealand New and Australia','Asia South-Eastern')
        data = data.replace('Asia South-eastern','Asia South-Eastern')
        data = data.replace('Africa Southern','Africa')
        data = data.replace('Asia Western','Middle East')
        data.rename(columns={column:new_name[j]},inplace = True)
    return data


# c_c = c_c.join(pd.concat({'Production':c_c.groupby(level=0).sum().xs('Trade flow (Mio$)',axis=1,level=1)},axis=1).reorder_levels([1,0],axis=1), how = 'outer')

c_c = pd.concat({'Production':c_c.groupby(level=0).sum().xs('Trade flow (Mio$)',axis=1,level=1)},axis=1).reorder_levels([1,0],axis=1).join(c_c, how = 'outer')
c_c.rename_axis(['Sector','Quantity'],axis = 1, inplace = True)

see = treat_data_countries(world,['country'],['Exporter'],country_map)
see = see[see['Exporter'].isin(c_c.index.get_level_values(0).drop_duplicates().to_list())]
see = see.set_index('Exporter').groupby(level=0).mean()
# see = pd.concat([see]*len(see)).sort_index()

for r in see.index.get_level_values(0):
    print(r)
    c_c.loc[r,'Longitude'] = see.loc[r,'longitude']
    c_c.loc[r,'Latitude'] = see.loc[r,'latitude']

# test = c_c.reset_index().merge(pd.concat({'':see},axis=1).reorder_levels([1,0],axis=1), how = 'left', on = ('Exporter',''))
# test = pd.concat({' ':see.set_index('Exporter')},axis=1).reorder_levels([1,0],axis=1).join(c_c, how = 'inner')

test = c_c.copy()

# test['longitude'] = see['longitude']

# firsts = c_c.index.get_level_values('Exporter')
# c_c['Production'] = c_c.groupby(level=0).sum().loc[firsts].values

# # c_c = c_c.sort_values('value',ascending=False).iloc[:100]
# c_co2_intensity = sol_all[y].co2_prod.groupby(level=0).sum().value / sol_all[y].output.groupby(level=0).sum().value
# c_co2_intensity = pd.DataFrame(c_co2_intensity).reset_index()
# c_co2_intensity.columns = ['row_country','co2_intensity']

# world['country'] = world.country.replace(country_map.set_index('country')['region_1'])
# world = world.groupby(['country']).mean().reset_index()

# # pos = {}
# # for country in country_list:
# #     if country != 'ROW':
# #         pos[country] = (world.longitude.loc[country],world.latitude.loc[country])
# #     else:
# #         pos[country] = (0,0)
# # pos = [(world.loc[country].latitude,world.loc[country].longitude) if country != 'ROW' else (0,0) for country in country_list]

# c_c = c_c.reset_index().merge(world,how='left',left_on='row_country',right_on ='country').drop('country',axis=1)
# # c_c = c_c.merge(c_co2_intensity,how='left',on='row_country')

# c_c.loc[c_c.row_country == 'Africa',['longitude']] = 0
# c_c.loc[c_c.row_country == 'Africa',['latitude']] = 0

# # c_c.loc[c_c.row_country == 'Asia Eastern',['longitude']] = 115
# # c_c.loc[c_c.row_country == 'Asia Eastern',['latitude']] = 35

# c_c = c_c.merge(c_c.groupby('row_country')[['value','new']].sum(),how='left',on='row_country',suffixes = ['','_total'])

# # c_c = c_c.replace('Zealand New and Australia','Oceania')

# c_c['labels-edge'] = ((c_c['hat']-1)*100).round(1)

# emissions_traded_goods = traded[y].groupby(level=[0,1]).sum().reset_index()
# emissions_traded_goods = emissions_traded_goods.merge((sol_all[y].co2_intensity/1e6).reset_index(),how='left',left_on='row_country',right_on ='country', suffixes = ['_traded_goods','_co2_intensity']).drop('country',axis=1)
# emissions_traded_goods['value'] = emissions_traded_goods['value_traded_goods'] * emissions_traded_goods['value_co2_intensity']
# emissions_traded_goods['new'] = emissions_traded_goods['new'] * emissions_traded_goods['value_co2_intensity']
# emissions_traded_goods = emissions_traded_goods.groupby('row_country')[['value','new']].sum().reset_index()

# emissions_traded_goods = traded[y].reset_index()

# emissions_traded_goods = emissions_traded_goods.merge((sol_all[y].co2_intensity/1e6).reset_index(),
#                                                       how='left',
#                                                       left_on=['row_country','row_sector'],
#                                                       right_on =['country','sector'], 
#                                                       suffixes = ['_traded_goods','_co2_intensity']).drop('country',axis=1)

# emissions_traded_goods['value'] = emissions_traded_goods['value_traded_goods'] * emissions_traded_goods['value_co2_intensity']

# emissions_traded_goods['new'] = emissions_traded_goods['new'] * emissions_traded_goods['value_co2_intensity']

# emissions_traded_goods = emissions_traded_goods.groupby('row_country')[['value','new']].sum().reset_index()

# c_c = c_c.merge(emissions_traded_goods,how='left',on='row_country',suffixes = ['','_emissions'])

# c_c = c_c.merge(country_map,how='left',left_on='row_country',right_on ='country').drop('country',axis=1)
# c_c.loc[c_c.row_country == 'ROW',['region_1','region_2']] = 'South Africa'
# c_c.loc[c_c.row_country == 'ROW',['Continent']] = 'Africa'
# c_c.loc[c_c.row_country == 'TWN',['region_1','region_2']] = 'Eastern Asia'
# c_c.loc[c_c.row_country == 'TWN',['Continent']] = 'Asia'
# c_c.loc[c_c.row_country == 'NZL',['longitude']] = 154

c_c = c_c.reset_index().T.reset_index().T
c_c.iloc[0] = c_c.iloc[0]+' , '+c_c.iloc[1]
c_c.drop('Quantity',inplace = True)


#%% write csv

# c_c.to_csv('/Users/simonl/Documents/taff/tax_model/graphs/test_with_sectors.csv')

# c_c.reset_index().to_csv('/Users/simonl/Documents/taff/tax_model/graphs/test_with_sectors2.csv')
c_c.to_csv('/Users/simonl/Documents/taff/fair_tax/graphs/regions_with_sectors_fair_tax_2018.csv')

#%%  by country with sectors

c_c = traded[y].groupby(level=[0,1,2]).sum().reset_index().copy()

world = pd.read_csv('/Users/simonl/Documents/taff/datas/country_coord/countries_codes_and_coordinates.csv')[['Alpha-3 code','Latitude (average)', 'Longitude (average)']]
world.columns = ['country','latitude','longitude']
for column in world.columns: 
    world[column] = world[column].str.replace('"','')
    world[column] = world[column].str.replace(' ','')
    # world[column] = world[column].astype('str')
world['country'] = world['country'].astype('string')
world.drop_duplicates('country',inplace=True)
# world['country'] = world['country'].to_string()
world['latitude'] = pd.to_numeric(world['latitude'])
world['longitude'] = pd.to_numeric(world['longitude'])
# world.set_index('country',inplace=True)

sector_map = pd.read_csv('/Users/simonl/Documents/taff/datas/OECD/industry_labels_after_agg_expl.csv',sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_map['row_sector'] = sector_list
sector_map = sector_map.reset_index().set_index('row_sector')

country_map = pd.read_csv('/Users/simonl/Documents/taff/datas/OECD/country_continent.csv',sep=';')

country_map['region_1'] = country_map.region_1.str.split().apply(reversed).apply(' '.join)

# c_c = c_c.groupby(['row_country','col_country']).sum()
c_c['row_sector'] = c_c.row_sector.replace(sector_map['industry'])
c_c = c_c.groupby(['row_country','row_sector','col_country']).sum()
c_c = c_c.rename_axis(['Exporter','Sector','Importer'])

c_c['value'] = c_c['value'].round(2)
c_c['new'] = c_c['new'].round(2)

c_c.columns = ['Trade flow (Mio$)','New trade flow (Mio$)']

c_c['Trade flow change (Mio$)'] = c_c['New trade flow (Mio$)'] - c_c['Trade flow (Mio$)']
# c_c['Trade flow change (%)'] = (c_c['Trade flow (Mio$)']/c_c['New trade flow (Mio$)'] - 1)*100

c_c = c_c.reset_index().pivot(index=['Exporter','Importer'],columns='Sector', values = ['Trade flow (Mio$)','New trade flow (Mio$)','Trade flow change (Mio$)'])
# test.rename_axis(['Trade flow', 'Trade flow change (Mio$)', 'Trade flow change (%)'],axis=1)

# test = 
# pd.concat([test.groupby(axis=1,level=0).sum(),test],axis=1,names=['Total',''])
c_c = pd.concat({' Total':c_c.groupby(axis=1,level=0).sum()},axis=1).reorder_levels([1,0],axis=1).join(c_c)

c_c[pd.MultiIndex.from_product([['Trade flow change (%)'],c_c.columns.get_level_values(1).drop_duplicates()])] = (c_c['New trade flow (Mio$)']/c_c['Trade flow (Mio$)']- 1)*100
c_c = c_c.drop(['New trade flow (Mio$)'],axis=1)
c_c = c_c.reorder_levels([1,0],axis=1).sort_index(axis=1)
c_c.rename_axis(['Sector','Quantity'],axis=1,inplace=True)

def treat_data_countries(data,columns,new_name,country_map):
    for j,column in enumerate(columns):
        data[column] = data[column].replace(country_map.set_index('country')['region_1'])
        data = data.replace('TWN','Asia Eastern')
        data = data.replace('ROW','Africa')
        data = data.replace('Africa Northern','Africa')
        data = data.replace('Africa Southern','Africa')
        data = data.replace('America Central','America Northern')
        data = data.replace('America Central','America Northern')
        data = data.replace('Europe Southern','Europe Western')
        data = data.replace('Asia Central','Asia Eastern')
        data = data.replace('Zealand New and Australia','Asia South-Eastern')
        data = data.replace('Asia South-eastern','Asia South-Eastern')
        data = data.replace('Africa Southern','Africa')
        data = data.replace('Asia Western','Middle East')
        data.rename(columns={column:new_name[j]},inplace = True)
    return data


# c_c = c_c.join(pd.concat({'Production':c_c.groupby(level=0).sum().xs('Trade flow (Mio$)',axis=1,level=1)},axis=1).reorder_levels([1,0],axis=1), how = 'outer')

c_c = pd.concat({'Production':c_c.groupby(level=0).sum().xs('Trade flow (Mio$)',axis=1,level=1)},axis=1).reorder_levels([1,0],axis=1).join(c_c, how = 'outer')
c_c.rename_axis(['Sector','Quantity'],axis = 1, inplace = True)

# see = treat_data_countries(world,['country'],['Exporter'],country_map)
# see = see[see['Exporter'].isin(c_c.index.get_level_values(0).drop_duplicates().to_list())]
# see = see.set_index('Exporter').groupby(level=0).mean()
# see = pd.concat([see]*len(see)).sort_index()

for c in c_c.index.get_level_values(0).drop_duplicates():
    if c == 'TWN':
        c_c.loc[c,'Longitude'] = 121
        c_c.loc[c,'Latitude'] = 25
    elif c == 'ROW':       
        c_c.loc[c,'Longitude'] = 0
        c_c.loc[c,'Latitude'] = 0
    else:
        c_c.loc[c,'Longitude'] = world.set_index('country').loc[c,'longitude']
        c_c.loc[c,'Latitude'] = world.set_index('country').loc[c,'latitude']

# test = c_c.reset_index().merge(pd.concat({'':see},axis=1).reorder_levels([1,0],axis=1), how = 'left', on = ('Exporter',''))
# test = pd.concat({' ':see.set_index('Exporter')},axis=1).reorder_levels([1,0],axis=1).join(c_c, how = 'inner')

test = c_c.copy()

# test['longitude'] = see['longitude']

# firsts = c_c.index.get_level_values('Exporter')
# c_c['Production'] = c_c.groupby(level=0).sum().loc[firsts].values

# # c_c = c_c.sort_values('value',ascending=False).iloc[:100]
# c_co2_intensity = sol_all[y].co2_prod.groupby(level=0).sum().value / sol_all[y].output.groupby(level=0).sum().value
# c_co2_intensity = pd.DataFrame(c_co2_intensity).reset_index()
# c_co2_intensity.columns = ['row_country','co2_intensity']

# world['country'] = world.country.replace(country_map.set_index('country')['region_1'])
# world = world.groupby(['country']).mean().reset_index()

# # pos = {}
# # for country in country_list:
# #     if country != 'ROW':
# #         pos[country] = (world.longitude.loc[country],world.latitude.loc[country])
# #     else:
# #         pos[country] = (0,0)
# # pos = [(world.loc[country].latitude,world.loc[country].longitude) if country != 'ROW' else (0,0) for country in country_list]

# c_c = c_c.reset_index().merge(world,how='left',left_on='row_country',right_on ='country').drop('country',axis=1)
# # c_c = c_c.merge(c_co2_intensity,how='left',on='row_country')

# c_c.loc[c_c.row_country == 'Africa',['longitude']] = 0
# c_c.loc[c_c.row_country == 'Africa',['latitude']] = 0

# # c_c.loc[c_c.row_country == 'Asia Eastern',['longitude']] = 115
# # c_c.loc[c_c.row_country == 'Asia Eastern',['latitude']] = 35

# c_c = c_c.merge(c_c.groupby('row_country')[['value','new']].sum(),how='left',on='row_country',suffixes = ['','_total'])

# # c_c = c_c.replace('Zealand New and Australia','Oceania')

# c_c['labels-edge'] = ((c_c['hat']-1)*100).round(1)

# emissions_traded_goods = traded[y].groupby(level=[0,1]).sum().reset_index()
# emissions_traded_goods = emissions_traded_goods.merge((sol_all[y].co2_intensity/1e6).reset_index(),how='left',left_on='row_country',right_on ='country', suffixes = ['_traded_goods','_co2_intensity']).drop('country',axis=1)
# emissions_traded_goods['value'] = emissions_traded_goods['value_traded_goods'] * emissions_traded_goods['value_co2_intensity']
# emissions_traded_goods['new'] = emissions_traded_goods['new'] * emissions_traded_goods['value_co2_intensity']
# emissions_traded_goods = emissions_traded_goods.groupby('row_country')[['value','new']].sum().reset_index()

# emissions_traded_goods = traded[y].reset_index()

# emissions_traded_goods = emissions_traded_goods.merge((sol_all[y].co2_intensity/1e6).reset_index(),
#                                                       how='left',
#                                                       left_on=['row_country','row_sector'],
#                                                       right_on =['country','sector'], 
#                                                       suffixes = ['_traded_goods','_co2_intensity']).drop('country',axis=1)

# emissions_traded_goods['value'] = emissions_traded_goods['value_traded_goods'] * emissions_traded_goods['value_co2_intensity']

# emissions_traded_goods['new'] = emissions_traded_goods['new'] * emissions_traded_goods['value_co2_intensity']

# emissions_traded_goods = emissions_traded_goods.groupby('row_country')[['value','new']].sum().reset_index()

# c_c = c_c.merge(emissions_traded_goods,how='left',on='row_country',suffixes = ['','_emissions'])

# c_c = c_c.merge(country_map,how='left',left_on='row_country',right_on ='country').drop('country',axis=1)
# c_c.loc[c_c.row_country == 'ROW',['region_1','region_2']] = 'South Africa'
# c_c.loc[c_c.row_country == 'ROW',['Continent']] = 'Africa'
# c_c.loc[c_c.row_country == 'TWN',['region_1','region_2']] = 'Eastern Asia'
# c_c.loc[c_c.row_country == 'TWN',['Continent']] = 'Asia'
# c_c.loc[c_c.row_country == 'NZL',['longitude']] = 154

c_c = c_c.reset_index().T.reset_index().T
c_c.iloc[0] = c_c.iloc[0]+' , '+c_c.iloc[1]
c_c.drop('Quantity',inplace = True)

#%% write csv

# c_c.to_csv('/Users/simonl/Documents/taff/tax_model/graphs/test_with_sectors.csv')

# c_c.reset_index().to_csv('/Users/simonl/Documents/taff/tax_model/graphs/test_with_sectors2.csv')
c_c.to_csv('/Users/simonl/Documents/taff/fair_tax/graphs/countries_with_sectors_fair_tax_2018.csv')
