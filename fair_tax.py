#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:17:32 2022

@author: simonl
"""


import pandas as pd
import numpy as np
import solver_funcs as s
from pathlib import Path
import os
from tqdm import tqdm
# import time
 
#%%

dir_num = 2

for y in range(2018,1994,-1):
       
    #%% load yearly data
    
    # y = 2018
    year=str(y)
    print(year)
    
    path = '/Users/simonl/Documents/taff/datas/OECD/yearly_CSV_agg_treated/datas'
    cons = pd.read_csv (path+year+'/consumption_'+year+'.csv')
    iot = pd.read_csv (path+year+'/input_output_'+year+'.csv')
    output = pd.read_csv (path+year+'/output_'+year+'.csv')
    va = pd.read_csv (path+year+'/VA_'+year+'.csv')
    co2_intensity = pd.read_csv(path+year+'/co2_intensity_prod_with_agri_ind_proc_fug_'+year+'.csv')
    co2_prod = pd.read_csv(path+year+'/prod_CO2_with_agri_agri_ind_proc_fug_'+year+'.csv')
    labor = pd.read_csv('/Users/simonl/Documents/taff/datas/World bank/labor_force/labor.csv')
    
    #%% get sectors/countries lists
    sector_list = iot['col_sector'].drop_duplicates().to_list()
    S = len(sector_list)
    country_list = iot['col_country'].drop_duplicates().to_list()
    C = len(country_list)
    
    # set_indexes
    iot.set_index(
        ['row_country','row_sector','col_country','col_sector']
        ,inplace = True)
    iot.sort_index(inplace = True)
    
    cons.set_index(
        ['row_country','row_sector','col_country']
        ,inplace = True)
    
    output.rename(columns={
        'row_country':'country'
        ,'row_sector':'sector'}
        ,inplace = True
        )
    output.set_index(['country','sector'],inplace = True)
    
    va.set_index(['col_country','col_sector'],inplace = True)
    
    co2_intensity.set_index(['country','sector'],inplace = True)
    
    co2_prod.set_index(['country','sector'],inplace = True)
    
    labor.set_index('country', inplace = True)
    labor.sort_index(inplace = True)
    
    #%%
    # scale with numeraire
    
    numeraire_type = 'wage' #'wage' or 'output'
    country_num = 'USA'
    num_index = country_list.index(country_num)
    
    if numeraire_type == 'output':
        num = output.loc[country_num].value.sum()
    if numeraire_type == 'wage':
        num = va.loc[country_num].value.sum() / labor.loc[country_num,year]
        
    
    cons['value'] = cons['value'] / num
    iot['value'] = iot['value'] / num
    output['value'] = output['value'] / num
    va['value'] = va['value'] / num
    co2_intensity['value'] = co2_intensity['value'] * num
    
    
    # compute gammas
    
    gamma_labor = va / output
    
    gamma_sector = iot.groupby(level=[1,2,3]).sum().div(output.rename_axis(['col_country','col_sector']))
    gamma_sector = gamma_sector.reorder_levels([2,0,1]).sort_index()
    
    # compute shares
    
    share_cs_o = iot.div( iot.groupby(level=[1,2,3]).sum() ).reorder_levels([3,0,1,2]).fillna(0)
    share_cs_o.sort_index(inplace = True)
    share_cons_o = cons.div( cons.groupby(level=[1,2]).sum() ).reorder_levels([2,0,1]).fillna(0)
    share_cons_o.sort_index(inplace = True)
    deficit = cons.groupby(level=2).sum() - va.groupby(level=0).sum()
    va_share = va.div(va.groupby(level = 0).sum())
    
    # transform in numpy array
    
    cons_np = cons.value.values.reshape(C,S,C)
    iot_np = iot.value.values.reshape(C,S,C,S)
    output_np = output.value.values.reshape(C,S)
    co2_intensity_np = co2_intensity.values.reshape(C,S) 
    co2_prod_np = co2_prod.values.reshape(C,S)
    gamma_labor_np = gamma_labor.values.reshape(C,S) 
    gamma_sector_np = gamma_sector.values.reshape(S,C,S) 
    share_cs_o_np = share_cs_o.values.reshape(C,S,C,S)
    share_cons_o_np = share_cons_o.values.reshape(C,S,C)
    va_np = va.value.values.reshape(C,S)
    va_share_np = va_share.value.values.reshape(C,S)
    deficit_np = deficit.value.values
    cons_tot_np = cons.groupby(level=2).sum().value.values
        
    #%%
        
    carb_cost_lin = np.linspace(0,2e-4, num=11)
    # carb_cost_log = np.logspace(-3,-2, num=11)[1:]
    # carb_cost_l = np.concatenate((carb_cost_lin,carb_cost_log))
    # carb_cost_l = [1e-4]
    eta = 4 
    sigma = 4
    
    
    
    for carb_cost in tqdm(carb_cost_lin):
    #%%    
        # carb_cost = 1e-4
        step = 1/3
        T_tol = 1e-6
        T_old = np.zeros(C)
        T_new = np.ones(C)
        it = 0
        carb_cost = carb_cost / num
        while min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()) > T_tol:
            print('iteration',it)
            if it !=0:
                T_old=T_new*step+T_old*(1-step)
            args = (
                eta,
                sigma,
                C,
                S,
                numeraire_type,
                country_num,
                num_index,
                cons_np,
                iot_np,
                output_np,
                co2_intensity_np,
                gamma_labor_np,
                gamma_sector_np,
                share_cs_o_np,
                share_cons_o_np,
                va_np,
                va_share_np,
                deficit_np + T_old,
                cons_tot_np
                )
            
            E_hat_sol , p_hat_sol = s.solve_E_p(carb_cost , *args)
            
            #compute solution quantities
            
            q_hat_sol = E_hat_sol /p_hat_sol
            
            emissions_sol = np.einsum('js,js->', q_hat_sol , co2_prod_np)
            
            # print('emissions = ',emissions_sol)
            
            args1 = ( eta ,
                    carb_cost ,
                    co2_intensity_np ,
                    share_cs_o_np)        
            iot_hat_unit = s.iot_eq_unit( p_hat_sol , *args1) 
            
            args2 = ( sigma , 
                    carb_cost ,
                    co2_intensity_np ,
                    share_cons_o_np)
            cons_hat_unit = s.cons_eq_unit( p_hat_sol , *args2)
            
            A = va_np + np.einsum('it,itjs,itjs->js' , 
                                  p_hat_sol*carb_cost*co2_intensity_np,
                                  iot_hat_unit,
                                  iot_np)      
            
            K = cons_tot_np - np.einsum( 'it,it,itj,itj -> j', 
                                                      p_hat_sol , 
                                                      carb_cost*co2_intensity_np , 
                                                      cons_hat_unit , 
                                                      cons_np ) 
            
            one_over_K = np.divide(1,K) 
            
            I_hat_sol = (np.einsum('js,js -> j' , E_hat_sol,A)+ deficit_np  + T_old)*one_over_K 
        
            
            #
            
            #compute the contribution for a fair tax
            
            beta = np.einsum('itj->tj',cons_np) / np.einsum('itj->j',cons_np)
            va_new = E_hat_sol * va_np
            
            taxed_price = p_hat_sol*(1+carb_cost*co2_intensity_np) 
            iot_new = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , iot_np)
            cons_new = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , cons_np)
            
            price_agg_no_pow = np.einsum('it,itj->tj'
                                      ,taxed_price**(1-sigma) 
                                      , share_cons_o_np 
                                      )       
            # price_agg = np.divide(1, 
            #                 price_agg_no_pow , 
            #                 out = np.ones_like(price_agg_no_pow), 
            #                 where = price_agg_no_pow!=0 ) ** (1/(sigma - 1))    
            price_agg = price_agg_no_pow ** (1/(1 - sigma))    
            
            H = cons_tot_np*(price_agg**(beta)).prod(axis=0)
            
            G = np.einsum('js->j',va_new) \
                + np.einsum('it,itjs->j',carb_cost*co2_intensity_np ,iot_new) \
                + np.einsum('it,itj->j',carb_cost*co2_intensity_np ,cons_new) \
                + deficit_np
            
            T_new = (H*G.sum()/H.sum()-G)
    
            # compute some things to write directly
                                                                  
            cons_hat_sol = np.einsum('j,itj->itj',  I_hat_sol , cons_hat_unit) 
            
            utility_cs_hat_sol = np.einsum('itj,itj->tj', 
                                            cons_hat_sol**((sigma-1)/sigma) , 
                                            share_cons_o_np ) ** (sigma / (sigma-1))
            
            beta = np.einsum('itj->tj',cons_np) / np.einsum('itj->j',cons_np)
            
            utility_c_hat_sol = (utility_cs_hat_sol**beta).prod(axis=0)
            
            utility_hat_sol = np.einsum('j,j->' , utility_c_hat_sol , cons_tot_np/(cons_tot_np.sum()))
            

            print('variance from average utility', utility_c_hat_sol.var())
            print('condition', min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()))
            
            it = it +1
            
            # T_df = pd.DataFrame(data=np.array([T_old,T_new*step+T_old*(1-step),utility_c_hat_sol]).transpose(),index=country_list,columns=['contrib_old','contrib_new','util'])
            # T_df = T_df.sort_values('util')
            # fig, ax = plt.subplots(figsize=(18,12))
            # ax1 = ax.twinx()
            # ax.plot(T_df['contrib_old'],label='contribution old')
            # ax.plot(T_df['contrib_new'],color='r',label='contribution new')
            # # ax.plot(deficit_np,color='g',label='deficit')
            # yabs_max = abs(max(ax.get_ylim(), key=abs))
            # ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            
            # ax1.plot(T_df['util']-T_df['util'].mean(),color=sns.color_palette()[1],label='utility',ls = '--')
            # yabs_max = abs(max(ax1.get_ylim(), key=abs))
            # ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            # # ax1.bar(T_df.index.to_list(),G.ravel(),color=sns.color_palette()[1])
            # # ax1.bar(T_df.index.to_list(),H*G.sum()/H.sum().ravel(),color=sns.color_palette()[1])
            # ax.set_xticklabels(T_df.index
            #                     , rotation=90
            #                     , ha='right'
            #                     , rotation_mode='anchor'
            #                     ,fontsize=15)
            # ax.legend(loc='upper center')
            # ax1.legend(loc='upper right')
            # plt.title(str(it))
            # plt.show()
            # it = it +1
        
#%% write results        
 
        path = 'results/'+year+'_'+str(dir_num)
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(path+'/runs'):
            runs = pd.DataFrame(columns = ['year',
                                            'carb_cost',
                                            'sigma',
                                            'eta',
                                            'path',
                                            'num',
                                            'num_type',
                                            'num_country',
                                            'emissions',
                                            'utility'])
            runs.to_csv(path+'/runs',index=False)
        
        runs = pd.read_csv(path+'/runs')
        run = [year,carb_cost*num,sigma,eta,path+'/carb_cost='+str(carb_cost),num,numeraire_type,country_num,emissions_sol,utility_hat_sol]
        runs.loc[len(runs)] = run
        runs.to_csv(path+'/runs',index=False)
        
        results = pd.DataFrame(index = output.index , columns = ['output_hat','price_hat'])
        results['output_hat'] = E_hat_sol.ravel()
        results['price_hat'] = p_hat_sol.ravel()
        results.to_csv(path+'/carb_cost='+str(carb_cost))
        
        util_contrib = pd.DataFrame(index = deficit.index , columns = ['utility_hat','contribution'])
        util_contrib['utility_hat'] = utility_c_hat_sol
        util_contrib['contribution'] = T_new
        util_contrib.to_csv(path+'/carb_cost='+str(carb_cost)+'_util_contrib')
        
        
        