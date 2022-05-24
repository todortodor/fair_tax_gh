#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:39:51 2022

@author: simonl
"""

import pandas as pd
import numpy as np
import solver_funcs as s
from tqdm import tqdm

def load_baseline(year):
    # path = '/Users/simonl/Documents/taff/datas/OECD/yearly_CSV_agg_treated/datas'
    path = 'data/yearly_CSV_agg_treated/datas'
    cons = pd.read_csv (path+year+'/consumption_'+year+'.csv')
    iot = pd.read_csv (path+year+'/input_output_'+year+'.csv')
    output = pd.read_csv (path+year+'/output_'+year+'.csv')
    va = pd.read_csv (path+year+'/VA_'+year+'.csv')
    co2_intensity = pd.read_csv(path+year+'/co2_intensity_prod_with_agri_ind_proc_fug_'+year+'.csv')
    co2_prod = pd.read_csv(path+year+'/prod_CO2_with_agri_agri_ind_proc_fug_'+year+'.csv')
    # labor = pd.read_csv('/Users/simonl/Documents/taff/datas/World bank/labor_force/labor.csv')
    labor = pd.read_csv('data/World bank/labor_force/labor.csv')
    
    # get sectors/countries lists
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
    
    return cons, iot, output, va, co2_prod, co2_intensity

def shares(cons, iot, output, va, co2_prod, co2_intensity):
        
        sh = {}    
    
        sector_list = iot.index.get_level_values(1).drop_duplicates().to_list()
        S = len(sector_list)
        country_list = iot.index.get_level_values(0).drop_duplicates().to_list()
        C = len(country_list)
        
        # compute shares
        share_cs_o = iot.div( iot.groupby(level=[1,2,3]).sum() ).reorder_levels([3,0,1,2]).fillna(0)
        share_cs_o.sort_index(inplace = True)
        share_cons_o = cons.div( cons.groupby(level=[1,2]).sum() ).reorder_levels([2,0,1]).fillna(0)
        share_cons_o.sort_index(inplace = True)
        deficit = cons.groupby(level=2).sum() - va.groupby(level=0).sum()
        # va_share = va.div(va.groupby(level = 0).sum())
        
        # transform in numpy array
        sh['cons_np'] = cons.value.values.reshape(C,S,C)
        sh['iot_np'] = iot.value.values.reshape(C,S,C,S)
        sh['output_np'] = output.value.values.reshape(C,S)
        sh['co2_intensity_np'] = co2_intensity.values.reshape(C,S) 
        sh['co2_prod_np'] = co2_prod.values.reshape(C,S)
        # gamma_labor_np = gamma_labor.values.reshape(C,S) 
        # gamma_sector_np = gamma_sector.values.reshape(S,C,S) 
        sh['share_cs_o_np'] = share_cs_o.values.reshape(C,S,C,S)
        sh['share_cons_o_np' ]= share_cons_o.values.reshape(C,S,C)
        sh['va_np']= va.value.values.reshape(C,S)
        # va_share_np = va_share.value.values.reshape(C,S)
        sh['deficit_np'] = deficit.value.values
        sh['cons_tot_np' ]= cons.groupby(level=2).sum().value.values  
        
        return sh

class sol:
    def __init__(self,y,dir_num,carb_cost):
        year = str(y)
        
        # path = '/Users/simonl/Documents/taff/tax_model/results/'+year+'_'+str(dir_num)
        path = 'results/'+year+'_'+str(dir_num)
            
        runs = pd.read_csv(path+'/runs')
        
        run = runs.iloc[np.argmin(np.abs(runs['carb_cost'] - carb_cost))]
        
        sigma = run.sigma
        eta = run.eta
        num = run.num
        carb_cost = run.carb_cost       
        
        # path = '/Users/simonl/Documents/taff/datas/OECD/yearly_CSV_agg_treated/datas'
        path = 'data/yearly_CSV_agg_treated/datas'
        cons = pd.read_csv (path+year+'/consumption_'+year+'.csv')
        iot = pd.read_csv (path+year+'/input_output_'+year+'.csv')
        output = pd.read_csv (path+year+'/output_'+year+'.csv')
        va = pd.read_csv (path+year+'/VA_'+year+'.csv')
        co2_intensity = pd.read_csv(path+year+'/co2_intensity_prod_with_agri_ind_proc_fug_'+year+'.csv')
        co2_prod = pd.read_csv(path+year+'/prod_CO2_with_agri_agri_ind_proc_fug_'+year+'.csv')
        # labor = pd.read_csv('/Users/simonl/Documents/taff/datas/World bank/labor_force/labor.csv')
        labor = pd.read_csv('data/World bank/labor_force/labor.csv')
        
        # get sectors/countries lists
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
        
        # cons['value'] = cons['value'] / num
        # iot['value'] = iot['value'] / num
        # output['value'] = output['value'] / num
        # va['value'] = va['value'] / num
        # co2_intensity['value'] = co2_intensity['value'] * num
        
        # compute gammas
        # gamma_labor = va / output
        # gamma_sector = iot.groupby(level=[1,2,3]).sum().div(output.rename_axis(['col_country','col_sector']))
        # gamma_sector = gamma_sector.reorder_levels([2,0,1]).sort_index()
        
        # compute shares
        share_cs_o = iot.div( iot.groupby(level=[1,2,3]).sum() ).reorder_levels([3,0,1,2]).fillna(0)
        share_cs_o.sort_index(inplace = True)
        share_cons_o = cons.div( cons.groupby(level=[1,2]).sum() ).reorder_levels([2,0,1]).fillna(0)
        share_cons_o.sort_index(inplace = True)
        deficit = cons.groupby(level=2).sum() - va.groupby(level=0).sum()
        # va_share = va.div(va.groupby(level = 0).sum())
        
        # transform in numpy array
        cons_np = cons.value.values.reshape(C,S,C)
        iot_np = iot.value.values.reshape(C,S,C,S)
        output_np = output.value.values.reshape(C,S)
        co2_intensity_np = co2_intensity.values.reshape(C,S) 
        co2_prod_np = co2_prod.values.reshape(C,S)
        # gamma_labor_np = gamma_labor.values.reshape(C,S) 
        # gamma_sector_np = gamma_sector.values.reshape(S,C,S) 
        share_cs_o_np = share_cs_o.values.reshape(C,S,C,S)
        share_cons_o_np = share_cons_o.values.reshape(C,S,C)
        va_np = va.value.values.reshape(C,S)
        # va_share_np = va_share.value.values.reshape(C,S)
        deficit_np = deficit.value.values
        cons_tot_np = cons.groupby(level=2).sum().value.values

        res = pd.read_csv(run.path).set_index(['country','sector'])
        res_utility = pd.read_csv(run.path+'_util_contrib').set_index(['country'])
            
        E_hat_sol = res.output_hat.values.reshape(C,S)
        p_hat_sol = res.price_hat.values.reshape(C,S)
        
        T = res_utility.contribution.values
        
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
        
        iot_new = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , iot_np)
        
        va_new = E_hat_sol * va_np
        output_new = E_hat_sol * output_np
        
        A = va_np + np.einsum('it,itjs,itjs->js' , 
                              p_hat_sol*carb_cost*co2_intensity_np,
                              iot_hat_unit,
                              iot_np)    
        
        # B = np.einsum('itj,itj->itj',
        #               cons_hat_unit,
        #               cons_np)    
        
        K = cons_tot_np - np.einsum( 'it,it,itj,itj -> j', 
                                                  p_hat_sol , 
                                                  carb_cost*co2_intensity_np , 
                                                  cons_hat_unit , 
                                                  cons_np ) 
        
        one_over_K = np.divide(1,K) 
        
        I_hat_sol = (np.einsum('js,js -> j' , E_hat_sol,A)+ deficit_np + T*run.num)*one_over_K                                                            
                                                           
        cons_new = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , cons_np)
        
        co2_prod_new = E_hat_sol * co2_prod_np / p_hat_sol
        
        cons['new'] = cons_new.ravel()
        iot['new'] = iot_new.ravel()
        output['new'] = output_new.ravel()
        va['new'] = va_new.ravel()
        co2_prod['new'] = co2_prod_new.ravel()
        
        self.run = run
        self.res = res
        self.cons = cons
        self.iot = iot
        self.output = output
        self.va = va
        self.co2_prod = co2_prod
        self.co2_intensity = co2_intensity
        self.utility = pd.DataFrame(res_utility.utility_hat)
        self.contrib = pd.DataFrame(res_utility.contribution*run.num)


def sol_from_loaded_data(carb_cost, run, cons, iot, output, va, co2_prod, sh):
        res = pd.read_csv(run.path).set_index(['country','sector'])
        res_utility = pd.read_csv(run.path+'_util_contrib').set_index(['col_country']).rename_axis(['country'])
        sector_list = res.index.get_level_values(1).drop_duplicates().to_list()
        S = len(sector_list)
        country_list = res.index.get_level_values(0).drop_duplicates().to_list()
        C = len(country_list)

        sigma = run.sigma
        eta = run.eta
        num = run.num
        carb_cost = run.carb_cost
        
        E_hat_sol = res.output_hat.values.reshape(C,S)
        p_hat_sol = res.price_hat.values.reshape(C,S)
        
        T = res_utility.contribution.values
        
        args1 = ( eta ,
                carb_cost ,
                sh['co2_intensity_np'] ,
                sh['share_cs_o_np'])        
        iot_hat_unit = s.iot_eq_unit( p_hat_sol , *args1) 
        
        args2 = ( sigma , 
                carb_cost ,
                sh['co2_intensity_np'] ,
                sh['share_cons_o_np'])
        cons_hat_unit = s.cons_eq_unit( p_hat_sol , *args2)
        
        iot_new = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , sh['iot_np'])
        
        va_new = E_hat_sol * sh['va_np']
        output_new = E_hat_sol * sh['output_np']
        
        A = sh['va_np'] + np.einsum('it,itjs,itjs->js' , 
                              p_hat_sol*carb_cost*sh['co2_intensity_np'],
                              iot_hat_unit,
                              sh['iot_np'])    
        
        # B = np.einsum('itj,itj->itj',
        #               cons_hat_unit,
        #               cons_np)    
        
        K = sh['cons_tot_np']- np.einsum( 'it,it,itj,itj -> j', 
                                                  p_hat_sol , 
                                                  carb_cost*sh['co2_intensity_np'] , 
                                                  cons_hat_unit, 
                                                  sh['cons_np'] ) 
        
        one_over_K = np.divide(1,K) 
        
        I_hat_sol = (np.einsum('js,js -> j' , E_hat_sol,A)+ sh['deficit_np'] + T*run.num)*one_over_K                                                               
                                                                           
        cons_new = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , sh['cons_np'])
        
        taxed_price = p_hat_sol*(1+carb_cost*sh['co2_intensity_np'])
        
        price_agg_no_pow = np.einsum('it,itj->tj'
                                  ,taxed_price**(1-sigma) 
                                  ,sh['share_cons_o_np'] 
                                  )       
        price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(sigma - 1))        
        
        betas = (sh['cons_np']/(sh['cons_tot_np' ][None,None,:])).sum(axis=0)
        
        price_index = (price_agg**betas).prod(axis=0)
        
        co2_prod_new = E_hat_sol * sh['co2_prod_np'] / p_hat_sol
        
        cons['new'] = cons_new.ravel()
        iot['new'] = iot_new.ravel()
        output['new'] = output_new.ravel()
        va['new'] = va_new.ravel()
        co2_prod['new'] = co2_prod_new.ravel()
        contrib = pd.DataFrame(index = country_list, data = T*run.num, columns = ['new']).rename_axis('country')

        return cons, iot, output, va, co2_prod , price_index, contrib
            
def load_everything(year_l,carb_cost_l,dir_num):

    cons_years_l = []
    iot_years_l = []
    output_years_l = []
    va_years_l = []
    co2_prod_years_l = []
    price_index_years_l = []    
    for y in year_l:
        year = str(y)
        print('loading year'+year)
        # path = '/Users/simonl/Documents/taff/tax_model/results/'+year+'_'+str(dir_num)
        path = 'results/'+year+'_'+str(dir_num)
        runs = pd.read_csv(path+'/runs')
             
        cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b = load_baseline(year)
        sector_list = iot_b.index.get_level_values(1).drop_duplicates().to_list()
        country_list = iot_b.index.get_level_values(0).drop_duplicates().to_list()
        sh = shares(cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b)
        
        cons_l = []
        iot_l = []
        output_l = []
        va_l = []
        co2_prod_l = []
        price_index_l = []
        for carb_cost in tqdm(carb_cost_l):
            run = runs.iloc[np.argmin(np.abs(runs['carb_cost'] - carb_cost))]
            carb_cost = run.carb_cost
            cons, iot, output, va, co2_prod, price_index = sol_from_loaded_data(carb_cost, run, cons_b, iot_b, output_b, va_b, co2_prod_b, sh)
            price_index_df = pd.DataFrame(index = country_list , data = price_index , columns = ['value']).rename_axis(['country'])

            cons_l.append(cons.new)
            iot_l.append(iot.new)
            output_l.append(output.new)
            va_l.append(va.new)
            co2_prod_l.append(co2_prod.new)
            price_index_l.append(price_index_df.value)
        # cons_year = pd.concat(cons_l,keys=carb_cost_l, names=['carb_cost'])
        # iot_year = pd.concat(iot_l,keys=carb_cost_l, names=['carb_cost'])
        # output_year = pd.concat(output_l,keys=carb_cost_l, names=['carb_cost'])
        # va_year = pd.concat(va_l,keys=carb_cost_l, names=['carb_cost'])
        # co2_prod_year = pd.concat(co2_prod_l,keys=carb_cost_l, names=['carb_cost'])
        # price_index_year = pd.concat(price_index_l,keys=carb_cost_l, names=['carb_cost'])
        cons_years_l.append(pd.concat(cons_l,keys=carb_cost_l, names=['carb_cost']))
        iot_years_l.append(pd.concat(iot_l,keys=carb_cost_l, names=['carb_cost']))
        output_years_l.append(pd.concat(output_l,keys=carb_cost_l, names=['carb_cost']))
        va_years_l.append(pd.concat(va_l,keys=carb_cost_l, names=['carb_cost']))
        co2_prod_years_l.append(pd.concat(co2_prod_l,keys=carb_cost_l, names=['carb_cost']))
        price_index_years_l.append(pd.concat(price_index_l,keys=carb_cost_l, names=['carb_cost']))
        
    cons_years = pd.concat(cons_years_l,keys=year_l, names=['year'])
    iot_years = pd.concat(iot_years_l,keys=year_l, names=['year'])
    output_years = pd.concat(output_years_l,keys=year_l, names=['year'])
    va_years = pd.concat(va_years_l,keys=year_l, names=['year'])
    co2_prod_years = pd.concat(co2_prod_years_l,keys=year_l, names=['year'])
    price_index_years = pd.concat(price_index_years_l,keys=year_l, names=['year'])    
    
    return cons_years, iot_years, output_years, va_years, co2_prod_years, price_index_years

