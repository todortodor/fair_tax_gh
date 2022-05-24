#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:23:32 2022

@author: simonl
"""
import numpy as np
import matplotlib.pyplot as plt

def cons_eq_unit( price, *args ):    
    ( sigma , carb_cost , co2_intensity_np , share_cons_o_np ) = args
    
    taxed_price = price*(1+carb_cost*co2_intensity_np)
    
    price_agg_no_pow = np.einsum('it,itj->tj'
                          ,taxed_price**(1-sigma) 
                          , share_cons_o_np 
                          )
    
    Q = np.einsum('tj,it -> itj' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) ,  
                  taxed_price ** (-sigma))
    return Q   

def iot_eq_unit( price , *args ):    
    ( eta , carb_cost , co2_intensity_np , share_cs_o_np ) = args
    
    taxed_price = price*(1+carb_cost*co2_intensity_np)
    
    price_agg_no_pow = np.einsum('it,itjs->tjs'
                          ,taxed_price**(1-eta) 
                          , share_cs_o_np 
                          )
    
    M = np.einsum('tjs,it -> itjs' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) , 
                  taxed_price ** (-eta))
    return M 

def solve_p(E , carb_cost , *args):
    
    (eta ,
    sigma ,
    C ,
    S ,
    numeraire_type ,
    country_num ,
    num_index ,
    cons_np ,
    iot_np ,
    output_np ,
    co2_intensity_np ,
    gamma_labor_np ,
    gamma_sector_np ,
    share_cs_o_np ,
    share_cons_o_np ,
    va_np ,
    va_share_np ,
    deficit_np , 
    cons_tot_np) = args
    
    tol_p = 1e-8
    p_step = 1
       
    price_new = np.ones(C*S).reshape(C,S)
    price_old = np.zeros((C,S))    
    
    while np.linalg.norm(price_new - price_old)/np.linalg.norm(price_new) > tol_p:        
        price_old = (p_step * price_new + (1-p_step) * price_old)       
        taxed_price = price_old*(1+carb_cost*co2_intensity_np)        
        price_agg_no_pow = np.einsum('it,itjs->tjs'
                                  ,taxed_price**(1-eta) 
                                  , share_cs_o_np 
                                  )       
        price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(eta - 1))            
        prod = ( price_agg ** gamma_sector_np ).prod(axis = 0)       
        wage_hat = np.einsum('js,js->j', E , va_share_np )       
        price_new = wage_hat[:,None]**gamma_labor_np * prod
    
    return price_new

# # deficit_hat = I_hat
# def solve_E(carb_cost , *args):
    
#     (eta ,
#     sigma ,
#     C ,
#     S ,
#     numeraire_type ,
#     country_num ,
#     num_index ,
#     cons_np ,
#     iot_np ,
#     output_np ,
#     co2_intensity_np ,
#     gamma_labor_np ,
#     gamma_sector_np ,
#     share_cs_o_np ,
#     share_cons_o_np ,
#     va_np ,
#     va_share_np ,
#     deficit_np , 
#     cons_tot_np) = args
        
#     E_step = 2/3
#     numeric_tol = 1e-15
#     tol_E = 1e-15
    
#     count = 0
#     condition = True
#     window = 16

#     E_new = np.ones(C*S).reshape(C,S)
#     E_old = np.zeros((C,S))
#     convergence = np.ones(1)
    
#     while condition:
    
#         E_old = (E_step * E_new + (1-E_step) * E_old)
        
#         price = solve_p(E_old , carb_cost , *args)
            
#         args1 = ( eta ,
#                     carb_cost ,
#                     co2_intensity_np ,
#                     share_cs_o_np
#                     )        
#         iot_hat_unit = iot_eq_unit( price , *args1) 
        
#         args2 = ( sigma , 
#                 carb_cost ,
#                 co2_intensity_np ,
#                 share_cons_o_np)        
#         cons_hat_unit = cons_eq_unit( price , *args2)    
           
#         A = va_np + np.einsum('it,itjs,itjs->js' , 
#                               price*carb_cost*co2_intensity_np,
#                               iot_hat_unit,
#                               iot_np)    
#         B = np.einsum('itj,itj->itj',
#                       cons_hat_unit,
#                       cons_np)    
#         K = cons_tot_np - deficit_np - np.einsum( 'it,it,itj,itj -> j', 
#                                                   price , 
#                                                   carb_cost*co2_intensity_np , 
#                                                   cons_hat_unit , 
#                                                   cons_np )    
#         Z = np.einsum('itjs,itjs->itjs',
#                       iot_hat_unit,
#                       iot_np)        
#         one_over_K = np.divide(1, 
#                         K , 
#                         out = np.ones_like(K) * numeric_tol, 
#                         where=K!=numeric_tol )        
#         temp = np.einsum('itj,js,j -> itjs' , B , A , one_over_K ) + Z    
#         T = np.einsum('js,itjs -> it', E_old , temp )
        
#         E_new = price * T / output_np
        
#         E_new = E_new / np.linalg.norm(E_new)
        
#         if count == 0:
#             convergence = np.array([np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old)])
#         else:    
#             convergence = np.append( convergence , 
#                                     np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old) )
        
#         if count > window:
#             condition = np.any( convergence[-window:] > tol_E) 
        
#         count += 1
        
#         return E_new
 
# deficit_hat = 1 
def solve_E(carb_cost , *args):
    
    (eta ,
    sigma ,
    C ,
    S ,
    numeraire_type ,
    country_num ,
    num_index ,
    cons_np ,
    iot_np ,
    output_np ,
    co2_intensity_np ,
    gamma_labor_np ,
    gamma_sector_np ,
    share_cs_o_np ,
    share_cons_o_np ,
    va_np ,
    va_share_np ,
    deficit_np , 
    cons_tot_np) = args
        
    E_step = 2/3
    numeric_tol = 1e-15
    tol_E = 1e-8
    
    count = 0
    condition = True
    window = 10

    E_new = np.ones(C*S).reshape(C,S)
    E_old = np.zeros((C,S))
    convergence = np.ones(1)
    
    while condition:
    
        E_old = (E_step * E_new + (1-E_step) * E_old)
        
        price = solve_p(E_old , carb_cost , *args)
            
        args1 = ( eta ,
                    carb_cost ,
                    co2_intensity_np ,
                    share_cs_o_np
                    )        
        iot_hat_unit = iot_eq_unit( price , *args1) 
        
        args2 = ( sigma , 
                carb_cost ,
                co2_intensity_np ,
                share_cons_o_np)        
        cons_hat_unit = cons_eq_unit( price , *args2)    
           
        A = va_np + np.einsum('it,itjs,itjs->js' , 
                              price*carb_cost*co2_intensity_np,
                              iot_hat_unit,
                              iot_np)    
        B = np.einsum('itj,itj->itj',
                      cons_hat_unit,
                      cons_np)    
        K = cons_tot_np - np.einsum( 'it,it,itj,itj -> j', 
                                                  price , 
                                                  carb_cost*co2_intensity_np , 
                                                  cons_hat_unit , 
                                                  cons_np )    
        Z = np.einsum('itjs,itjs->itjs',
                      iot_hat_unit,
                      iot_np)        
        # one_over_K = np.divide(1, 
        #                 K , 
        #                 out = np.ones_like(K) * numeric_tol, 
        #                 where=K!= numeric_tol )  
        one_over_K = np.divide(1, 
                        K) 
        F = np.einsum('itj,js,j -> itjs' , B , A , one_over_K ) + Z    
        T = np.einsum('js,itjs -> it', E_old , F )
        
        # E_new = price * (T + np.einsum('j,j->',deficit_np,one_over_K)) / output_np
        E_new = price * (T + np.einsum('itj,j,j->it',B,deficit_np,one_over_K)) / output_np
        
        if np.any(np.einsum('itj,j,j->it',B,deficit_np,one_over_K) < -T):
            print('deficit too large')
        
        E_new = E_new / E_new.mean()
        
        if count == 0:
            convergence = np.array([np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old)])
        else:    
            convergence = np.append( convergence , 
                                    np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old) )
        
        if count > window:
            condition = np.any( convergence[-window:] > tol_E) 
        
        count += 1
        
        # print('diff norme ',np.linalg.norm(E_new - E_old))
        # # print('norme ',(E_new.sum()))
        # print('condition',convergence[-1])
        
        # plt.plot(convergence[-20:])
        # plt.show()
        
    return E_new

def solve_E_p(carb_cost , *args):
    (eta ,
    sigma ,
    C ,
    S ,
    numeraire_type ,
    country_num ,
    num_index ,
    cons_np ,
    iot_np ,
    output_np ,
    co2_intensity_np ,
    gamma_labor_np ,
    gamma_sector_np ,
    share_cs_o_np ,
    share_cons_o_np ,
    va_np ,
    va_share_np ,
    deficit_np , 
    cons_tot_np) = args
    
    E = solve_E(carb_cost , *args)
    
    if numeraire_type == 'output':
        norm_factor = np.einsum('s,s->', E[num_index,:] , output_np[num_index,:])
    if numeraire_type == 'wage':
        norm_factor = np.einsum('s,s->', E[num_index,:] , va_share_np[num_index,:] )
        
    E = E / norm_factor
    
    price = solve_p(E , carb_cost , *args)

    return E , price

# args = (
#         eta ,
#         sigma ,
#         C ,
#         S ,
#         numeraire_type ,
#         country_num ,
#         num_index ,
#         cons_np ,
#         iot_np ,
#         output_np ,
#         co2_intensity_np ,
#         gamma_labor_np ,
#         gamma_sector_np ,
#         share_cs_o_np ,
#         share_cons_o_np ,
#         va_np ,
#         va_share_np ,
#         deficit_np , 
#         cons_tot_np
#         )
    