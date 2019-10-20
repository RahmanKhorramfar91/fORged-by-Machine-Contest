# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:34:26 2019

@author: Rahman Khorramfar
"""
import pandas as pd;
import numpy as np;
#import matplotlib.pyplot as plt;
import Functions as fc;
import datetime
start = datetime.datetime.now()
#%% Fetch Data and show
Data = pd.read_csv('Ten-Year-Demand.csv', header = 0);
#Data = Data[['month','x']];
#Data.x.plot();
#plt.show();
Data = Data['x'].values.tolist();

Next_24_Months = pd.read_csv('Next-24-Months.csv', header = 0);
Next_24_Months = Next_24_Months['x'].values.tolist();

Data.extend(Next_24_Months);
#%% Problem Data
unit_holding_cost= 1;
unit_holding_cost2 = 2;
unit_backlog_cost = 3;

Forecasting_Period = 24; #month

Tree = [1,7,3,1];
Inv0 = 73; Qprev = 0; Bprev = 0;

Holding_Cost = [];
Backlog_Cost = [];

if(Inv0<=90):
    Holding_Cost.append(Inv0*unit_holding_cost);
else:
    Holding_Cost.append(Inv0*unit_holding_cost2);
Backlog_Cost.append(Bprev*unit_backlog_cost);
Invs = [];
Bs = [];Qs = [];
Invs.append(Inv0);

#%% Main Loop
for t in range(Forecasting_Period):

    Demand, Probs = fc.Create_Scenario_Tree(Tree, Data[:120+t]);

    qq = fc.Solve_Scenario_Tree(Tree,Inv0,Bprev,Demand,Probs);

    Inv0,Bprev = fc.Update_Inventory_System(qq, Inv0, Bprev, Data[120+t]);
    Qprev = qq;

    if(Inv0<=90):
        Holding_Cost.append(Inv0*unit_holding_cost);
    else:
        Holding_Cost.append(Inv0*unit_holding_cost2);

    Backlog_Cost.append(Bprev*unit_backlog_cost);

    Invs.append(Inv0);Bs.append(Bprev);Qs.append(Qprev);

    End = datetime.datetime.now();
    Elapsed = End-start;
    #print('Elapsed Time: {0} seconds'.format(Elapsed.seconds));
    print('Iteration: {0}, Elapsed Time: {1}'.format( t+1,Elapsed.seconds));



End = datetime.datetime.now();
Elapsed = End-start;
print('Total Elapsed Time: {0} seconds'.format(Elapsed.seconds));

#%% Print the results in an Excel filef
fc.Print_Excel(Holding_Cost,Backlog_Cost,Tree,Next_24_Months,Qs,Invs,Elapsed.seconds);

