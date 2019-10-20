# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:47:17 2019

@authors: Rahman Khorramfar and Saeed Chavoshi
"""
import numpy as np;
#import pandas as pd;
#import matplotlib.pyplot as plt;
from statsmodels.tsa.statespace.sarimax import SARIMAX; # with trend and seasonality
from statsmodels.tsa.holtwinters import ExponentialSmoothing; # with trend and seasonality
from  gurobipy import quicksum, GRB, Model;
import numpy as np;
from scipy import stats;



def SARIMA_Forecast(data, config):

    #order:  A tuple p, d, and q parameters for the modeling of the trend.
    # sesonal_order: A tuple of P, D, Q, and m parameters for the modeling the seasonality
    # trend: A parameter for controlling a model of the deterministic trend as one of ‘n’,’c’,’t’,’ct’ for no trend, constant, linear, and constant with linear trend, respectively.

	order, sorder, trend = config
	# define model
	model = SARIMAX(data, order=order, seasonal_order=sorder,
                 trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	forecast = model_fit.get_forecast();
	return forecast.predicted_mean, forecast.se_mean;

def ESM_Forecast(data):
    data = np.array(data);
    M0fit = ExponentialSmoothing(data,seasonal_periods=12, trend='add', seasonal='mul').fit(use_boxcox=False);
    forecast = M0fit.predict();

    return forecast[0];

def Demand_Aroung_Base(base, std, num):
    lst = list();

    if(num==7):
        lst.append(max(0,base-3*std));
        lst.append(max(0,base-2*std));
        lst.append(max(0,base-std));
        lst.append(base);
        lst.append(base+std);
        lst.append(base+2*std);
        lst.append(base+3*std);

    elif(num==5):
        lst.append(max(0,base-2*std));
        lst.append(max(0,base-std));
        lst.append(base);
        lst.append(base+std);
        lst.append(base+2*std);

    elif(num==3):
        lst.append(max(0,base-std));
        lst.append(base);
        lst.append(base+std);

    elif(num==1):
        lst.append(base);


    return lst;



def Predecessor_List(Tree = np.array([1,5,3,1])):
    PrNodes = list([0]);
    for t in range(1,len(Tree),1):
        num = 1; num2 = PrNodes[len(PrNodes)-1];
        for i in range(0,t+1,1):
            num = num*Tree[i];
        num2 = num+num2;
        PrNodes.append(num2);

    NN =PrNodes[len(PrNodes)-1]+1;

    Pred = np.zeros(NN);

    for t in range(len(Tree)-1,1,-1):
        s1 = PrNodes[t] - PrNodes[t-1];
        s0 =  PrNodes[t-1];
        jump = 0;
        for i in range(s1):
            jump  = jump+1; NN = NN-1;
            Pred[NN] = s0;
            if (jump== Tree[t]):
                s0 = s0-1; jump = 0;
    Pred[0] = -1;

    return Pred, PrNodes;





def Cumulitive_Data(Pred, data2,Demands, Dh):
    #Dh = Pred[Dh];
    Dh = int(Dh);
    rho = list();
    while(Dh!=0):
        rho.append(Dh);
        Dh = int(Pred[Dh]);

    rho = sorted(rho);
    for i in range(len(rho)):
        data2.append(Demands[rho[i]]);


    return data2;



def Distribution_Matching(D):
    mean = np.mean(D);
    moment = [np.var(D), stats.skew(D), stats.kurtosis(D)];

    N = len(D);
    T1 = range(N);
    K = range(len(moment));  #number of moments

    W = np.array([0.5,0.35,0.15]);
    Omega = np.random.uniform(0,1,(1,N));
    osum = np.sum(Omega);
    Omega= [i/osum  for i in Omega[0]];
    # Gurobi Model: Define variables
    ProbModel = Model();
    P = ProbModel.addVars(N, lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = "P");
    S1 = ProbModel.addVars(N, lb = 0, vtype = GRB.CONTINUOUS, name = "S1");
    S2 = ProbModel.addVars(N, lb = 0, vtype = GRB.CONTINUOUS, name = "S2");

    M1 = ProbModel.addVars(K, lb = 0, vtype = GRB.CONTINUOUS, name = "M1");
    M2 = ProbModel.addVars(K, lb = 0, vtype = GRB.CONTINUOUS, name = "M2");

    #  Objective Function
    Z = quicksum(W[k]*(M1[k]+M2[k]) for k in K[1:]) + quicksum(Omega[j]*(S1[j]+S2[j]) for j in T1)
    ProbModel.setObjective(Z, GRB.MINIMIZE);

    # Define Constraints

    ProbModel.addConstr(quicksum(P[j] for j in T1) == 1);
    ProbModel.addConstr(quicksum(D[j]*P[j] for j in T1) == mean);
    ProbModel.addConstrs(quicksum((D[j] -mean)**(k + 2) * P[j] for j in T1) + M1[k] - M2[k] == moment[k] for k in K);
    ProbModel.addConstrs(stats.norm.cdf((D[j]- mean)/moment[0])
                          - quicksum(P[jp] for jp in T1)== S1[j]-S2[j] for j in T1)
    ProbModel.addConstrs(P[j]>= 0.1 for j in T1);
    # Solve and publish
    ProbModel.Params.OutputFlag = 0;
    ProbModel.optimize();

    return ProbModel.x[:N];




def Create_Scenario_Tree(Tree, Data):
    config = [(1,0,2),(2,0,1,12),'n'];
    Pred, PrNodes = Predecessor_List(Tree);

    #Data = pd.read_csv('Ten-Year-Demand.csv', header = 0);
    #data = Data['x'].values.tolist();  # directly convery dataframe to numpy array
    data = Data;
    TreeLength = len(Tree);
    base, std = SARIMA_Forecast(data, config);
    Demands = list([data[-1]]);
    T1b = list(base);
    T1s = list(std);

    Prob = [1];
    ct1 = 0;
    #
    for t in range(1,TreeLength,1):
        T0b = T1b.copy();
        T0s = T1s.copy();
        T1b.clear(); T1s.clear();
        Dh = PrNodes[t-1];
        for p in range(len(T0s)):

            NewBases = Demand_Aroung_Base(T0b[p],T0s[p],Tree[t]);
            ProbRoot = Prob[ct1];
            if(len(NewBases) > 1):
                Pvals = Distribution_Matching(NewBases);
            else:
                Pvals = [1];
            for i in range(len(Pvals)):
                Prob.append(ProbRoot*Pvals[i]);
            #print(Prob);
            ct1 = ct1+1;


            for i in range(len(NewBases)):
                Demands.append(NewBases[i]);

            for i in range(len(NewBases)):
                Dh = Dh+1;
                data2 = data.copy();
                cdata = Cumulitive_Data(Pred,data2,Demands,Dh);
                #print(cdata[-3:])
                base, std = SARIMA_Forecast(cdata, config);
                T1b.append(base[0]); T1s.append(std[0]);

    return Demands, Prob;


def Helper4_Solve_Scenario_Tree(Tree, Demand, Prob):

    nS = np.prod(Tree);

    P = Prob[-nS:];

    D = np.zeros((nS,len(Tree)));
    Eta = list([]);
    for t in range(len(Tree)):
        s1 = 1;
        for t2 in range(t+1):
            s1 = s1*Tree[t2];
        Eta.append(s1);

    Pi = list([]);
    for t in range(len(Eta)):
        Pi.append((int)(nS/Eta[t]));

    ct = -1;
    for t in range(len(Tree)):
        ct2=0;
        for r in range(Eta[t]):
            ct = ct+1;
            for j in range(Pi[t]):
                D[ct2,t] = Demand[ct];
                ct2 = ct2+1;
    return D, P,Eta, Pi;




def Solve_Scenario_Tree(Tree,Inv0,Bprev,Demand,Probs):
    nS = np.prod(Tree);
    D, P,Eta, Pi = Helper4_Solve_Scenario_Tree(Tree, Demand, Probs);
    #print(D);
    epsilon = 0.0001;
    l = len(Tree);
    M = 100000;
    T1 = range(l);
    T2 = range(l + 1);
    T3 = range(l + 1);
    T4 = range(l - 1);
    S = range(nS);
    #
    OptModel = Model()

    I = OptModel.addVars(T2,S, lb = 0, vtype = GRB.CONTINUOUS, name = "I");
    B = OptModel.addVars(T3,S, lb = 0, vtype = GRB.CONTINUOUS, name = "B");
    W = OptModel.addVars(T2,S, lb = 0, vtype = GRB.CONTINUOUS, name = "W");
    Q = OptModel.addVars(T1,S, lb = 0, vtype = GRB.CONTINUOUS, name = "Q");
    Y = OptModel.addVars(T2,S, vtype = GRB.BINARY, name = "Y");

    #Z = quicksum(P[s]*(2*I[t,s] - W[t,s] + 3*B[t,s]) for t in T2 for s in S);
    Z = quicksum(P[s]*(2*I[t+1,s]-W[t+1,s]) for t in T1 for s in S)+quicksum(P[s]*3*B[t,s] for t in T1 for s in S);
    #
    OptModel.setObjective(Z, GRB.MINIMIZE);

    OptModel.addConstrs(I[t,s] >= (90+epsilon)*(1 - Y[t,s]) for t in T2 for s in S);
    OptModel.addConstrs(I[t,s] <= 90 + M*(1 - Y[t,s]) for t in T2 for s in S);
    OptModel.addConstrs(W[t,s] <= M*Y[t,s] for t in T2 for s in S);
    OptModel.addConstrs(W[t,s] <= I[t,s] for t in T2 for s in S);
    OptModel.addConstrs(W[t,s] >= I[t,s] - M*(1 - Y[t,s]) for t in T2 for s in S);
    OptModel.addConstrs(I[0,s] == Inv0 for s in S);
    OptModel.addConstrs(B[0,s] == Bprev for s in S);
    #OptModel.addConstrs(I[0,s] >= Inv0-Bprev for s in S);
    #OptModel.addConstrs(B[0,s] >= Bprev-Inv0 for s in S);
    #OptModel.addConstrs(I[0,s] == D[s][0]+Qprev - B[0,s] for s in S);
    OptModel.addConstrs(I[t + 1,s]==I[t,s]+Q[t,s]-D[s][t+1]+B[t + 1,s]-B[t,s] for t in T4 for s in S);


    # non-anticipitivity constraints
    for t in T2:
        if(t >= len(Pi) or  Pi[t]==1):continue;
        for c in range(Eta[t]):
            for s in range(Pi[t]-1):
                OptModel.addConstr(I[t,c*Pi[t]+s] == I[t,c*Pi[t]+s+1]);
                OptModel.addConstr(W[t,c*Pi[t]+s] == W[t,c*Pi[t]+s+1]);
                OptModel.addConstr(Y[t,c*Pi[t]+s] == Y[t,c*Pi[t]+s+1]);
                OptModel.addConstr(B[t,c*Pi[t]+s] == B[t,c*Pi[t]+s+1]);
                OptModel.addConstr(Q[t,c*Pi[t]+s] == Q[t,c*Pi[t]+s+1]);


    #
    OptModel.Params.OutputFlag = 0;
    OptModel.optimize();
    #OptModel.write("MCLP.sol");
    #print("Optimal Solution: ", OptModel.ObjVal);
    ii = []; qq= []; bb= []; demands_in_t= [];
    for s in range(nS):

        ix = I[1,s].x;
        if(ix in ii): continue;
        demands_in_t.append(D[s,1]);
        ii.append(I[1,s].x);
        qq.append(Q[0,s].x);
        bb.append(B[1,s].x);
    #return ii, qq, bb,demands_in_t;
    return qq[0];

def Index_of_Closest(arr, value):
    arr = np.array(arr);
    idx = (np.abs(arr-value)).argmin()
    return idx;


def Update_Inventory_System(Q, Inv, B, RealDemand):

    Inv_inHand  = Q+Inv-B-RealDemand;

    if Inv_inHand<0:
        B = abs(Inv_inHand); Inv = 0;
    else:
        Inv = Inv_inHand; B = 0;


    return Inv, B;


def Print_Excel(Holding_Cost,Backlog_Cost,Tree,Next_24_Months,Qs,Invs, Elapsed):

    Forecasting_Period = len(Qs);
    total_holding = np.sum(Holding_Cost);
    total_holding = float(total_holding);
    total_backlog = np.sum(Backlog_Cost);
    total_backlog = float(total_backlog);

    import xlwt;
    from xlwt import Workbook

    wb = Workbook();
    # add_sheet is used to create sheet.
    #Name = 'Tree'+str(Tree[1])+str(Tree[2])+str(Tree[3])+'.xls';
    Name = 'Results.xls';
    sheet1 = wb.add_sheet('Results');
    #sheet1.set_col_default_width = 5000;
    sheet1.col(1).width = 2000;
    sheet1.col(2).width = 4500;
    sheet1.col(3).width = 4500;


    style0 = xlwt.easyxf('font: bold 1 , color red, height 200;align: horiz center');
    style1 = xlwt.easyxf('font: bold 1 , color blue, height 200;align: horiz center');
    style3 = xlwt.easyxf('font: bold 1 , height 200;align: horiz center');
    style4 = xlwt.easyxf('align: horiz center');

    sheet1.write(0,1,'Month',style3)
    sheet1.write(0,2,'Realized Demand',style0)
    #style = xlwt.easyxf('font: bold 1, color red, height 200');
    sheet1.write(0,3,'Order Quantity', style0)
    sheet1.col(4).width = 3000;
    sheet1.col(5).width = 3000;
    sheet1.col(6).width = 3000;
    sheet1.col(7).width = 3000;
    sheet1.write(0,4,'Initial Inv', style4);
    sheet1.write(0,5,'Ending Inv', style4);
    sheet1.write(1,4,Invs[0], style4);
    sheet1.write(1,5,Invs[0], style4);
    sheet1.write(0,6,'Holding Cost', style4);
    sheet1.write(0,7,'Backlog Cost', style4);


    sheet1.col(8).width = 5000;
    sheet1.write(0,8,'Total Holding Cost', style1)
    sheet1.write(1,8,total_holding, style4)
    sheet1.col(9).width = 5000;
    sheet1.write(0,9,'Total Back Order Cost', style1)
    sheet1.write(1,9,total_backlog, style4)
    sheet1.col(10).width = 3000;
    sheet1.write(0,10,'Total Cost', style1)
    sheet1.write(1,10,total_backlog+total_holding, style4)
    sheet1.write(3,9,'Elapsed Time(sec)',style1);
    sheet1.write(3,10,Elapsed,style4);


    sheet1.write(2,0,2006,style4);
    sheet1.write(14,0,2007,style4);

    for t in range(Forecasting_Period):


        sheet1.write(t+2,1, t+1, style4);
        sheet1.write(t+2,2, Next_24_Months[t],style4);
        sheet1.write(t+1,3, Qs[t],style4);
        sheet1.write(t+2,4,Invs[t]+Qs[t], style4);
        sheet1.write(t+2,5,Invs[t+1], style4);
        sheet1.write(t+1,6,Holding_Cost[t],style4);
        sheet1.write(t+1,7, Backlog_Cost[t],style4);
    wb.save(Name)










