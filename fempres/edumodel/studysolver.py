"""
This module provides tools for solving study policies, simulating time series, 
and generating moments based on given education model parameterisation.

Functions
---------
edu_solver_factory(og: class) -> (edu_iterate: func), (TSALL: func), (gen_moments: func)
    Given a parameterized education model class (og), this function returns:
    - An iterator (edu_iterate) solves policy functions using Bellman iter.
    - A function (TSALL) generating a time-series for N individuals in sample.
    - A function (gen_moments) to produce moments based on the simulation.

generate_study_pols(og: class) -> TxM array
    A wrapper function that accepts a parameterized education model class. 
    It solves the policies and simulates N individuals for S samples, 
    returning a panel of average moments for T weeks across all the S samples.
"""


# Import packages
import sys
from interpolation.splines import extrap_options as xto
from interpolation.splines import eval_linear
from quantecon.optimize import nelder_mead
from numba import njit
import time
import numpy as np
import gc

import warnings

warnings.filterwarnings('ignore')


def edu_solver_factory(og,verbose=False):

    """
    Generates solvers for worker policies, produces time-series, and moments.

    Parameters
    ----------
    og : EduModel class instance
        Represents the parameterized education model.

    Returns
    -------
    edu_iterate : function
        A Bellman iterator function for solving study policies.
    TSALL : function
        A function to generate a time-series of N students.
    gen_moments : function
        Generates moments based on S samples of N students. 

    Notes
    -----
    - T denotes the total number of weeks: teaching weeks (10), study week (1), 
      and final exam week (1), summing up to 12.
    - In the edu_iterate function, python-based indexing reduces all indices by 1.
    - The TSALL function uses original week indexes (not Python-based) to align 
      with external data moments.
    """

    # Unpack  paramters from class
    # See settings/settings.yml for variable definitions
    T = og.T
    N = og.N
    H = og.H
    S = og.S
    rho_E = og.rho_E
    delta = og.delta 
    d = og.d
    kappa_1 = og.kappa_1
    kappa_2 = og.kappa_2
    kappa_3 = og.kappa_3
    kappa_4 = og.kappa_4
    theta = np.array(og.theta)
    iota_c = og.iota_c
    iota_i = og.iota_i
    a = og.a

    # Unpack uniform shocks for sim.
    U = og.U # beta and es shocks
    U_z = og.U_z # zeta shocks
    
    # Grids and shocks
    MM = og.MM
    M = og.M
    Mh = og.Mh

    beta_hat = og.beta_hat
    beta_stat = og.beta_stat
    P_beta = og.P_beta
    es_hat = og.es_hat
    es_stat = og.es_stat
    P_es = og.P_es

    Q_shocks_ind = og.Q_shocks_ind
    EBA_P2 = og.EBA_P2
    
    zeta_hhat = og.zeta_hhat
    zeta_hat = og.zeta_hat
    P_zeta = og.P_zeta
    P_zetah = og.P_zetah

    # Unpack functions
    u_l = og.u_l
    S_effort_to_IM = og.S_effort_to_IM
    fin_exam_grade = og.fin_exam_grade

    # Final period value function (EV_{T}) and continuation value
    VF_prime_T = og.VT

    # Constants
    LB1 = og.LB1
    LB01 = og.LB01
    LB0 = og.LB0
    UBNEL1 = og.UBNEL1
    UBNEL2 = og.UBNEL2
    init1 = og.init1
    init2 = og.init2


    @njit
    def eval_obj(S_vec,m,mh,beta,VF_prime_ind,es,t):
        
        """
        Evaluate Q function, that is, the objective function for each t 
        inside braces of the Bellman operator (Equation (15) in paper)

        Parameters
        ----------
        S_vec : 4-D array
            Study hours vector; indices:
            - S_vec[0]: SAQ
            - S_vec[1]: e-book
            - S_vec[2]: mcq
            - S_vec[3]: sim

        m : float64
            Exam knowledge at start of t, not yet depreciated from last period.

        mh : float64
            Coursework grade at start of t.

        beta : float64
            Hyperbolic discount rate at t.

        VF_prime_ind : 2D array
            Continuation function for t+1, conditioned on t shock.

        es : float64
            Productivity shock value. This is set to 
            a constant index since this version of model has no 
            productivity shocks. 

        t : int
            Week.

        Returns
        -------
        util : float64
            Q function value or continuation value at start of period utility,
            sum of within period utility  and continuation value 
            at the of period. 

        Notes
        -----

        """

        # If study hours exceed total hours, return penalty 
        # ensures study hopurs are bounded 
        S_total = np.sum(S_vec)  

        if S_total >= H or S_vec[0]<0 or S_vec[1]<0 or S_vec[2]<0 or S_vec[3]<0:
            return np.inf
        else:
            # Calculate total knowlege produced and CW marks produced
            IM, IMh, S_mcq_hat,S_hap_hat,S_eb_hat,S_saq_hat\
                 = S_effort_to_IM(S_vec[0], S_vec[1],\
                    S_vec[2],S_vec[3], m,mh,es,t)

            # Next period exam knowledge and CW grade
            m_prime = min((1-d)*m + IM,M[-1])
            mh_prime = min(100,mh + IMh)

            # Intra-period utility from leisure + CW grade 
            # Assume students get utility form coursework if new CW 
            # grades are able to be accrued 
            IMh_util = 0 
            if mh< 100:
                IMh_util = IMh

            period_utl = u_l(S_total + kappa_1*S_vec[0]\
                                     + kappa_2*S_vec[1]\
                                     + kappa_3*S_vec[2]\
                                     + kappa_4*S_vec[3],\
                                       mh_prime)
            
            # Evaluate continuation value 
            points = np.array([m_prime,mh_prime])
            v_prime_val = eval_linear(MM,VF_prime_ind,points)

            return period_utl + beta*delta*v_prime_val

    @njit
    def do_bell(t,VF_prime):

        """ Evaluates the Bellman Equation across all state 
        values

        Equation (15) in the paper.  

        Parameter
        ---------
        t: int
            Week (python indexed)
        VF_prime: 4D array
            Value function interpolant for next period.

        Returns
        -------
        V_pol: 4D array
            Value function interpolant for current period 
        S_pol: 5D array
            Study policies for each study bucket for the week. 

        Notes
        -----
        """

        # Generate empty value and policies
        VF_new = np.empty((len(beta_hat)*len(es_hat), len(M), len(Mh)))
        S_pol = np.empty((4,len(beta_hat)*len(es_hat), len(M), len(Mh)))

        # Loop through all time t state points and 
        # evaluate Value (W(t)) function
        for i in range(len(EBA_P2)):
            # Get the beta and es values for i
            beta = beta_hat[Q_shocks_ind[i,0]]
            es = es_hat[Q_shocks_ind[i,1]]
            for j in range(len(M)):
                for k in range(len(Mh)):
                    m = M[j]
                    mh = Mh[k]

                    # Get the continuation value array for this exog. state
                    VF_prime_ind = VF_prime[i]
                    initial = np.array([UBNEL1,UBNEL1,UBNEL1,UBNEL1])
                    bounds = np.array([[0, 100], [0, 100],[0, 100], [0, 100]])
                    sols = nelder_mead(eval_obj,\
                                         initial,\
                                         bounds = bounds,\
                                         args = (m, mh,beta, VF_prime_ind,es,t))

                    # Solve for the RHS of "W"alue 
                    # Ordering in study policy vector same as study hours 
                    # inputs in S_effort_to_IM and eval_obj docstring  
                    S_pol[0,i,j,k] = sols.x[0]
                    S_pol[1,i,j,k] = sols.x[1]
                    S_pol[2,i,j,k] = sols.x[2]
                    S_pol[3,i,j,k] = sols.x[3]

                    # Calculate total knowlege produced 
                    # and new CW grade for optimal study vector 
                    IM, IMh, S_mcq_hat,S_hap_hat,S_eb_hat,S_saq_hat\
                            = S_effort_to_IM(S_pol[0,i,j,k], S_pol[1,i,j,k],\
                                 S_pol[2,i,j,k],S_pol[3,i,j,k], m,mh,es, t)

                    # In week 2, when SAQ hours are low
                    # refine policy function with a lower starting
                    # value in the optimizer 
                    if t == 1:
                        bounds = np.array([[0, init1], [0, UBNEL2],[0, UBNEL2], [0, UBNEL2]])
                        initial = np.array([LB0,S_pol[1,i,j,k],S_pol[2,i,j,k],S_pol[3,i,j,k]])
                        sols_1 = nelder_mead(eval_obj,\
                                             initial,\
                                             bounds = bounds,\
                                             args = (m, mh,beta, VF_prime_ind,es,t))
                        S_pol[0,i,j,k] = sols_1.x[0]

                    m_prime = (1-d)*m + IM
                    mh_prime = mh + IMh
                    points_star = np.array([m_prime, mh_prime])
                    VF_new[i,j,k] = sols.fun \
                                        + beta*(1-delta)\
                                        *eval_linear(MM,VF_prime_ind,points_star)

        return VF_new,S_pol

    @njit
    def cond_VF(VF_new):
        """ Condition the t+1 continuation vaue on 
         time t shock information

        Parameters
        ----------
        VF_new: 4D array
            Value function interpolant conditioned on next
            period shocks
        
        VF_prime: 4D array
            Expected value of VF conditioned on current period shocks 


         Notes
         -----
         """

        # rows of EBA_P2 correspond to time t all exogenous state index
        # cols of EBA_P2 correspond to transition to t+1 exogenous state index
        # numpy dot sum product over last axis of matrix_A 
        # (t+1 continuation value unconditioned)
        # see numpy dot docs
        VF_prime = np.empty((np.shape(VF_new)))
        for i in range(len(M)):
            for j in range(len(Mh)):
                for b in range(len(EBA_P2)):
                    VF_prime[b,i,j] = np.dot(EBA_P2[b], VF_new[:, i,j])

        return VF_prime

    @njit
    def run_TS_i(s,i, S_pol_all):
        """Generate a time series for a single agent.

        Parameters
        ----------
        s : int
            Sample index.
        i : int
            Individual index.
        S_poll_all : float64
            Policy functions.

        Returns
        -------
        TS_all : 2D array
            Time series of variables by week.

        Todo
        ----
        - Re-check the timing and accumulation paths to ensure conformity with data.
        - Explicitly derive the eb_hat s_hat term from the calibration.

        Notes
        -----
        - Function checked on 17/06/2023 by AS.
        """

        #   The indices for time in the moments and this function and 
        #   time-series will be as follows:

        #   TS_all[0]: Week minus 1 is a dummy week to
        #               generate auto-corrs easily using the np.cov function
        #   TS_all[1]: Teaching week 1
        #   TS_all[t]: Teaching week t
        #   ----
        #   TS_all[10]: Teaching week 10
        #   TS_all[11]: Study week
        #   TS_all[12]: Final exam week

        # Generate empty grid (recall we appended in one extra week before week 0)
        TS_all = np.zeros((T+1,36))
        TS_all[0, 1] = 1

        # draw initial values of the shocks
        beta_ind = np.arange(len(beta_hat))\
                [np.searchsorted(np.cumsum(beta_stat), U[s,i,0,0])]
        
        es_ind = np.arange(len(es_stat))\
                [np.searchsorted(np.cumsum(es_stat), U[s,i,0,1])]

        beta = beta_hat[beta_ind]
        es = es_hat[es_ind]

        # draw exam perceived and real ability 
        zetah_ind = np.arange(len(zeta_hhat))\
                [np.searchsorted(np.cumsum(P_zetah[0]), U_z[s,i,0])]

        zeta_ind = np.arange(len(zeta_hat))\
                [np.searchsorted(np.cumsum(P_zeta[0]), U_z[s,i,1])]

        # Re-shape the policies so they are indexed by the shocks
        # on thier own axis; S_pol_all[t, j, k,l, m,n] is
        # Study policy of at week t, beta shock j, es shock j, 
        # exam knowledge m and CW stock n. 

        S_pol_all = S_pol_all.reshape((T-1, 4,len(beta_hat),len(es_hat), len(M), len(Mh)))
        TS_all[0, 1] = 1
        TS_all[0, 3] = 1
        
        for t in range(1,T):
            # Note here t will loop to 1,....,T-1
            # where T-1 = 11 and TS_all[11] is the study week
            # Thus the knowledge capital the cumulative study inputs
            # at t = 11 will be knowledge taken into t = 12, the exam week

            # Capital and CW grade at time t
            m = TS_all[t-1, 1]*(1-d)
            mh = TS_all[t-1, 3]
            points  = np.array([m,mh])

            # Evaluate study policues 
            S_saq = eval_linear(MM, S_pol_all[t-1, 0,beta_ind,es_ind,:],points)
            S_eb = eval_linear(MM, S_pol_all[t-1, 1,beta_ind,es_ind,:],points)
            S_mcq = eval_linear(MM, S_pol_all[t-1, 2,beta_ind,es_ind,:],points)
            S_hap = eval_linear(MM, S_pol_all[t-1, 3,beta_ind,es_ind,:],points)

            # Set sim to close to zero but positive when activity was not availible.
            if t < 5:
                S_hap = LB1
            if t < 2:
                S_saq = LB1

            # Force week 2 SAQs to hit the lower bound if they are too 
            # close to zero and use refined value
            if t == 2:
                S_saq = max(LB1,S_saq)

            # Note week 1 MCQs will be zero due to small theta value for MCQ
            # checked and verified in simulated output data. 

            # Calculate total study ours and examy
            S_total = S_saq + S_eb + S_mcq + S_hap

            IM, IMh, S_mcq_hat,S_hap_hat,S_eb_hat,S_saq_hat\
                = S_effort_to_IM(S_saq, S_eb, S_mcq,S_hap, m,mh,es,t-1)

            # Update with vals for time t
            TS_all[t,0] = TS_all[t-1, 1] # t Knowledge capital 
            TS_all[t,1] = min(M[-1], m + IM) # t+1 Knowledge capital
            TS_all[t,2] = mh # t-1 Coursework grade (CW grade at the beggining of t!)
            TS_all[t,3] = min(mh + IMh,100) # t+1 Coursework grade (coursework grade at the end of t)
            TS_all[t,4] = TS_all[t-1,5]  # t-1  S_saq_hours 
            TS_all[t,5] = TS_all[t-1,5] + S_saq # t  S_saq_hours cum
            TS_all[t,6] = TS_all[t-1,7]  # t-1  S_eb 
            TS_all[t,7] = S_eb + TS_all[t-1,7]   # t   S_eb_bours cum
            TS_all[t,8] = TS_all[t-1,9]  # t-1  S_mcq_bours  cum
            TS_all[t,9] = TS_all[t-1,9] + S_mcq # t  S_mcq_bours  cum
            TS_all[t,10] = TS_all[t-1,11] # t-1  S_sim)hours
            TS_all[t,11] = max(LB1, S_hap) + TS_all[t-1,11] # t  S_sim_hours cum
            TS_all[t,12] = TS_all[t-1,13] # t-1  total study
            TS_all[t,13] = S_total # t total study (not cum)
            TS_all[t,14] = TS_all[t-1,15]  # t-1  S_saq_hat 
            TS_all[t,15] = S_saq_hat + TS_all[t-1,15] # t  S_saq_hat cum
            TS_all[t,16] = TS_all[t-1,17]  # t-1  S_eb_hat
            TS_all[t,17] = S_eb_hat + TS_all[t-1,17]  # t S_eb_hat cum 
            TS_all[t,18] = TS_all[t-1, 19]  # t-1  S_mcq 
            TS_all[t,19] = TS_all[t-1, 19] + S_mcq_hat # t  S_mcq_hat cum
            TS_all[t,20] = TS_all[t-1,21] # t-1  S_hap_hat
            TS_all[t,21] = TS_all[t-1,21] + S_hap_hat # t  S_hap_hat cum

            TS_all[t,22] = TS_all[t-1, 23]
            TS_all[t,23] = a*(theta[t-1]*iota_c + (1-theta[t-1])*iota_i)*TS_all[t,19] 
                                                                                                
            TS_all[t,28] = TS_all[t-1, 29]  # t-1 mcq_Cattempt_nonrev 
            TS_all[t,29] = a*(theta[t-1]*iota_c + (1-theta[t-1])*iota_i)*TS_all[t,19] # t mcq_Cattempt_nonrev 

            # Create weekly pairwise sums
            TS_all[t,30] = S_eb + S_hap
            TS_all[t,31] = S_eb + S_mcq
            TS_all[t,32] = S_eb + S_saq
            TS_all[t,33] = S_hap + S_mcq
            TS_all[t,34] = S_hap + S_saq
            TS_all[t,35] = S_mcq + S_saq


            # Bound TS outputs to keep within range during SMM
            if t<5:
                TS_all[t,15] = min(5,TS_all[t,15])

            # update the preference shock for the next period
            beta_ind_new = np.arange(len(beta_hat))\
                [np.searchsorted(np.cumsum(P_beta[beta_ind]), U[s,i,t,0])]
            beta = beta_hat[beta_ind_new]
            beta_ind = beta_ind_new

            es_ind_new = np.arange(len(es_hat))\
            [np.searchsorted(np.cumsum(P_es[es_ind]), U[s,i,t,1])]
            es = es_hat[es_ind_new]
            es_ind = es_ind_new

        TS_all[T,24] = fin_exam_grade(zeta_hat[zeta_ind], TS_all[T-1,1]*(1-d))*rho_E # Actual exam grade
        TS_all[T,25] = fin_exam_grade(zeta_hhat[zetah_ind], TS_all[T-1,1]*(1-d))*rho_E # Randomised percieved exam grade
        TS_all[T,26] = TS_all[T,24] + (1-rho_E)*TS_all[T-1,3] # Actual final mark
        TS_all[T,27] = zeta_hat[zeta_ind] # Actual final shock 

        # Fill in final exam marks into previous periods 
        for t in range(1,T):
            TS_all[t,24] = TS_all[T,24]
            TS_all[t,25] = TS_all[T,25]
            TS_all[t,26] = TS_all[T,26]
            TS_all[t,27] = TS_all[T,27]
        
        

        return TS_all

    @njit 
    def TSALL(S_pol_all,s):
        """ Generates the s'th sample of N individual series

        Parameters
        ----------
        S_pol_all: 5D array
            Study policy functions.
        s: int
            Sample index. 

        Returns 
        -------
        TS_all: 3D array
            Time series of N individuals.


        Notes
        -----
        - First index of TS_all indexes individuals 
        - Second for t and third for variable as ordered by construction 
            of variables in function run_TS_i
        """

         
        TS_all = np.empty((N,T+1, 36))

        # Generate s'th samples of N individual time-series 
        for i in range(N):
            TS_all[i,:,:] = run_TS_i(s,i,S_pol_all )

        return TS_all

    def edu_iterate():

        """ Solves for policy functions via backward induction

        Parameters
        ---------- 

        Returns
        -------
        S_pol_all: 5D array 
            Policy functions for all study buckets across all weeks
        
        VF_prime: 4D array 
            Value function for first period 
    

        Notes
        -----
        """

        # Start with the final period continuation value given 
        # by utility from total course grade
        # this is correct, repeats along first dimension by the times of 
        # the number of shocks
        VF_prime = np.repeat(og.VT[np.newaxis, :,:], \
                            repeats = len(EBA_P2), axis = 0)  

        # Generate empty policy functions for lentgh T-1
        S_pol_all = np.empty((T-1, 4, len(EBA_P2), len(M), len(Mh)))

        for t in np.arange(0,int(T-1))[::-1]:
            #print("solving week {}".format(t))
            start = time.time()
            VF_UC,S_pol = do_bell(t,VF_prime)
            #print("do bell in {}".format((time.time()-start)/60))
            S_pol_all[t,:] = S_pol

            start = time.time()
            VF_cond = cond_VF(VF_UC)
            #print("cond in {}".format((time.time()-start)/60))
            VF_prime = VF_cond

        return S_pol_all, VF_prime

    def gen_moments(TSALL):

        """ Generates moments for a sampple time-series of agents 

        Parameters
        ----------
        TSALL: NxTxK array
                Time series for N agents, T+1 weeks and K variables

        Returns
        -------
        moments_out: TxM array
                        T weeks and M moments

        Todo
        ----

        Confirm with Isabella the timing of the ACs
        edit  moments_out[:,15] doc
        check corrs matricies again 

        Notes
        -----
        - Order moments in the same order as the list of moments .yml file

        """

        # Empty arrays to fill in with means and and cov
        # from the data for each t
        means = np.empty((T+1,36))
        cov = np.empty((T+1,36,36))
        #median = np.empty((T+1,30))

        for t in range(1,T+1):
            means[t] = np.mean(TSALL[:,t,:], axis = 0)
            cov[t,:] = np.cov(TSALL[:,t,:], rowvar=False)

        # Create moments in the same order as data moments
        # We remove the moments for the first week (weel minus 1) in the 
        # means table since that is a dummy week
        #
        # We first populate moments out with the full T = 12 weeks
        # (including the exam week)
        # then remove the exam week so we have 11 teaching/study weeks

        moments_out = np.zeros((T, 55))

        # Final grades
        moments_out[:,0] = means[1:T+1,24] # av_final
        moments_out[:,1] = means[1:T+1,26] # av_mark
        moments_out[:,2] = means[1:T+1,25] # av_markw13_exp1
        moments_out[:,3] = means[1:T+1,1]   #av_knowledge_cumul

        # Study hours inputs
        moments_out[:,4] = means[1:T+1,11]# av_sim_session_hours_cumul
        moments_out[:,5] = means[1:T+1,7] # av_ebook_session_hours_cumul
        moments_out[:,6] = means[1:T+1,9] # avg mcq session hours cum
        moments_out[:,7] = means[1:T+1,5] # avg saq session hours cum

        # Observable study outputs 
        moments_out[:,8] = means[1:T+1,21] # av_happy_deploym_cumul.
        moments_out[:,9] = means[1:T+1,19] # av_mcq_attempt_nonrev_cumul'
        moments_out[:,10] = means[1:T+1,15] # av_saq_attempt_cumul
        moments_out[:,11] = means[1:T+1,17] # av_totebook_pageviews_cumul

        # SDs 
        moments_out[:,12] = np.std(TSALL[:,1: T+1,24], axis = 0)    # sd_final
        moments_out[:,13] = np.std(TSALL[:,1: T+1,26], axis = 0)    # sd_mark
        moments_out[:,14] = np.std(TSALL[:,1: T+1,25], axis = 0)    # sd_markw13_exp1
        moments_out[:,15] = np.std(TSALL[:,1: T+1,1], axis = 0)     # sd_markw13_exp2 #becomes mean CW grade since moment not used in SMM 

        moments_out[:,16] = np.std(TSALL[:,1: T+1,11], axis = 0)    # sd_sim_session_hours_cumul
        moments_out[:,17] = np.std(TSALL[:,1: T+1,7], axis = 0)     # sd_ebook_session_hours_cumul
        moments_out[:,18] = np.std(TSALL[:,1: T+1,9], axis = 0)     # sd_mcq_session_hours_cumul
        moments_out[:,19] = np.std(TSALL[:,1: T+1,5], axis = 0)     # sd_saq_session_hours_cumul

        moments_out[:,20] = np.std(TSALL[:,1: T+1,21], axis = 0)    # sd_happy_deploym_cumul
        moments_out[:,21] = np.std(TSALL[:,1: T+1,19], axis = 0)    # sd_mcq_attempt_nonrev_cumul
        moments_out[:,22] = np.std(TSALL[:,1: T+1,15], axis = 0)    # sd_sa_attempt_cumul
        moments_out[:,23] = np.std(TSALL[:,1: T+1,17], axis = 0)    # sd_totebook_pageviews_cumul

        # Autocorrelations
        moments_out[:,24] = cov[1:T+1,11,10]/(np.std(TSALL[:,1: T+1,11], axis = 0)\
                                                *np.std(TSALL[:,1: T+1,10], axis = 0) ) # acsim_session_hours
        moments_out[:,25] = cov[1:T+1,7,6]/(np.std(TSALL[:,1: T+1,6], axis = 0)\
                                                *np.std(TSALL[:,1: T+1,7], axis = 0) )  # acebook_session_hours
        moments_out[:,26] = cov[1:T+1,8,9]/(np.std(TSALL[:,1: T+1,8], axis = 0)\
                                                *np.std(TSALL[:,1: T+1,9], axis = 0) )  # acmcq_session_hours
        moments_out[:,27] = cov[1:T+1,5,4]/(np.std(TSALL[:,1: T+1,5], axis = 0)\
                                                *np.std(TSALL[:,1: T+1,4], axis = 0) )  # acsaq_session_hours
        moments_out[:,28] = cov[1:T+1,17,16]/(np.std(TSALL[:,1: T+1,16], axis = 0)\
                                                *np.std(TSALL[:,1: T+1,17], axis = 0) )  # actotebook_pageviews
        moments_out[:,29] = cov[1:T+1,29,28]/(np.std(TSALL[:,1: T+1,29], axis = 0)\
                                                *np.std(TSALL[:,1: T+1,28], axis = 0) )  # acmcq_Cattempt_nonrev 

        # Correlations (week by week hours between types of study)
        moments_out[:,30] = cov[1:T+1,9,5]/(np.std(TSALL[:,1: T+1,9], axis = 0)\
                                *np.std(TSALL[:,1: T+1,5], axis = 0) ) # co_mcsaq_session_hours
        moments_out[:,31] = cov[1:T+1,5,11]/(np.std(TSALL[:,1: T+1,5], axis = 0)\
                                *np.std(TSALL[:,1: T+1,11], axis = 0) ) # co_simsaq_session_hours
        moments_out[:,32] = cov[1:T+1,9,11]/(np.std(TSALL[:,1: T+1,9], axis = 0)\
                                *np.std(TSALL[:,1: T+1,11], axis = 0) ) # co_simmcq_session_hours
        moments_out[:,33] = cov[1:T+1,7,5]/(np.std(TSALL[:,1: T+1,7], axis = 0)\
                                *np.std(TSALL[:,1: T+1,5], axis = 0) ) # co_esaq_session_hours
        moments_out[:,34] = cov[1:T+1,7,9]/(np.std(TSALL[:,1: T+1,7], axis = 0)\
                                *np.std(TSALL[:,1: T+1,9], axis = 0) )# co_emcq_session_hours
        moments_out[:,35] = cov[1:T+1,7,11]/(np.std(TSALL[:,1: T+1,7], axis = 0)\
                                *np.std(TSALL[:,1: T+1,11], axis = 0)) # co_esim_session_hours

        # Correlations (pairwise with week by week cumul output and final exam)
        moments_out[:,36] = cov[1:T+1,11,24]/(np.std(TSALL[:,1: T+1,24], axis = 0)\
                                *np.std(TSALL[:,1: T+1,11], axis = 0) ) # co_fsim_session_hours_cumul
        moments_out[:,37] = cov[1:T+1,7,24]/(np.std(TSALL[:,1: T+1,7], axis = 0)\
                                *np.std(TSALL[:,1: T+1,24], axis = 0) ) # co_febook_session_hours_cumul
        moments_out[:,38] = cov[1:T+1,9,24]/(np.std(TSALL[:,1: T+1,9], axis = 0)\
                                *np.std(TSALL[:,1: T+1,24], axis = 0) ) # co_fmcq_session_hours_cumul
        moments_out[:,39] = cov[1:T+1,5,24]/(np.std(TSALL[:,1: T+1,5], axis = 0)\
                                *np.std(TSALL[:,1: T+1,24], axis = 0) ) # co_fsaq_session_hours_cumul
        moments_out[:,40] = cov[1:T+1,21,24]/(np.std(TSALL[:,1: T+1,24], axis = 0)\
                                *np.std(TSALL[:,1: T+1,21], axis = 0) ) # co_fhappy_deploym_cumul
        moments_out[:,41] = cov[1:T+1,17,24]/(np.std(TSALL[:,1: T+1,17], axis = 0)\
                                *np.std(TSALL[:,1: T+1,24], axis = 0) ) # co_ftotebook_pageviews_cumul
        moments_out[:,42] = cov[1:T+1,19,24]/(np.std(TSALL[:,1: T+1,19], axis = 0)\
                                *np.std(TSALL[:,1: T+1,24], axis = 0) ) # co_fmcq_attempt_nonrev_cumul
        moments_out[:,43] = cov[1:T+1,15,24]/(np.std(TSALL[:,1: T+1,15], axis = 0)\
                                *np.std(TSALL[:,1: T+1,24], axis = 0) ) # co_fsa_attempt_cumul

        # Correlations (week by week hours between study hours and output for each type)
        moments_out[:,44] = cov[1:T+1,11,21]/(np.std(TSALL[:,1: T+1,11], axis = 0)\
                                *np.std(TSALL[:,1: T+1,21], axis = 0) ) # co_sim
        moments_out[:,45] = cov[1:T+1,7,17]/(np.std(TSALL[:,1: T+1,7], axis = 0)\
                                *np.std(TSALL[:,1: T+1,17], axis = 0) ) # co_ebook
        moments_out[:,46] = cov[1:T+1,9,19]/(np.std(TSALL[:,1: T+1,9], axis = 0)\
                                *np.std(TSALL[:,1: T+1,19], axis = 0) ) # co_mcq
        moments_out[:,47] = cov[1:T+1,15,5]/(np.std(TSALL[:,1: T+1,5], axis = 0)\
                                *np.std(TSALL[:,1: T+1,15], axis = 0) ) # co_saq
        
        # Correlations (weekly hours sum two types of study pairs, final exam)
        moments_out[:,48] = cov[1:T+1,30,24]/(np.std(TSALL[:,1: T+1,30], axis = 0)\
                                        *np.std(TSALL[:,1: T+1,24], axis = 0) ) # cesim_session_hours
        moments_out[:,49] = cov[1:T+1,31,24]/(np.std(TSALL[:,1: T+1,31], axis = 0)\
                                        *np.std(TSALL[:,1: T+1,24], axis = 0) ) # emcq_session_hours
        moments_out[:,50] = cov[1:T+1,32,24]/(np.std(TSALL[:,1: T+1,32], axis = 0)\
                                        *np.std(TSALL[:,1: T+1,24], axis = 0) ) # cesaq_session_hours
        moments_out[:,51] = cov[1:T+1,33,24]/(np.std(TSALL[:,1: T+1,33], axis = 0)\
                                        *np.std(TSALL[:,1: T+1,24], axis = 0) ) # csimmcq_session_hours
        moments_out[:,52] = cov[1:T+1,34,24]/(np.std(TSALL[:,1: T+1,34], axis = 0)\
                                        *np.std(TSALL[:,1: T+1,24], axis = 0) ) # csimsaq_session_hours
        moments_out[:,53] = cov[1:T+1,35,24]/(np.std(TSALL[:,1: T+1,35], axis = 0)\
                                        *np.std(TSALL[:,1: T+1,24], axis = 0) ) # cmcsaq_session_hours

        # Atar corr with final exam 
        if np.sum(np.std(TSALL[:,1: T+1,24], axis = 0))>0:
            moments_out[:,54] = cov[1:T+1,27,24]/(np.std(TSALL[:,1: T+1,27], axis = 0)\
                                                    *np.std(TSALL[:,1: T+1,24], axis = 0))  # c_atar_ii 

        # Return moments for 11 weeks including study week 
        return moments_out[0:T-1,:]
   
    return edu_iterate, TSALL,gen_moments

def generate_study_pols(og):

    """ Generates average moments for an instance of 
            EduModel Class

    Parameters
    ----------
    og : EduModel object

    Returns
    -------
    moments_out : TxM array

    
    Notes
    -----
    - The function generates S samples of N individuals and takes the
        averge moments across the samples. 

    """
    edu_iterate,TSALL,gen_moments = edu_solver_factory(og)
    S_pol_all, VF_prime = edu_iterate()
    time_str = time.time()
    
    # create S samples of N individual time-series 

    start =  time.time()

    moments_all = np.empty((og.S,og.T-1, 55))
    
    for s in range(og.S):
        TS_all = TSALL(S_pol_all,s)
        moments_all[s] = gen_moments(TS_all)
        del TS_all
        gc.collect()

    # Generate the mean of the moments across the S samples
    moments_out = np.mean(moments_all, axis = 0)
    del S_pol_all
    del VF_prime
    del moments_all
    gc.collect()

    return moments_out
