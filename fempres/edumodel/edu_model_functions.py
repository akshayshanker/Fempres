"""
Module generates oeprator, edumodel_function_factor, that
returns all EduModel functions given a 
set of parameters that are read in 
"""

# Import packages
import numpy as np
from numba import njit, prange, jit, vectorize

def edumodel_function_factory(params, constants, theta,share_saq,\
                                share_eb,share_mcq,share_hap):  

    """ Operator to create EduModel functions

    Parameters
    ----------
    params: dict
        dictionary of parameter names and values
    theta: list
        list of mcq correct rate values
    share_saq: list
        list of study hours to output share for SAQ
    share_eb: list
        list of study hours to output share for e-book
    share_mcq: list
        list of study hours to output share for MCQ
    share_hap: list
        list of study hours to output share for sim

    Returns
    -------
    u_grade: function 
        utility from final exam grade 
    fin_exam_grade: function 
        final exam grade as a function of knowledge and ability
    S_effort_to_IM: function 
        new knowledge as a function of effort 
            across study buckets
    u_l: function 
        per-period utility during semeseter 
            
    """

    # Read in parameters from parameter dictionary 
    # See settings/settings.yml for variable definitions
    units_psi = params['units_psi']
    units_a = params['units_a']
    units_phi = params['units_phi']
    UB_CW  = params['UB_CW']
    B1  = params['B1']
    B2  = params['B2']
    B3  = params['B3']
    B4  = params['B4']
    
    rho_E = params['rho_E']
    psi_E = params['psi_E']*units_psi
    alpha = params['alpha']*units_a
    H  = params['H']
    phi = params['phi']*units_phi
    phi_star = params['phi_star']*units_phi
    a = params['a']
    b = params['b']
    M_max = params['M_max']
    varphi = params['varphi']
    varphi_sim = params['varphi_sim']
    varphi_sim_star = varphi_sim

    # CES
    gamma_1 = params['gamma_1']
    gamma_2 = params['gamma_2']
    gamma_3 = params['gamma_3']
    gamma_4 = 1 -gamma_1 -gamma_2 - gamma_3
    sigma_M = params['sigma_M']
    T = params['T']
    A = 1

    iota_c = params['iota_c']
    iota_i = params['iota_i']

    # Convert MCQ correct rate and study share to np vectors 
    s_share_saq = np.array(share_saq)
    s_share_eb = np.array(share_eb)
    s_share_mcq = np.array(share_mcq)
    s_share_hap = np.array(share_hap)
    theta = np.array(theta)

    LB1 = constants['LB1']
    LB01 = constants['LB01']
    LB0 = constants['LB0']
    UBNEL1 = constants['UBNEL1']
    UBNEL2 = constants['UBNEL2']
    init1 = constants['init1']
    init2 = constants['init2']



    @njit
    def u_grade(FG, Mh_T):
        """ Utility for agent from final course grade
            
        Parameters
        ----------
        FG: float64
            Exam grade
        Mh_T: float64
            Coursework grade

        Notes
        -----
        - Note the final course grade is out of 100
        - Recall log(rho_E*FG) = log(rho_E) + log(FG)


        """     

        # If pass, then receives utility 
        if FG> 50:
            return alpha*np.log(FG) 
        else:
            return - np.inf

    @njit 
    def fin_exam_grade(zeta_T, M_T):
        """ Final exam grade of student, out of 100

        Parameters
        ----------
        zeta_T: float64
            ability shock realisation 
        M_T: float64
            exam knowledge capital 
        Returns
        -------
        grade: float64
            final exam grade 

        Notes
        -----

        """
        return ((1-np.exp(-psi_E*(zeta_T+M_T))))*100 


    @njit 
    def study_output_factor(m, book = False):
        """"
        Function that produces the scaling factor 
        that converts study hours to study output. 

        The function is the function g in Equation 
        (8) of paper. 

        Parameters
        ----------
        m: float
            knowledge stock
        book: Bool
            Indicator for whether study book is e-book 

        Notes
        -----


        """
        if book == False: 
            return max(B1,min(B2,\
                        (1-np.exp(-varphi_sim*(m/phi)))/varphi_sim))
        
        if book == True: 
            return max(B3,min(B4,\
                        (1-np.exp(-varphi_sim*(m/phi)))/varphi_sim))

    @njit
    def CES(S_saq, S_eb, S_mcq,S_hap, sigma):
        """ Top level CES study function 
            Equation (7) in paper.

        Paramters
        ---------
        S_saq: float64
            SAQ study hours 
        S_eb: float64
            E-book study hours 
        S_mcq: float64
            MCQ study hours 
        S_hap: float64
            Simulation study hours
        sigma: float64
            CES parameters

        Returns
        -------
        m: float64
            non-normalised new knowledge 

        Notes
        -----
        - Note we have used adjustable weights specification of CES

        """
        inside_sum = (gamma_1**(1/sigma))*S_saq**((sigma - 1)/sigma)\
                        + (gamma_2**(1/sigma))*S_eb**((sigma - 1)/sigma)\
                        + (gamma_3**(1/sigma))*S_mcq**((sigma - 1)/sigma)\
                        + (gamma_4**(1/sigma))*S_hap**((sigma - 1)/sigma)

        return (inside_sum**(sigma/(sigma - 1)))

    @njit 
    def S_effort_to_IM(S_saq, S_eb, S_mcq,S_hap, m,mh,es, t):
        """ Converts study effort to exam knowledge, study outputs 
            and CW grade accurred in the week. 

        Parameters
        ----------
        S_saq: float64
            hours of SAQ
        S_eb: float64
            hours of e-books
        S_mcq: float64
            hours of MCQ points
        S_hap: float64
            hours in the sim 
        m: float64
            knowledge capital at *start* of time period t 
                (after it has depreciated)
        mh: float64
            coursework grade at *start* time t
        t: int
            week with python indexing

        Returns
        -------
        IM: float64
            new knowledge created 
        IMh: float64
            new coursework grades
        S_mcq_hat: float64
            MCQ outputs 
        S_hap_hat: float64
            Simulation output
        S_eb_hat: float64
            E-book output
        S_saq_hat: float64
            SAQ output 

        Notes
        -----
        - Coursework grades throughout the semester
            capped at 100

        """
        
        # Make sure study hours are non-negative 
        S_saq = max(1e-10,S_saq)
        S_eb = max(1e-10,S_eb)
        S_mcq = max(1e-10,S_mcq)
        S_hap = max(1e-10,S_hap)

        # Study hours in sim before week 5 unproductive as
        # sim not available in week 5
        # (equivalent to high cost)
        # note MCQ theta is set to 0 hence will give no output, 
        # forcing zero study on MCQ in week 1
        if t < 4:
            S_hap = .01
        if t == 0:
            S_saq = .01

        IM  = es*phi*CES(S_saq,S_eb,S_mcq,S_hap, sigma_M)

        if s_share_mcq[t] > 0:
            S_mcq_hat = (S_mcq/s_share_mcq[t])*study_output_factor(m)
        else:
            S_mcq_hat  = S_mcq*.01
        if s_share_hap[t] >0:
            S_hap_hat =  (S_hap/s_share_hap[t])*study_output_factor(m)
        else:
            S_hap_hat = S_hap*.01
        if s_share_eb[t]> 0:
            S_eb_hat =  (S_eb/s_share_eb[t])*study_output_factor(m,book= True)
        else: 
            S_eb_hat = S_eb*.01
        if s_share_saq[t]> 0:
            S_saq_hat = (S_saq/s_share_saq[t])*study_output_factor(m)
        else:
            S_saq_hat = S_saq*.01

        # Rate of current MCQ answers 
        rate_of_correct = theta[t]
        b_actual = b
        IMh = min(max(0, a*(rate_of_correct*iota_c \
                            + (1-rate_of_correct)*iota_i)*S_mcq_hat\
                            + b_actual*S_hap_hat), UB_CW)

        return IM, IMh, S_mcq_hat,S_hap_hat,S_eb_hat,S_saq_hat

    @njit
    def u_l(S,IMh):
        """ Per-period utility leisure for study and CW grade improvement
                Equation (12) in paper (without final exam utility)

        Parameters
        ----------
        S: float64
            total study hours
        IMh: float64
            coursework grade total at end of week
        
        Returns
        -------
        u_week: float64
            utility within period 
        """
        l = H - S
        # Ensure non-study hours do not exceed 168
        l = max(l, 1e-200)

        return np.log(l)  + alpha*np.log(IMh*(1-rho_E))/T  

    return u_grade, fin_exam_grade, S_effort_to_IM, u_l