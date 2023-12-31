"""
Script plots results for using baseline data and 
counterfactual data generated by gen_baseline.py
and gen_counterfactuals.py. 

Plotting functions here generate plots used in paper 

Functions:
---------
plot_results_paper2
    Generates baseline profile plots for paper
plot_cfs_lvl
    Generates counterfactual plots in paper 

Main block of script can be run using Ipython shell. 

"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import time


def plot_results_paper2(moments_sim_all,
                        moments_data_all,
                        variable_list,
                        variable_list_names,
                        group_list_names, 
                        group_list_diffs, 
                        results_path, list_moments_plot,ID):


    """ Function generates baseline profile plots

    Parameters
    ----------
    moments_sim_all: numpy array
        Simulated baseline profiles
    moments_data_all: numpy array
        Data profiles
    variable_list: list
        list of variables to plot
    variable_list_names: list
        names of each variable used in plot
    group_list_diffs: list 
        names of groups plotted in each sub-plot
    group_list_diffs: list
        tau index of each group  plotted eacub 
                        sub-plot
    results_path: string
        directory to save results
    ID: Unique result ID 
    

    Returns
    -------

    Notes
    -----
     - The first index in the moments_sim_all is the tau group index
     - The variables and moments in the array moments_sim_all and 
        moments_data_all are it the same order as the variable list.

    """

    # Line names and formatting used in legend 
    line_names = [['Data: WP-DG', 'Sim:WP-DG','Data: WP-UG', 'Sim: WP-UG'],\
                    ['Data: WP-DG', 'Sim:WP-DG','Data: WP-UG', 'Sim: WP-UG'],\
                    ['Data: WP-DG', 'Sim:WP-DG','Data: MP-DG', 'Sim: MP-DG'],\
                    ['Data: WP-DG', 'Sim:WP-DG','Data: MP-DG', 'Sim: MP-DG']]

    linestyles=["-","-"]
    col_dict = {'data': 'black', 'sim':'gray'}
    markers=['x', 'o']
    
    # Make sure plot path exists 
    plot_path = results_path + "/plots/baseline/plot_paper2_{}".format(ID)
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    # Loop through each variable and plot with custom format 
    # for each variable 

    for i, name in zip(np.arange(len(variable_list)),variable_list):

        fig, ax1 = plt.subplots(5,2,figsize=(8, 6))
        ax = ax1.flatten()[list([2,3,6,7])]

        if variable_list[i] in list_moments_plot:
            
            for k in np.arange(len(group_list_names)):

                xs = np.arange(1,12)
                ys = moments_data_all[int(group_list_diffs[k][0]),:,i]

                if variable_list_names[i]  == 'Time solving MCQs (hours)':
                    ys[0] = np.nan
                if variable_list_names[i] == 'Number of happines units':
                    ys = ys/1000

                
                p, = ax[k].plot(xs, ys, marker = markers[0], color = col_dict['data'], linestyle = linestyles[0],
                                label = line_names[k][0], linewidth = 2, markersize = 8)
                ys = moments_data_all[int(group_list_diffs[k][1]),:,i]
                if variable_list_names[i] == 'Time solving MCQs (hours)':
                    ys[0] = np.nan
                if variable_list_names[i] == 'Number of happines units':
                    ys = ys/1000
                p, = ax[k].plot(xs, ys, marker = markers[1], color = col_dict['data'], linestyle = linestyles[0],
                    label = line_names[k][2], linewidth = 2)

                moments_sim_all[int(group_list_diffs[k][0]),:,i]\
                        [np.isnan(moments_data_all[int(group_list_diffs[k][0]),:,i] )] = np.nan
                
                ys = moments_sim_all[int(group_list_diffs[k][0]),:,i]
                if variable_list_names[i] == 'Time solving MCQs (hours)':
                    ys[0] = np.nan
                if variable_list_names[i] == 'Number of happines units':
                    ys = ys/1000

                p, = ax[k].plot(xs, ys, marker=markers[0], color=col_dict['sim'], linestyle = linestyles[1],
                                    label=line_names[k][1], linewidth=2, markersize=8)
                moments_sim_all[int(group_list_diffs[k][1]),:,i]\
                        [np.isnan(moments_data_all[int(group_list_diffs[k][1]),:,i] )] = np.nan
                ys = moments_sim_all[int(group_list_diffs[k][1]),:,i]
                if variable_list_names[i] == 'Time solving MCQs (hours)':
                    ys[0] = np.nan

                if variable_list_names[i] == 'Number of happines units':
                    ys = ys/1000

                p, = ax[k].plot(xs, ys, marker=markers[1], color=col_dict['sim'], linestyle = linestyles[1],
                                    label=line_names[k][3], linewidth=2)


                ax[k].spines['top'].set_visible(False)
                ax[k].spines['right'].set_visible(False)
                ax[k].set_xticks([1,2,3,4,5,6,7,8,9,10,11]) 
            

                if variable_list_names[i] == 'Time studying the textbook (hours)' or variable_list_names[i] == 'Time earning happines units (hours)':
                    ax[k].set_ylim(0,60)
                    ax[k].set_yticks(np.arange(0, 60+1, 20.0))
                    ax[k].axhline(y=20, color='grey', linestyle='-',linewidth=1)
                    ax[k].axhline(y=40, color='grey', linestyle='-', linewidth=1)
                    ax[k].set_xlim(1,11)

                if variable_list_names[i] == 'Time solving MCQs (hours)':
                    ax[k].set_ylim(0,20)
                    ax[k].set_yticks(np.arange(0, 20+1, 5.0))
                    ax[k].axhline(y=8, color='grey', linestyle='-',linewidth=1)
                    ax[k].axhline(y=16, color='grey', linestyle='-', linewidth=1)
                    ax[k].set_xlim(1,11)

                if  variable_list_names[i] == 'Time answering SAQs (hours)':
                    ax[k].set_ylim(0,5)
                    ax[k].set_yticks((0,1,3,5))
                    ax[k].axhline(y=2, color='grey', linestyle='-',linewidth=1)
                    ax[k].axhline(y=4, color='grey', linestyle='-', linewidth=1)
                    ax[k].set_xlim(1,11)

                if variable_list_names[i] == 'Number of MCQ attempts':
                    ax[k].set_ylim(0,900)
                    ax[k].set_yticks((0,300,600,900))
                    ax[k].axhline(y=300, color='grey', linestyle='-',linewidth=1)
                    ax[k].axhline(y=600, color='grey', linestyle='-', linewidth=1)
                    ax[k].set_xlim(1,11)


                if variable_list_names[i] == 'Number of textbook pages':
                    ax[k].set_ylim(0,900)
                    ax[k].set_yticks((0,300,600,900))
                    ax[k].axhline(y=300, color='grey', linestyle='-',linewidth=1)
                    ax[k].axhline(y=600, color='grey', linestyle='-', linewidth=1)
                    ax[k].set_xlim(1,11)

                if variable_list_names[i] == 'Number of SAQ attempts':
                    ax[k].set_ylim(0,20)
                    ax[k].set_yticks(np.arange(0, 20+1, 5.0))
                    ax[k].axhline(y=8, color='grey', linestyle='-',linewidth=1)
                    ax[k].axhline(y=16, color='grey', linestyle='-', linewidth=1)
                    ax[k].set_xlim(1,11)

                if variable_list_names[i] == 'Number of happines units':
                    ax[k].set_ylim(0,300)
                    ax[k].set_yticks(np.arange(0, 300+1, 100.0))
                    ax[k].axhline(y=100, color='grey', linestyle='-',linewidth=1)
                    ax[k].axhline(y=200, color='grey', linestyle='-', linewidth=1)
                    ax[k].set_xlim(1,11)

            ax1 = ax1.flatten()

            ax1[0].set_ylim([0.3,0.3]);ax1[0].axis("off")
            ax1[1].set_ylim([0.3,0.3]);ax1[1].axis("off")
            ax1[4].axis("off")
            ax1[5].axis("off")
            ax1[8].axis("off")
            ax1[9].axis("off")
            ax1[2].set_xlabel('Week',fontsize=11)
            ax1[3].set_xlabel('Week',fontsize=11)       
            ax1[6].set_xlabel('Week',fontsize=11)
            ax1[7].set_xlabel('Week',fontsize=11)

            ax1[6].set_title('Males'.format(i+1), fontweight='bold')
            ax1[7].set_title('Females'.format(i+1), fontweight='bold')
            ax1[2].set_title('Males'.format(i+1), fontweight='bold')
            ax1[3].set_title('Females'.format(i+1), fontweight='bold')
            fig.subplots_adjust(hspace=0.5, bottom=0.1)

            fig.suptitle(variable_list_names[i],y=0.83)
            ax1.flatten()[7].legend(loc='upper right', \
                    bbox_to_anchor=(0.9, -0.7), ncol=4,fontsize=10, frameon=False)
            ax1.flatten()[3].legend(loc='upper right', \
                    bbox_to_anchor=(0.9, -0.7), ncol=4,fontsize=10, frameon=False)
            fig.savefig(plot_path + "/{}.png".format(name + 'final'), transparent=True)

    return 

def plot_cfs_lvl(moments_sim_gr10, moments_sim_gr20,variable_list,
                        variable_list_names,results_path, ID):

    
    """ Function plots counterfactuals for gr1 and gr2 females 
            with select gr3  female parameters


    Parameters
    ----------
    moments_sim_gr10: array
        gr10 female profiles with gr3 params
    moments_sim_gr20: array
        gr10 female profiles with gr3 params
    variable_list: array
    variable_list_names: array
        names used in plots for each variable 
    results_path: str
    ID: str

    
    Returns
    -------

    """ 

    plot_path = results_path + "/plots/counterfactuals/plot_paper2_cflvl_{}/".format(ID)
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    
    # Labels for the CFs. Top left to bottom right 
    cf_name = [r'Patience ($\beta$)', r'Self-control ($\delta$)',r'Grade weight ($\alpha$)',\
                    r'Exam difficulty ($\lambda^{E}$)',  r'Real ability ($\xi$)',\
                    r' Perceived ability ($\xi^{\star}$)', r'SAQ effort cost ($e^{SAQ}$)',\
                    r'All effort cost ($e$)'
                ]
    # Markets for each CF same order as names 
    markers = ['x', 'o', "v","P","s", "D", "*", "h"]

    moments_sim_list = [moments_sim_gr10, moments_sim_gr20]
    
    # Loop through each of the moments 
    for i, name in zip(np.arange(len(variable_list)),variable_list): 
            fig, ax1 = plt.subplots(8,2,figsize=(8, 9))
            ax = ax1.flatten()[list([2,3,6,7,10,11,14,15])]
            xs = np.arange(1,12)

            # First plot the baseline 
            # gr20 (WP-UG) goes on left panels
            # gr10 (MP-DF) goes on right panels 
            for l in [0,2,4,6]:
                ys = moments_sim_gr20[0,:,i]
                p, = ax[l].plot(xs, ys, color = 'black', linestyle = "-",\
                                    marker = "+", linewidth = 1.5,\
                                    markersize=8, zorder = 8,  label = 'Baseline'
                                )
                ys = moments_sim_gr10[0,:,i]
                p, = ax[l+1].plot(xs, ys, color = 'black', linestyle = "-",\
                                    marker = "+",linewidth = 1.5,\
                                    markersize=8, zorder = 8,  label = 'Baseline'
                                )
            # Plot the counterfactuals in same order
            # loop through the counterfactuals 
            # iq1 and iq2 are the indices for the left and right panels
            for k in np.arange(len(cf_name)):
                if k<=1:
                    iq1 = 1
                    iq2 = 0
                if k>1 and k<=3:
                    iq1 = 3
                    iq2 = 2
                if k>3 and k<=5:
                    iq1 = 5
                    iq2 = 4
                if k>5 and k<=7:
                    iq1 = 7
                    iq2 = 6
                if k == 1 or k == 4 or k == 5 or k ==7:
                    mrk_size = 7 
                else:
                    mrk_size = 8
                if k == 6: 
                    mrk_size = 9

                ys = moments_sim_gr10[k+1,:,i] 
                p, = ax[iq1].plot(xs, ys, marker = markers[k], color = 'grey', linestyle = "-",
                        label = cf_name[k], linewidth = 1.5, markersize = mrk_size)
                ys = moments_sim_gr20[k+1,:,i] 
                p, = ax[iq2].plot(xs, ys, marker = markers[k], color = 'grey', linestyle = "-",
                        label = cf_name[k], linewidth = 1.5, markersize = mrk_size)
                
                ax[0].set_title("WP-UG")
                ax[1].set_title("MP-DG")
                ax[2].set_title("WP-UG")
                ax[3].set_title("MP-DG")
                ax[4].set_title("WP-UG")
                ax[5].set_title("MP-DG")
                ax[6].set_title("WP-UG")
                ax[7].set_title("MP-DG")
                ax[iq2].spines['top'].set_visible(False)
                ax[iq2].spines['right'].set_visible(False)
                ax[iq2].set_xticks([1,2,3,4,5,6,7,8,9,10]) 
                ax[iq1].spines['top'].set_visible(False)
                ax[iq1].spines['right'].set_visible(False)
                ax[iq1].set_xticks([1,2,3,4,5,6,7,8,9,10]) 
                ax[iq1].set_xlabel('Week',fontsize=11)
                ax[iq2].set_xlabel('Week',fontsize=11)

            # Adjust the plot markets and axes to taste 
            for n in np.arange(8):
                if variable_list_names[i] == 'Knowledge accumulation':
                        if n == 0:
                            ax[n].set_ylim(0,100)
                            ax[n].set_yticks((0,50, 100))
                            ax[n].axhline(y= 35, color='grey', linestyle='-', linewidth=1)
                            ax[n].axhline(y= 70, color='grey', linestyle='-', linewidth=1)
                        if n == 2:
                            ax[n].set_ylim(0,100)
                            ax[n].set_yticks((0,50, 100))
                            ax[n].axhline(y= 35, color='grey', linestyle='-', linewidth=1)
                            ax[n].axhline(y= 70, color='grey', linestyle='-', linewidth=1)
                        if n == 1:
                            ax[n].set_ylim(0,100)
                            ax[n].set_yticks((0,50, 100))
                            ax[n].axhline(y= 35, color='grey', linestyle='-', linewidth=1)
                            ax[n].axhline(y= 70, color='grey', linestyle='-', linewidth=1)
                        if n == 3:
                            ax[n].set_ylim(0,100)
                            ax[n].set_yticks((0,50, 100))
                            ax[n].axhline(y= 35, color='grey', linestyle='-', linewidth=1)
                            ax[n].axhline(y= 70, color='grey', linestyle='-', linewidth=1)
                            #ax[n].axhline(y= 75, color='grey', linestyle='-', linewidth=1)
                        if n == 4:
                            ax[n].set_ylim(0,100)
                            ax[n].set_yticks((0,50, 100))
                            ax[n].axhline(y=35, color='grey', linestyle='-',linewidth=1)
                            ax[n].axhline(y=70, color='grey', linestyle='-', linewidth=1)
                        if n == 5:
                            ax[n].set_ylim(0,100)
                            ax[n].set_yticks((0,50, 100))
                            ax[n].axhline(y= 35, color='grey', linestyle='-', linewidth=1)
                            ax[n].axhline(y= 70, color='grey', linestyle='-', linewidth=1)
                        if n == 6:
                            ax[n].set_ylim(0,100)
                            ax[n].set_yticks((0,50, 100))
                            ax[n].axhline(y= 35, color='grey', linestyle='-', linewidth=1)
                            ax[n].axhline(y= 70, color='grey', linestyle='-', linewidth=1)
                        if n == 7:
                            ax[n].set_ylim(0,100)
                            ax[n].set_yticks((0,50, 100))
                            ax[n].axhline(y= 35, color='grey', linestyle='-', linewidth=1)
                            ax[n].axhline(y= 70, color='grey', linestyle='-', linewidth=1)
                            
                if variable_list_names[i] == 'Time earning happines units (hours)':
                        ax[n].set_ylim(0,50)
                        ax[n].axhline(y=20, color='grey', linestyle='-',linewidth=1)
                        ax[n].axhline(y=40, color='grey', linestyle='-', linewidth=1)

                if variable_list_names[i] == 'Time solving MCQs (hours)':
                        ax[n].set_ylim(0,20)
                        ax[n].axhline(y=8, color='grey', linestyle='-',linewidth=1)
                        ax[n].axhline(y=16, color='grey', linestyle='-', linewidth=1)
                
                if  variable_list_names[i] == 'Time answering SAQs (hours)':
                        ax[n].set_ylim(0,3)
                        ax[n].set_yticks((1,2, 3))
                        ax[n].axhline(y=1, color='grey', linestyle='-',linewidth=1)
                        ax[n].axhline(y=2, color='grey', linestyle='-', linewidth=1)

                if variable_list_names[i] == 'Time studying the textbook (hours)':
                        ax[n].set_ylim(0,60)
                        ax[n].set_yticks((1,2, 3))
                        ax[n].axhline(y=20, color='grey', linestyle='-',linewidth=1)
                        ax[n].axhline(y=40, color='grey', linestyle='-', linewidth=1)

                if variable_list_names[i] == 'Exam mark':
                        ax[n].set_ylim(23,35)
                        ax[n].set_yticks((23,28,32,35))
                        ax[n].axhline(y=26, color='grey', linestyle='-',linewidth=1)
                        ax[n].axhline(y=32, color='grey', linestyle='-', linewidth=1)


                if  variable_list_names[i] == 'Number of SAQ attempts':
                    if (n % 2) == 0:
                        ax[n].set_ylim(0,15)
                        ax[n].set_yticks((0,5, 10,15))
                        ax[n].axhline(y=5, color='grey', linestyle='-',linewidth=1)
                        ax[n].axhline(y=10, color='grey', linestyle='-', linewidth=1)
                    else:
                        ax[n].set_ylim(0,10)
                        ax[n].set_yticks((0,5, 10))
                        ax[n].axhline(y=3, color='grey', linestyle='-',linewidth=1)
                        ax[n].axhline(y=7, color='grey', linestyle='-', linewidth=1)


            fig.subplots_adjust(hspace=0.5, bottom=0.2, top = 1)

            ax1.flatten()[6].legend(loc='center', bbox_to_anchor=(1.1, -1.1),\
                                     ncol = 3,fontsize=10, frameon=False)
            ax1.flatten()[10].legend(loc='center', bbox_to_anchor=(1.1, -1),\
                                     ncol = 3,fontsize=10, frameon=False)
            ax1.flatten()[14].legend(loc='center', bbox_to_anchor=(1.1, -1),\
                                         ncol = 3,fontsize=10, frameon=False)
            ax1.flatten()[2].legend(loc='center', bbox_to_anchor=(1.1, -1),\
                                         ncol =3,fontsize=10, frameon=False)
            fig.suptitle(variable_list_names[i],y=1.1)
            ax1 = ax1.flatten()
            ax1[0].set_ylim([0.3,0.3]);ax1[0].axis("off")
            ax1[1].set_ylim([0.3,0.3]);ax1[1].axis("off")
            ax1[4].axis("off")
            ax1[5].axis("off")
            ax1[8].axis("off")
            ax1[9].axis("off")
            ax1[12].axis("off")
            ax1[13].axis("off")

            # Here are the list of CFs for which plots are generated
            if variable_list_names[i] == 'Knowledge accumulation':
                fig.savefig(plot_path + "{}.png".format(name + 'CFlvl'), transparent=True)

            plt.close()

if __name__ == "__main__":


    # Estimation name and path of results to save plots 
    estimation_name = 'Final_MSR2_v4'
    results_path = 'results/' + estimation_name
    
    # An ID for the plots is created based on the time the script is run
    ID = str(time.strftime("%Y%m%d-%H%M%S"))

    # Load the simulated data 
    moments_data_all = np.load('settings/estimates/{}/baseline/moments_data_all.npy'\
                                .format(estimation_name))
    moments_sim_all = np.load('settings/estimates/{}/baseline/moments_sim_all.npy'\
                                .format(estimation_name))
    plot_cfs_tau10 = np.load('settings/estimates/{}/counterfactuals/cf_moments_tau_10.npy'\
                                .format(estimation_name))
    plot_cfs_tau20 = np.load('settings/estimates/{}/counterfactuals/cf_moments_tau_20.npy'\
                                .format(estimation_name))


    # List of variable names used in plot
    # change name here if name change in 
    # plot is required 
    variable_names = ['Exam mark',\
            'Total mark',\
            'av_markw13_exp1',\
            'Knowledge accumulation',\
            'Time earning happines units (hours)',\
            'Time studying the textbook (hours)',\
            'Time solving MCQs (hours)',\
            'Time answering SAQs (hours)',\
            'Number of happines units',\
            'Number of MCQ attempts',\
            'Number of SAQ attempts', \
            'Number of textbook pages',\
                     'sd_final_grade',
                     'sd_course_grade',
                     'sd_final_grade_exp',
                     'sd_knowledge_cumul',
                     'sd_sim_session_hours_cumul',
                     'sd_ebook_session_hours_cumul',
                     'sd_mcq_session_hours_cumul',
                     'sd_saq_session_hours_cumul',
                     'sd_happy_deploym_cumul',
                     'sd_mcq_cumul',
                     'sd_saq_cumul',
                     'sd_totebook_pageviews_cumul',
                     'acsim_session_hours',
                     'acebook_session_hours',
                     'acmcq_session_hours',
                     'acsaq_session_hours',
                     'actotebook_pageviews',
                     'acmcq_Cattempt',
                     'co_mcsaq_session_hours',
                     'co_simsaq_session_hours',
                     'co_simmcq_session_hours',
                     'co_esaq_session_hours',
                     'co_emcq_session_hours',
                     'co_esim_session_hours',
                     'co_fsim_session_hours_cumul',
                     'co_febook_session_hours_cumul',
                     'co_fmcq_session_hours_cumul',
                     'co_fsaq_session_hours_cumul',
                     'co_fhappy_deploym_cumul',
                     'co_ftotebook_pageviews_cumul',
                     'co_fmcq_cumul',
                     'co_fsaq_cumul',
                     'co_sim',
                     'co_ebook',
                     'co_mcq',
                     'co_saq',
                     'cesim_session_hours',
                     'cemcq_session_hours',
                     'cesaq_session_hours',
                     'csimmcq_session_hours',
                     'csimsaq_session_hours',
                     'cmcsaq_session_hours',
                     'co_fatar_ii']

    # List of variables that are being plotted
    # do not change this list as it corressponds
    # to moment names used in the smm and profile
    # simulation 

    list_moments = ['av_final_grade',
                     'av_course_grade',
                     'av_final_grade_exp',
                     'av_knowledge_cumul',
                     'av_sim_session_hours_cumul',
                     'av_ebook_session_hours_cumul',
                     'av_mcq_session_hours_cumul',
                     'av_saq_session_hours_cumul',
                     'av_happy_deploym_cumul',
                     'av_mcq_cumul',
                     'av_saq_cumul',
                     'av_totebook_pageviews_cumul',
                     'sd_final_grade',
                     'sd_course_grade',
                     'sd_final_grade_exp',
                     'sd_knowledge_cumul',
                     'sd_sim_session_hours_cumul',
                     'sd_ebook_session_hours_cumul',
                     'sd_mcq_session_hours_cumul',
                     'sd_saq_session_hours_cumul',
                     'sd_happy_deploym_cumul',
                     'sd_mcq_cumul',
                     'sd_saq_cumul',
                     'sd_totebook_pageviews_cumul',
                     'acsim_session_hours',
                     'acebook_session_hours',
                     'acmcq_session_hours',
                     'acsaq_session_hours',
                     'actotebook_pageviews',
                     'acmcq_Cattempt',
                     'co_mcsaq_session_hours',
                     'co_simsaq_session_hours',
                     'co_simmcq_session_hours',
                     'co_esaq_session_hours',
                     'co_emcq_session_hours',
                     'co_esim_session_hours',
                     'co_fsim_session_hours_cumul',
                     'co_febook_session_hours_cumul',
                     'co_fmcq_session_hours_cumul',
                     'co_fsaq_session_hours_cumul',
                     'co_fhappy_deploym_cumul',
                     'co_ftotebook_pageviews_cumul',
                     'co_fmcq_cumul',
                     'co_fsaq_cumul',
                     'co_sim',
                     'co_ebook',
                     'co_mcq',
                     'co_saq',
                     'cesim_session_hours',
                     'cemcq_session_hours',
                     'cesaq_session_hours',
                     'csimmcq_session_hours',
                     'csimsaq_session_hours',
                     'cmcsaq_session_hours',
                     'co_fatar_ii']

    list_moments_plot = ['av_sim_session_hours_cumul',
                     'av_ebook_session_hours_cumul',
                     'av_mcq_session_hours_cumul',
                     'av_saq_session_hours_cumul',
                     'av_happy_deploym_cumul',
                     'av_mcq_cumul',
                     'av_saq_cumul',
                     'av_totebook_pageviews_cumul']


    # Treatment group names plotted within each of the four
    # sub-plots for the baseline
    group_names = ['Female pres., gender disc. vs. unsdisc.',\
                   'Female pres., gender disc. vs. unsdisc.',\
                    'Gender disc., female vs. male pres.',\
                    'Gender disc., female vs. male pres.']

    # The tau indices of each of the treatment groups 
    # plotted in the baseline each sub-tuple is a panel
    group_diffs_ind = [[6, 4], [7, 5],\
                [6, 2], [7,3]]

    # Plots for the baseline 
    plot_results_paper2(moments_sim_all, moments_data_all, list_moments, variable_names,\
                             group_names, group_diffs_ind,results_path,list_moments_plot,ID)

    # Plots for the counterfactuals     
    plot_cfs_lvl(plot_cfs_tau10,plot_cfs_tau20,list_moments,variable_names,results_path,ID)

    # Now save the counterfactuals as csv files 
    cf_names = ['baseline', 'ambition','patience','self-control',\
                        'Exam difficulty','ability', 'perc. ability',\
                        'Effort cost SAQ', 'All effort costs']

    csv_path = results_path + "/plots/counterfactuals/plot_paper2_cflvl_{}/csv/".format(ID)
    
    Path(csv_path).mkdir(parents=True, exist_ok=True)
    
    for i, v_name in zip(np.arange(len(variable_names)), variable_names):
        df = pd.DataFrame(plot_cfs_tau10[:,:,i].transpose(), columns =cf_names) 
        df = df.add_suffix('_gr10' + '_' + v_name)
        df20 = pd.DataFrame(plot_cfs_tau20[:,:,i].transpose(), columns =cf_names) 
        df20 = df20.add_suffix('_gr20' + '_' + v_name)
        df = pd.concat([df, df20], axis = 1)
        df.insert(0, 'Week', range(1,12))
        df.to_csv(csv_path + "results_cfs_{}.csv".format(v_name),index=False)