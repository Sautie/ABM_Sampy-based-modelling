from ORM_related_addons.graph_from_ORM_xml import GraphFromORMxml
from ORM_related_addons.ORM_like_agents import ORMMongooses
from sampy.data_processing.csv_manager import CsvManager
import numpy as np


# ------------------------
# PARAMETERS

path_to_xml_landscape = "D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/PR_base_landscape_190318.xml"
path_to_csv_of_parameters = "D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/sampy_par_CS_popbuild_221216.csv"
path_to_output_pop_csv = "D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/pop_build_up_3.csv"

# ------------------------
# READ THE CSV OF PARAMETERS

dict_types = {'nb_years_sim': int,
              'pregnancy_duration': int,
              'max_age_mongoose_in_year': int,
              'age_independence': int,
              'mortality_male': float,
              'mortality_female': float,
              'old_mortality_bias': float,
              'nb_offsprings': int,
              'prob_nb_offsprings': float,
              'age_adult': int,
              'age_mature_adult': int,
              'prob_failure_pregnancy_adult': float,
              'prob_failure_pregnancy_juv': float,
              'min_age_reproduction': int,
              'raw_mean_mate_week': int,
              'raw_var_mate_week': int,
              'dispersion_female_juv_week': bool,
              'dispersion_male_juv_week': bool,
              'dispersion_female_adult_week': bool,
              'dispersion_male_adult_week': bool,
              'dispersion_female_juv_prob_by_nb_step': float,
              'dispersion_male_juv_prob_by_nb_step': float,
              'dispersion_female_adult_prob_by_nb_step': float,
              'dispersion_male_adult_prob_by_nb_step': float}

csv_manager = CsvManager(path_to_csv_of_parameters, ',', dict_types=dict_types)
param = csv_manager.get_parameters()
#print(param.nb_years_sim)

# ------------------------
# add the missing parameters
arr_weekly_mortality_male = []
for x in param.mortality_male:
    for _ in range(52):
        arr_weekly_mortality_male.append(x)
arr_weekly_mortality_male = np.array(arr_weekly_mortality_male)
arr_weekly_mortality_male = 1 - (1 - arr_weekly_mortality_male) ** (1. / 52.)

arr_weekly_mortality_female = []
for x in param.mortality_female:
    for _ in range(52):
        arr_weekly_mortality_female.append(x)
arr_weekly_mortality_female = np.array(arr_weekly_mortality_female)
arr_weekly_mortality_female = 1 - (1 - arr_weekly_mortality_female) ** (1. / 52.)

list_ind_mate_week = [i for i, val in enumerate(param.raw_mean_mate_week) if val != 0]
# param.mean_mate_week = np.array([param.raw_mean_mate_week[i] for i in list_ind_mate_week])
# param.var_mate_week = np.array([param.raw_var_mate_week[i] for i in list_ind_mate_week])

param.mean_mate_week = np.array([1, 15, 30, 45])
param.var_mate_week = np.array([4, 4, 4, 4])

param.allowed_dispersion_week_female_juv = [i for i, val in enumerate(param.dispersion_female_juv_week) if val]
param.allowed_dispersion_week_male_juv = [i for i, val in enumerate(param.dispersion_male_juv_week) if val]
param.allowed_dispersion_week_female_adult = [i for i, val in enumerate(param.dispersion_female_adult_week) if val]
param.allowed_dispersion_week_male_adult = [i for i, val in enumerate(param.dispersion_male_adult_week) if val]

param.dispersion_female_juv_nb_step = np.arange(start=0, stop=len(param.dispersion_female_juv_prob_by_nb_step), step=1)
param.dispersion_male_juv_nb_step =np.arange(start=0, stop=len(param.dispersion_male_juv_prob_by_nb_step), step=1)
param.dispersion_female_adult_nb_step =np.arange(start=0, stop=len(param.dispersion_female_adult_prob_by_nb_step), step=1)
param.dispersion_male_adult_nb_step  =np.arange(start=0, stop=len(param.dispersion_male_adult_prob_by_nb_step), step=1)

# ------------------------
# START THE BUILD UP

# create the map
landscape = GraphFromORMxml(path_to_xml=path_to_xml_landscape)

# create the population object
mongooses = ORMMongooses(pregnancy_duration=param.pregnancy_duration,
                         graph=landscape)

# create the initial population
nb_initial_couples = landscape.number_vertices
dict_new_agents = dict()
dict_new_agents['age'] = [52 for _ in range(nb_initial_couples * 2)]
dict_new_agents['gender'] = [i % 2 for i in range(nb_initial_couples * 2)]
dict_new_agents['territory'] = [i // 2 for i in range(nb_initial_couples * 2)]
dict_new_agents['position'] = [i // 2 for i in range(nb_initial_couples * 2)]
mongooses.add_agents(dict_new_agents)

for week in range(param.nb_years_sim * 52 + 1):
    if week%4==0:
        print(week)
    if mongooses.df_population.nb_rows == 0:
       raise ValueError('Population died off at week ' + str(week) + '.')

    mongooses.tick()
    mongooses.increment_number_of_weeks_of_pregnancy()
    landscape.tick()

    # kill population
    mongooses.kill_too_old(param.max_age_mongoose_in_year * 52)

    if mongooses.df_population.nb_rows == 0:
       raise ValueError('Population died off at week ' + str(week) + '.')

    count_in_k = mongooses.df_population['age'] >= param.age_independence
    mongooses.natural_death_orm_methodology(arr_weekly_mortality_male, arr_weekly_mortality_female,
                                            condition_count=count_in_k, bias=param.old_mortality_bias)
    mongooses.kill_children_whose_mother_is_dead(param.age_independence)

    if mongooses.df_population.nb_rows == 0:
        raise ValueError('Population died off at week ' + str(week) + '.')

    # reproduction
    adult = mongooses.df_population['age'] >= param.age_mature_adult
    mongooses.give_birth_if_needed(param.nb_offsprings, param.prob_nb_offsprings,
                                   condition=adult, prob_failure=param.prob_failure_pregnancy_adult)

    yy = (mongooses.df_population['age'] < param.age_mature_adult) & \
         (mongooses.df_population['age'] >= param.age_adult)
    mongooses.give_birth_if_needed(param.nb_offsprings, param.prob_nb_offsprings,
                                   condition=yy, prob_failure=param.prob_failure_pregnancy_juv)
    mongooses.weekly_mating_checks_and_update(week % 52 + 1, param.mean_mate_week, param.var_mate_week,
                                              param.age_independence, param.min_age_reproduction)

    # add dispersion
    adult = mongooses.df_population['age'] >= param.age_mature_adult
    yy = ~adult & (mongooses.df_population['age'] >= param.age_independence) #independance independence
    male = mongooses.get_males()

    male_yy = male & yy
    female_yy = ~male & yy
    male_adult = male & adult
    female_adult = ~male & adult

    mongooses.orm_dispersion_with_resistance(week, param.allowed_dispersion_week_male_juv,
                                             male_yy,
                                             param.dispersion_male_juv_nb_step,
                                             param.dispersion_male_juv_prob_by_nb_step)
    mongooses.orm_dispersion_with_resistance(week, param.allowed_dispersion_week_female_juv,
                                             female_yy,
                                             param.dispersion_female_juv_nb_step,
                                             param.dispersion_female_juv_prob_by_nb_step)

    mongooses.orm_dispersion_with_resistance(week, param.allowed_dispersion_week_male_adult,
                                             male_adult,
                                             param.dispersion_male_adult_nb_step,
                                             param.dispersion_male_adult_prob_by_nb_step)

    mongooses.orm_dispersion_with_resistance(week, param.allowed_dispersion_week_female_adult,
                                             female_adult,
                                             param.dispersion_female_adult_nb_step,
                                             param.dispersion_female_adult_prob_by_nb_step)


mongooses.save_population_to_csv(path_to_output_pop_csv)
