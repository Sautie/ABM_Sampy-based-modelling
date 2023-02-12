from ORM_related_addons.graph_from_ORM_xml import GraphFromORMxml
from ORM_related_addons.ORM_like_agents import ORMMongooses
from sampy.data_processing.csv_manager import CsvManager
from sampy.data_processing.write_file import counts_to_csv
from experimental_addons.mongoose_disease import RabiesMongooses
import numpy as np
import numba as nb
import multiprocessing as mlp


# ------------------------
# PARAMETERS

path_to_xml_landscape = "D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/PR_base_landscape_190318.xml"
path_to_csv_of_parameters = "D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/sampy_par_CS_popbuild_221216.csv"
path_to_input_pop_csv = "D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/pop_build_up_2.csv"
path_folder_outputs = "D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/output"
nb_processes = 2


#@nb.njit
def extract_age_array(arr_age, max_age):
    r_arr = np.full((max_age + 1,), 0, dtype=int)
    for i in range(arr_age.shape[0]):
        r_arr[arr_age[i]] += 1
    return r_arr


def worker(id_proc, nb_cores, path_to_csv_param,
           path_to_csv_pop, path_to_xml, path_folder_outputs):

    rng_seed = 1789
    np.random.seed(rng_seed + id_proc)

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
                  'dispersion_male_adult_prob_by_nb_step': float,
                  'activity_area': float,
                  'transmission_probability': float,
                  'nb_week_incubation': int,
                  'incubation': float,
                  'rabies_indu_mort': float,
                  'nb_week_infectious_period': int,
                  'prob_infectious_period': float,
                  'week_of_extraction': int,
                  'week_of_initial_infection': int,
                  'level_initial_infection': float
                   # 'cell_inf_1': str,
                   # 'cell_inf_2': str,
                   # 'cell_inf_3': str,
                   # 'path_output_infected': str,
                   # 'path_output_pop_per_cell': str,
                   # 'path_output_demography_file': str
                  }  # not included here are cell_inf_1, cell_inf_2, cell_inf_3, path_output_infected,
                     # path_output_pop_per_cell and path_output_demography_file

    csv_manager = CsvManager(path_to_csv_param, ',', dict_types=dict_types, nb_cores=nb_cores, id_process=id_proc)
    param = csv_manager.get_parameters()

    while param is not None:

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
        param.mean_mate_week = np.array([param.raw_mean_mate_week[i] for i in list_ind_mate_week])
        param.var_mate_week = np.array([param.raw_var_mate_week[i] for i in list_ind_mate_week])

        param.allowed_dispersion_week_female_juv = [i for i, val in enumerate(param.dispersion_female_juv_week) if val]
        param.allowed_dispersion_week_male_juv = [i for i, val in enumerate(param.dispersion_male_juv_week) if val]
        param.allowed_dispersion_week_female_adult = [i for i, val in enumerate(param.dispersion_female_adult_week) if val]
        param.allowed_dispersion_week_male_adult = [i for i, val in enumerate(param.dispersion_male_adult_week) if val]

        param.dispersion_female_juv_nb_step = np.arange(start=0, stop=len(param.dispersion_female_juv_prob_by_nb_step),step=1)
        param.dispersion_male_juv_nb_step = np.arange(start=0, stop=len(param.dispersion_male_juv_prob_by_nb_step),step=1)
        param.dispersion_female_adult_nb_step = np.arange(start=0, stop=len(param.dispersion_female_adult_prob_by_nb_step), step=1)
        param.dispersion_male_adult_nb_step = np.arange(start=0, stop=len(param.dispersion_male_adult_prob_by_nb_step),step=1)

        # create the map
        landscape = GraphFromORMxml(path_to_xml=path_to_xml)

        # create the population object
        mongooses = ORMMongooses(pregnancy_duration=param.pregnancy_duration, graph=landscape)
        mongooses.load_population_from_csv(path_to_csv_pop)

        # create the disease object
        rabies = RabiesMongooses(host=mongooses, disease_name='rabies')

        # create the lists for extraction
        total_pop_per_cell = []
        total_inf_per_cell = []

        # create the demography csv
        with open(param.path_output_demography_file, 'a') as demog_out:
            demog_out.write("week;tot_pop;natural_deaths;rabies_death;average_age;nb_occupied_cells;" + \
                            ";".join(["age_" + str(q) for q in range(param.max_age_mongoose_in_year + 1)]) + \
                            ";tot_infected;tot_contagious\n")
        print(param.path_output_demography_file)
        for week in range(param.nb_years_sim * 52):
            if week%50==0:
                print(week)
            if mongooses.df_population.nb_rows == 0:
                with open(param.path_output_demography_file, 'a') as demog_out:
                    demog_out.write("Pop died out at week " + str(week))
                break

            if week % 52 + 1 == param.week_of_extraction:
                total_pop_per_cell.append(mongooses.count_pop_per_vertex(position_attribute="territory"))
                total_inf_per_cell.append(mongooses.count_pop_per_vertex(position_attribute="territory",
                                                                condition=mongooses.df_population['inf_rabies'] |
                                                                          mongooses.df_population['con_rabies']))

            mongooses.tick()
            mongooses.increment_number_of_weeks_of_pregnancy()
            landscape.tick()
            rabies.tick()

            nb_initial_pop = mongooses.df_population.nb_rows
            # kill population
            mongooses.kill_too_old(param.max_age_mongoose_in_year * 52)

            if mongooses.df_population.nb_rows == 0:
                with open(param.path_output_demography_file, 'a') as demog_out:
                    demog_out.write("Pop died out at week " + str(week))
                break

            if week + 1 == param.week_of_initial_infection:
                arr_new_contaminated = rabies.contaminate_vertices(
                                           [param.cell_inf_1, param.cell_inf_2, param.cell_inf_3],
                                           param.level_initial_infection,
                                           condition=mongooses.df_population['age'] >= param.age_independence)
                rabies.contaminate_dependant_youngs(arr_new_contaminated, param.age_independence)
                rabies.initialize_counters_of_newly_infected(arr_new_contaminated,
                                                             param.nb_week_incubation, param.incubation)

            count_in_k = mongooses.df_population['age'] >= param.age_independence
            mongooses.natural_death_orm_methodology(arr_weekly_mortality_male, arr_weekly_mortality_female,
                                                    condition_count=count_in_k, bias=param.old_mortality_bias)
            mongooses.kill_children_whose_mother_is_dead(param.age_independence)

            if mongooses.df_population.nb_rows == 0:
                with open(param.path_output_demography_file, 'a') as demog_out:
                    demog_out.write("Pop died out at week " + str(week))
                break

            pop_after_natural_death = mongooses.df_population.nb_rows

            # rabies
            mongooses.mov_around_territory(1. - param.activity_area)
            condition_concerned_by_rabies = mongooses.df_population['age'] > param.age_independence
            newly_infected = rabies.contact_contagion(param.transmission_probability, return_arr_new_infected=True,
                                                      condition=condition_concerned_by_rabies)
            rabies.contaminate_dependant_youngs(newly_infected, param.age_independence)
            rabies.initialize_counters_of_newly_infected(newly_infected, param.nb_week_incubation, param.incubation)
            rabies.transition_between_states('con', 'death', proba_death=param.rabies_indu_mort,
                                             return_transition_count=False)

            if mongooses.df_population.nb_rows == 0:
                with open(param.path_output_demography_file, 'a') as demog_out:
                    demog_out.write("Pop died out at week " + str(week))
                break

            pop_after_rabies_death = mongooses.df_population.nb_rows

            rabies.transition_between_states('con', 'imm')
            rabies.transition_between_states('inf', 'con', arr_nb_timestep=param.nb_week_infectious_period,
                                             arr_prob_nb_timestep=param.prob_infectious_period)

            # reproduction
            adult = mongooses.df_population['age'] >= param.age_mature_adult
            mongooses.give_birth_if_needed(param.nb_offsprings, param.prob_nb_offsprings,
                                           condition=adult, prob_failure=param.prob_failure_pregnancy_adult)

            yy = (mongooses.df_population['age'] < param.age_mature_adult) & \
                 (mongooses.df_population['age'] >= param.age_adult)
            mongooses.give_birth_if_needed(param.nb_offsprings, param.prob_nb_offsprings,
                                           condition=yy, prob_failure=param.prob_failure_pregnancy_juv)
            mongooses.weekly_mating_checks_and_update(week % 52, param.mean_mate_week, param.var_mate_week,
                                                      param.age_independence, param.min_age_reproduction)

            # add dispersion
            adult = mongooses.df_population['age'] >= param.age_mature_adult
            yy = ~adult & (mongooses.df_population['age'] >= param.age_independence)
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

            arr_count_age = extract_age_array(mongooses.df_population['age'] // 52, param.max_age_mongoose_in_year)
            #arr_count_age =mongooses.df_population['age'] // 52
            with open(param.path_output_demography_file, 'a') as demog_out:
                str_out = str(week) + ';' + str(mongooses.df_population.nb_rows) + ';' + \
                          str(nb_initial_pop - pop_after_natural_death) + ';' + \
                          str(pop_after_natural_death - pop_after_rabies_death) + ';' + \
                          str(mongooses.df_population['age'].sum()/float(pop_after_rabies_death)) + ';' + \
                          str(np.unique(mongooses.df_population['territory']).shape[0]) + ';' + \
                          ';'.join([str(count) for count in arr_count_age]) + ';' + \
                          str(mongooses.df_population['inf_rabies'].sum()) + ';' + \
                          str(mongooses.df_population['con_rabies'].sum()) + '\n'
                demog_out.write(str_out)
        path_output_infected="D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/output/Outputrab.txt"
        path_output_pop_per_cell="D:/WIAAgrosante/agenBasedResearchSAMPY/sampy-main-master/Caroline/scripts_and_sampy/output/pop_per_cell.txt"
        counts_to_csv(total_inf_per_cell, landscape, path_output_infected)
        counts_to_csv(total_inf_per_cell, landscape, path_output_pop_per_cell)

        param = csv_manager.get_parameters()


if __name__ == '__main__':
    list_job = []
    for m in range(nb_processes):
        list_job.append(mlp.Process(target=worker, args=(m, nb_processes, path_to_csv_of_parameters,
                                                         path_to_input_pop_csv, path_to_xml_landscape,
                                                         path_folder_outputs)))
        list_job[-1].start()

    for p in list_job:
        p.join()
