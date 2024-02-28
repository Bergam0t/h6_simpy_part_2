import simpy
import random
import pandas as pd

# Class to store global parameter values.
class g:
    # Inter-arrival times
    patient_inter = 5

    # Activity times
    mean_n_consult_time = 6

    # Resource numbers
    number_of_nurses = 1

    # Simulation meta parameters
    sim_duration = 2880
    number_of_runs = 100
    warm_up_period = 1440 ##NEW - this will be in addition to the sim_duration

# Class representing patients coming in to the clinic.
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = 0

# Class representing our model of the clinic.
class Model:
    # Constructor
    def __init__(self, run_number):
        # Set up SimPy environment
        self.env = simpy.Environment()

        # Set up counters to use as entity IDs
        self.patient_counter = 0

        # Set up resources
        self.nurse = simpy.Resource(self.env, capacity=g.number_of_nurses)

        # Set run number from value passed in
        self.run_number = run_number

        # Set up DataFrame to store patient-level results
        self.results_df = pd.DataFrame()
        self.results_df["Patient ID"] = [1]
        self.results_df["Q Time Nurse"] = [0.0]
        self.results_df.set_index("Patient ID", inplace=True)

        # Set up attributes that will store mean queuing times across the run
        self.mean_q_time_nurse = 0

    # Generator function that represents the DES generator for patient arrivals
    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            
            p = Patient(self.patient_counter)

            self.env.process(self.attend_clinic(p))

            sampled_inter = random.expovariate(1.0 / g.patient_inter)

            yield self.env.timeout(sampled_inter)

    # Generator function representing pathway for patients attending the
    # clinic.
    def attend_clinic(self, patient):
        # Nurse consultation activity
        start_q_nurse = self.env.now

        with self.nurse.request() as req:
            yield req

            end_q_nurse = self.env.now

            patient.q_time_nurse = end_q_nurse - start_q_nurse

            ##NEW - this checks whether the warm up period has passed before
            # adding any results
            if self.env.now > g.warm_up_period:
                self.results_df.at[patient.id, "Q Time Nurse"] = (
                    patient.q_time_nurse
                )

            sampled_nurse_act_time = random.expovariate(1.0 / 
                                                        g.mean_n_consult_time)

            yield self.env.timeout(sampled_nurse_act_time)

    # Method to calculate and store results over the run
    def calculate_run_results(self):
        ##NEW - as we now won't count the first patient, we need to remove
        # the dummy first patient result entry we created when we set up the
        # dataframe
        self.results_df.drop([1], inplace=True)

        self.mean_q_time_nurse = self.results_df["Q Time Nurse"].mean()

    # Method to run a single run of the simulation
    def run(self):
        # Start up DES generators
        self.env.process(self.generator_patient_arrivals())

        # Run for the duration specified in g class
        ##NEW - we need to tell the simulation to run for the specified duration
        # + the warm up period if we still want the specified duration in full
        self.env.run(until=(g.sim_duration + g.warm_up_period))

        # Calculate results over the run
        self.calculate_run_results()

        # Print patient level results for this run
        print (f"Run Number {self.run_number}")
        print (self.results_df)

# Class representing a Trial for our simulation
class Trial:
    # Constructor
    def  __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Mean Q Time Nurse"] = [0.0]
        self.df_trial_results.set_index("Run Number", inplace=True)

    # Method to calculate and store means across runs in the trial
    def calculate_means_over_trial(self):
        self.mean_q_time_nurse_trial = (
            self.df_trial_results["Mean Q Time Nurse"].mean()
        )
    
    # Method to print trial results, including averages across runs
    def print_trial_results(self):
        print ("Trial Results")
        print (self.df_trial_results)

        print (f"Mean Q Nurse : {self.mean_q_time_nurse_trial:.1f} minutes")

    # Method to run trial
    def run_trial(self):
        for run in range(g.number_of_runs):
            my_model = Model(run)
            my_model.run()
            
            self.df_trial_results.loc[run] = [my_model.mean_q_time_nurse]

        self.calculate_means_over_trial()
        self.print_trial_results()

# Create new instance of Trial and run it
my_trial = Trial()
my_trial.run_trial()

