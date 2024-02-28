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

    ##NEW - added values to specify how long nurse is unavailable and at what
    # frequency (in this example, every 2 hours, the nurse will be unavailable
    # for 15 minutes)
    unav_time_nurse = 15
    unav_freq_nurse = 120

    # Simulation meta parameters
    sim_duration = 2880
    number_of_runs = 1
    warm_up_period = 1440

# Class representing patients coming in to the clinic.
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = 0
        self.priority = random.randint(1,5)

# Class representing our model of the clinic.
class Model:
    # Constructor
    def __init__(self, run_number):
        # Set up SimPy environment
        self.env = simpy.Environment()

        # Set up counters to use as entity IDs
        self.patient_counter = 0

        # Set up resources
        self.nurse = simpy.PriorityResource(self.env, 
                                            capacity=g.number_of_nurses)

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

    ##NEW
    # Generator function to obstruct a nurse resource at specified intervals
    # for specified amounts of time
    def obstruct_nurse(self):
        while True:
            print ("The nurse will go on a break at around time",
                   f"{self.env.now + g.unav_freq_nurse}")
            
            # The generator first pauses for the frequency period
            yield self.env.timeout(g.unav_freq_nurse)

            # Once elapsed, the generator requests (demands?) a nurse with
            # a priority of -1.  This ensure it takes priority over any patients
            # (whose priority values start at 1).  But it also means that the
            # nurse won't go on a break until they've finished with the current
            # patient
            with self.nurse.request(priority=-1) as req:
                yield req

                print ("The nurse is now on a break and will be back at",
                       f"{self.env.now + g.unav_time_nurse}")
                
                # Freeze with the nurse held in place for the unavailability
                # time (ie duration of the nurse's break).  Here, both the
                # duration and frequency are fixed, but you could randomly
                # sample them from a distribution too if preferred.
                yield self.env.timeout(g.unav_time_nurse)
                
    # Generator function representing pathway for patients attending the
    # clinic.
    def attend_clinic(self, patient):
        # Nurse consultation activity
        start_q_nurse = self.env.now

        with self.nurse.request(priority=patient.priority) as req:
            yield req

            end_q_nurse = self.env.now

            patient.q_time_nurse = end_q_nurse - start_q_nurse

            if self.env.now > g.warm_up_period:
                self.results_df.at[patient.id, "Q Time Nurse"] = (
                    patient.q_time_nurse
                )

            sampled_nurse_act_time = random.expovariate(1.0 / 
                                                        g.mean_n_consult_time)

            yield self.env.timeout(sampled_nurse_act_time)

    # Method to calculate and store results over the run
    def calculate_run_results(self):
        self.results_df.drop([1], inplace=True)

        self.mean_q_time_nurse = self.results_df["Q Time Nurse"].mean()

    # Method to run a single run of the simulation
    def run(self):
        # Start up DES generators
        self.env.process(self.generator_patient_arrivals())
        ##NEW - we also need to start up the obstructor generator now too
        self.env.process(self.obstruct_nurse())

        # Run for the duration specified in g class
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

