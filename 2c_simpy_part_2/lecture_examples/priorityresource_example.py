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

    ##NEW - I've shortened the run time, removed the warm up period and set to
    # 1 run so it's easier to see what's happening with the priority queues.
    # Simulation meta parameters
    sim_duration = 120
    number_of_runs = 1
    warm_up_period = 0

# Class representing patients coming in to the clinic.
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = 0
        ##NEW - here we add an attribute of the patient that determines their
        # priority (lower value = higher priority).  In this example, we just
        # randomly pick a value between 1 and 5, but you can use whatever logic
        # you like (in reality, you'd likely have probabilities to determine
        # what priority a patient is based on your data)
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
        ##NEW - here we set up the nurse as an instance of PriorityResource
        # rather than Resource
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

    # Generator function representing pathway for patients attending the
    # clinic.
    def attend_clinic(self, patient):
        # Nurse consultation activity
        start_q_nurse = self.env.now

        ##NEW - added a print message so we can see how priority works
        print (f"Patient {patient.id} with priority {patient.priority} is",
               "queuing for the nurse.")

        ##NEW - now that the nurse is set up as a PriorityResource, we can pass
        # in the value that we want it to look at to determine who's seen next
        # when we request the resource (here, that's the priority attribute of
        # the patient we set up in the Patient class)
        with self.nurse.request(priority=patient.priority) as req:
            yield req

            end_q_nurse = self.env.now

            ##NEW - added a print message so we can see how priority works
            print (f"Patient {patient.id} with priority {patient.priority} is",
                   "being seen.***")

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

