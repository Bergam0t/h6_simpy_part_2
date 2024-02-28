import simpy
import random
import pandas as pd
import Lognormal

# Class to store global parameter values.
class g:
    # Inter-arrival times
    patient_inter = 5

    # Activity times
    mean_n_consult_time = 6
    sd_n_consult_time = 1

    # Resource numbers
    number_of_nurses = 1

    # Resource unavailability duration and frequency
    unav_time_nurse = 15
    unav_freq_nurse = 120

    # Simulation meta parameters
    sim_duration = 2880
    number_of_runs = 100
    warm_up_period = 1440
   
# Class representing patients coming in to the clinic.
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = 0
        self.priority = random.randint(1,5)

        ##NEW - added a new patience attribute of the patient.  This determines
        # how long the patient is prepared to wait for the nurse.  Here we just
        # randomly sample an integer between 5 and 50 (so the patient will be
        # prepared to wait for somewhere between 5 and 50 minutes in the queue),
        # but in a real world application you would probably want to have a
        # more refined way of allocating patience to patients (e.g basing
        # probabilities off prior data, or using a non-uniform named 
        # distribution).  You could have different patience levels for different
        # queues, or just a general patience level.  Or even get creative and
        # have a patience level that decreases the longer they've been in the
        # system!
        # If we want to see the effect of this, we can try changing the patience
        # levels - but you'll need to make the patience levels MUCH higher as
        # this system is in bad shape (remember, after 3 days patients are
        # waiting on average over 3 hours... and a lot are waiting much longer!)
        # Maybe try adding another nurse in to get the system under control
        # first!
        self.patience_nurse = random.randint(5, 50)

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

        ##NEW - we'll set up a new attribute that will store the number of
        # people that reneged from queues in the run (we only have one queue in
        # this model)
        self.num_reneged_nurse = 0

    # Generator function that represents the DES generator for patient arrivals
    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            
            p = Patient(self.patient_counter)

            self.env.process(self.attend_clinic(p))

            sampled_inter = random.expovariate(1.0 / g.patient_inter)

            yield self.env.timeout(sampled_inter)

    # Generator function to obstruct a nurse resource at specified intervals
    # for specified amounts of time
    def obstruct_nurse(self):
        while True:
            # The generator first pauses for the frequency period
            yield self.env.timeout(g.unav_freq_nurse)

            # Once elapsed, the generator requests (demands?) a nurse with
            # a priority of -1.  This ensure it takes priority over any patients
            # (whose priority values start at 1).  But it also means that the
            # nurse won't go on a break until they've finished with the current
            # patient
            with self.nurse.request(priority=-1) as req:
                yield req
                
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
            ##NEW - this statement now uses a vertical bar (|) / pipe as an "or"
            # statement.  It basically says "Wait for the request for the nurse
            # to be fulfilled OR until the patient's patience level has passed,
            # whichever comes first, and then store whatever the outcome was.
            result_of_queue = (yield req | 
                               self.env.timeout(patient.patience_nurse))

            ##NEW - we now need to check whether the patient waited or reneged,
            # as we could have got to this point of the generator function
            # either way.  We'll now only get them to see the nurse if they
            # waited.  If they didn't wait, we'll add to our counter of how
            # many patients reneged from the queue.
            if req in result_of_queue:
                end_q_nurse = self.env.now

                patient.q_time_nurse = end_q_nurse - start_q_nurse

                if self.env.now > g.warm_up_period:
                    self.results_df.at[patient.id, "Q Time Nurse"] = (
                        patient.q_time_nurse
                    )

                sampled_nurse_act_time = Lognormal.Lognormal(
                    g.mean_n_consult_time, g.sd_n_consult_time).sample()

                yield self.env.timeout(sampled_nurse_act_time)
            else:
                self.num_reneged_nurse += 1

                print (f"Patient {patient.id} reneged after waiting",
                       f"{patient.patience_nurse} minutes")

    # Method to calculate and store results over the run
    def calculate_run_results(self):
        self.results_df.drop([1], inplace=True)

        self.mean_q_time_nurse = self.results_df["Q Time Nurse"].mean()

    # Method to run a single run of the simulation
    def run(self):
        # Start up DES generators
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.obstruct_nurse())

        # Run for the duration specified in g class
        self.env.run(until=(g.sim_duration + g.warm_up_period))

        # Calculate results over the run
        self.calculate_run_results()

        # Print patient level results for this run
        print (f"Run Number {self.run_number}")
        print (self.results_df)
        ##NEW - we'll print out the number of patients that reneged from the
        # nurse queue in this run of the model.
        print (f"{self.num_reneged_nurse} patients reneged from nurse queue")

# Class representing a Trial for our simulation
class Trial:
    # Constructor
    def  __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Mean Q Time Nurse"] = [0.0]
        ##NEW - additional column of trial results to store the number of
        # patients that reneged in each run
        self.df_trial_results["Reneged Q Nurse"] = [0]
        self.df_trial_results.set_index("Run Number", inplace=True)

    # Method to calculate and store means across runs in the trial
    def calculate_means_over_trial(self):
        self.mean_q_time_nurse_trial = (
            self.df_trial_results["Mean Q Time Nurse"].mean()
        )

        ##NEW - we also now need to calculate the mean number of patients
        # reneging per run
        self.mean_reneged_q_nurse = (
            self.df_trial_results["Reneged Q Nurse"].mean()
        )
    
    # Method to print trial results, including averages across runs
    def print_trial_results(self):
        print ("Trial Results")
        print (self.df_trial_results)

        print (f"Mean Q Nurse : {self.mean_q_time_nurse_trial:.1f} minutes")
        ##NEW - we will also now print out the mean number of patients who
        # reneged from the nurse's queue per run
        print (f"Mean Reneged Q Nurse : {self.mean_reneged_q_nurse} patients")

    # Method to run trial
    def run_trial(self):
        for run in range(g.number_of_runs):
            my_model = Model(run)
            my_model.run()
            
            ##NEW - we also need to add the number of patients who reneged from
            # the nurse's queue as one of the results against each run
            self.df_trial_results.loc[run] = [my_model.mean_q_time_nurse,
                                              my_model.num_reneged_nurse]

        self.calculate_means_over_trial()
        self.print_trial_results()

# Create new instance of Trial and run it
my_trial = Trial()
my_trial.run_trial()

