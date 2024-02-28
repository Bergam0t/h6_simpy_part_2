import simpy
import random
import pandas as pd
import Lognormal
import matplotlib.pyplot as plt ##NEW - import matplotlib for graphs

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

    # Maximum allowable queue lengths
    max_q_nurse = 10 ##NEW - increased to show variability in graph better

    # Simulation meta parameters
    sim_duration = 2880
    number_of_runs = 1 ##NEW - updated to 1 run so we can easily see results
    warm_up_period = 1440
   
# Class representing patients coming in to the clinic.
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = 0
        self.priority = random.randint(1,5)
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

        # Set up attributes that will store queuing behaviour results across
        # run
        self.num_reneged_nurse = 0
        self.num_balked_nurse = 0

        # Set up lists to store patient objects in each queue
        self.q_for_nurse_consult = []

        ##NEW - added Pandas dataframe to record number in queue(s) over time
        self.queue_df = pd.DataFrame()
        self.queue_df["Time"] = [0.0]
        self.queue_df["Num in Q Nurse"] = [0]

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
        # Check if current queue length is less than maximum
        if len(self.q_for_nurse_consult) < g.max_q_nurse:
            # Nurse consultation activity
            start_q_nurse = self.env.now

            self.q_for_nurse_consult.append(patient)

            ##NEW - as we've added a patient to the queue, we now need to record
            # the current time against the number in the queue.  We use .loc and
            # give the index as the length of the queue to add the row to the
            # end of the dataframe.  Note - we'd need to add additional items
            # to the list forming the row if we were tracking more than one
            # queue.  Also note, we will only add queue lengths to be plot if
            # they are after the warm-up period.
            if self.env.now > g.warm_up_period:
                self.queue_df.loc[len(self.queue_df)] = [
                    self.env.now,
                    len(self.q_for_nurse_consult)
                ]

            with self.nurse.request(priority=patient.priority) as req:
                result_of_queue = (yield req | 
                                self.env.timeout(patient.patience_nurse))

                self.q_for_nurse_consult.remove(patient)

                ##NEW - as we've removed a patient from the queue, we now need
                # to record the current time against the number in the queue.  
                # We use .loc and give the index as the length of the queue to 
                # add the row to the end of the dataframe.  Note - we'd need to 
                # add additional items to the list forming the row if we were 
                # tracking more than one queue.  Also note, we will only add 
                # queue lengths to be plot if they are after the warm-up period.
                if self.env.now > g.warm_up_period:
                    self.queue_df.loc[len(self.queue_df)] = [
                        self.env.now,
                        len(self.q_for_nurse_consult)
                    ]
                
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
        else:
            self.num_balked_nurse += 1

    # Method to calculate and store results over the run
    def calculate_run_results(self):
        self.results_df.drop([1], inplace=True)

        self.mean_q_time_nurse = self.results_df["Q Time Nurse"].mean()

    ##NEW - method to plot and display queue lengths over time.  We plot the
    # data stored in our queue dataframe.  Here, we only have one queue, but if
    # we had more than one queue we could add it as another plot to the figure
    # with a different colour and / or linestyle and label.  "Time" will be the 
    # x-axis for all plots.
    def plot_queue_graphs(self):
        ##NEW drop the first element from the DataFrame as we don't want to plot
        # that first dummy entry that won't get used with a warm-up period
        self.queue_df.drop([0], inplace=True)

        fig, ax = plt.subplots()

        ax.set_xlabel("Time")
        ax.set_ylabel("Number of patients in queue")

        ax.plot(self.queue_df["Time"],
                self.queue_df["Num in Q Nurse"],
                color="red",
                linestyle="-",
                label="Q for Nurse Consultation")
        
        ax.legend(loc="upper right")
        
        fig.show()
    
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
        print (f"{self.num_reneged_nurse} patients reneged from nurse queue")
        print (f"{self.num_balked_nurse} patients balked at the nurse queue")
        ##NEW - we print the queues over time dataframe for this run
        print ("Queues over time")
        print (self.queue_df)

        ##NEW - call our new method to plot queue lengths over time
        self.plot_queue_graphs()

# Class representing a Trial for our simulation
class Trial:
    # Constructor
    def  __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Mean Q Time Nurse"] = [0.0]
        self.df_trial_results["Reneged Q Nurse"] = [0]
        self.df_trial_results["Balked Q Nurse"] = [0]
        self.df_trial_results.set_index("Run Number", inplace=True)

    # Method to calculate and store means across runs in the trial
    def calculate_means_over_trial(self):
        self.mean_q_time_nurse_trial = (
            self.df_trial_results["Mean Q Time Nurse"].mean()
        )

        self.mean_reneged_q_nurse = (
            self.df_trial_results["Reneged Q Nurse"].mean()
        )

        self.mean_balked_q_nurse = (
            self.df_trial_results["Balked Q Nurse"].mean()
        )
    
    # Method to print trial results, including averages across runs
    def print_trial_results(self):
        print ("Trial Results")
        print (self.df_trial_results)

        print (f"Mean Q Nurse : {self.mean_q_time_nurse_trial:.1f} minutes")
        print (f"Mean Reneged Q Nurse : {self.mean_reneged_q_nurse} patients")
        print (f"Mean Balked Q Nurse : {self.mean_balked_q_nurse} patients")

    # Method to run trial
    def run_trial(self):
        for run in range(g.number_of_runs):
            my_model = Model(run)
            my_model.run()
            
            self.df_trial_results.loc[run] = [my_model.mean_q_time_nurse,
                                              my_model.num_reneged_nurse,
                                              my_model.num_balked_nurse]

        self.calculate_means_over_trial()
        self.print_trial_results()

# Create new instance of Trial and run it
my_trial = Trial()
my_trial.run_trial()

