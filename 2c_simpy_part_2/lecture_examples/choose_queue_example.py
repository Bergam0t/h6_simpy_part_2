import simpy
import random
import pandas as pd
import Lognormal
import matplotlib.pyplot as plt

# Class to store global parameter values.
class g:
    # Inter-arrival times
    patient_inter = 2 ##NEW - decreased time to generate more frequent arrivals

    # Activity times
    mean_n_consult_time = 6
    sd_n_consult_time = 1

    mean_d_consult_time = 5 ##NEW - added mean consult time for doctor
    sd_d_consult_time = 3 ##NEW - added SD consult time for doctor

    # Resource numbers
    number_of_nurses = 1
    number_of_doctors = 1 ##NEW - added parameter to store number of doctors

    # Resource unavailability duration and frequency
    unav_time_nurse = 15
    unav_freq_nurse = 120

    # Maximum allowable queue lengths
    max_q_nurse = 10

    # Simulation meta parameters
    sim_duration = 480 ##NEW significantly shortened so can see clear queue plot
    number_of_runs = 1
    warm_up_period = 1440
   
# Class representing patients coming in to the clinic.
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = 0
        self.q_time_doc = 0 ##NEW - attribute to store queuing time for doctor
        self.priority = random.randint(1,5)
        self.patience_nurse = random.randint(5, 50)
        ##NEW - added random allocation of patience level to see doctor
        self.patience_doctor = random.randint(20, 100)

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
        ##NEW - added doctor resource also as PriorityResource
        self.doctor = simpy.PriorityResource(self.env,
                                             capacity=g.number_of_doctors)

        # Set run number from value passed in
        self.run_number = run_number

        # Set up DataFrame to store patient-level results
        self.results_df = pd.DataFrame()
        self.results_df["Patient ID"] = [1]
        self.results_df["Q Time Nurse"] = [0.0]
        ##NEW - added column to store queuing time for doctor for each patient
        self.results_df["Q Time Doctor"] = [0.0]
        self.results_df.set_index("Patient ID", inplace=True)

        # Set up attributes that will store mean queuing times across the run
        self.mean_q_time_nurse = 0
        self.mean_q_time_doctor = 0 ##NEW - store mean q time for doctor

        # Set up attributes that will store queuing behaviour results across
        # run
        self.num_reneged_nurse = 0
        self.num_balked_nurse = 0
        ##NEW - added equivalent queuing behaviour attributes for doctor
        # though no balking should occur for the doctor or the nurse in this
        # scenario - if there is no capacity in the nurse queue, the patient
        # will join the doctor queue, which has no limit
        self.num_reneged_doctor = 0
        self.num_balked_doctor = 0

        # Set up lists to store patient objects in each queue
        self.q_for_nurse_consult = []
        self.q_for_doc_consult = [] ##NEW - list to store queue for doctor

        # Pandas dataframe to record number in queue(s) over time
        self.queue_df = pd.DataFrame()
        self.queue_df["Time"] = [0.0]
        self.queue_df["Num in Q Nurse"] = [0]
        self.queue_df["Num in Q Doctor"] = [0] ##NEW added column for doctor

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
        ##NEW - check whether queue for the nurse is shorter than the queue for
        # the doctor AND that there is space in the nurse's queue (which is
        # constrained).  If both of these are true, then join the queue for the
        # nurse, otherwise join the queue for the doctor.
        if ((len(self.q_for_nurse_consult) < len(self.q_for_doc_consult)) and
            (len(self.q_for_nurse_consult) < g.max_q_nurse)):
            # Nurse consultation activity
            start_q_nurse = self.env.now

            self.q_for_nurse_consult.append(patient)

            # Record number in queue alongside the current time
            ##NEW need to also add length of current queue for doctor to the
            # list (need to add both even though this is just an update to the
            # length of the nurse list)
            if self.env.now > g.warm_up_period:
                self.queue_df.loc[len(self.queue_df)] = [
                    self.env.now,
                    len(self.q_for_nurse_consult),
                    len(self.q_for_doc_consult)
                ]

            with self.nurse.request(priority=patient.priority) as req:
                result_of_queue = (yield req | 
                                self.env.timeout(patient.patience_nurse))

                self.q_for_nurse_consult.remove(patient)

                # Record number in queue alongside the current time
                ##NEW need to also add length of current queue for doctor to the
                # list (need to add both even though this is just an update to
                # the length of the nurse list)
                if self.env.now > g.warm_up_period:
                    self.queue_df.loc[len(self.queue_df)] = [
                        self.env.now,
                        len(self.q_for_nurse_consult),
                        len(self.q_for_doc_consult)
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
            ##NEW - logic for patient to join queue for the doctor instead.
            # In this system, there should be no balking as if the queue for the
            # nurse has no more capacity, they'll just see the doctor which
            # doesn't have a limit.

            # Doctor consultation activity
            start_q_doc = self.env.now

            self.q_for_doc_consult.append(patient)

            # Record number in queue alongside the current time
            if self.env.now > g.warm_up_period:
                self.queue_df.loc[len(self.queue_df)] = [
                    self.env.now,
                    len(self.q_for_nurse_consult),
                    len(self.q_for_doc_consult)
                ]

            with self.doctor.request(priority=patient.priority) as req:
                result_of_queue = (yield req | 
                                self.env.timeout(patient.patience_doctor))

                self.q_for_doc_consult.remove(patient)

                # Record number in queue alongside the current time
                if self.env.now > g.warm_up_period:
                    self.queue_df.loc[len(self.queue_df)] = [
                        self.env.now,
                        len(self.q_for_nurse_consult),
                        len(self.q_for_doc_consult)
                    ]
                
                if req in result_of_queue:
                    end_q_doc = self.env.now

                    patient.q_time_doc = end_q_doc - start_q_doc

                    if self.env.now > g.warm_up_period:
                        self.results_df.at[patient.id, "Q Time Doctor"] = (
                            patient.q_time_doc
                        )

                    sampled_doc_act_time = Lognormal.Lognormal(
                        g.mean_d_consult_time, g.sd_d_consult_time).sample()

                    yield self.env.timeout(sampled_doc_act_time)
                else:
                    self.num_reneged_doctor += 1

    # Method to calculate and store results over the run
    def calculate_run_results(self):
        self.results_df.drop([1], inplace=True)

        self.mean_q_time_nurse = self.results_df["Q Time Nurse"].mean()
        ##NEW - added calculation for mean queuing time for doctor
        self.mean_q_time_doctor = self.results_df["Q Time Doctor"].mean()

    # Method to plot and display queue lengths over time
    def plot_queue_graphs(self):
        # Drop first dummy entry from queue dataframe
        self.queue_df.drop([0], inplace=True)

        fig, ax = plt.subplots()

        ax.set_xlabel("Time")
        ax.set_ylabel("Number of patients in queue")

        ax.plot(self.queue_df["Time"],
                self.queue_df["Num in Q Nurse"],
                color="red",
                linestyle="-",
                label="Q for Nurse Consultation")
        
        ##NEW added second plot on same graph to show queue size for doctor over
        # time
        ax.plot(self.queue_df["Time"],
                self.queue_df["Num in Q Doctor"],
                color="blue",
                linestyle="--",
                label="Q for Doctor Consultation")
        
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
        ##NEW added print statements for reneging and balking from doctor queue
        print (f"{self.num_reneged_doctor} patients reneged from the doctor",
               "queue")
        print (f"{self.num_balked_doctor} patients balked at the doctor queue")
        # Print queues over time dataframe for this run
        print ("Queues over time")
        print (self.queue_df)

        # Call method to plot queues over time
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
        ##NEW added columns to store number trial results relating to doctor
        self.df_trial_results["Mean Q Time Doctor"] = [0.0]
        self.df_trial_results["Reneged Q Doctor"] = [0]
        self.df_trial_results["Balked Q Doctor"] = [0]
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

        ##NEW added calculations for doctor queue and activity across trial
        self.mean_q_time_doc_trial = (
            self.df_trial_results["Mean Q Time Doctor"].mean()
        )

        self.mean_reneged_q_doc = (
            self.df_trial_results["Reneged Q Doctor"].mean()
        )

        self.mean_balked_q_doc = (
            self.df_trial_results["Balked Q Doctor"].mean()
        )
    
    # Method to print trial results, including averages across runs
    def print_trial_results(self):
        print ("Trial Results")
        print (self.df_trial_results)

        print (f"Mean Q Nurse : {self.mean_q_time_nurse_trial:.1f} minutes")
        print (f"Mean Reneged Q Nurse : {self.mean_reneged_q_nurse} patients")
        print (f"Mean Balked Q Nurse : {self.mean_balked_q_nurse} patients")

        ##NEW added print statements for trial results related to doctor
        print (f"Mean Q Doctor : {self.mean_q_time_doc_trial:.1f} minutes")
        print (f"Mean Reneged Q Doctor : {self.mean_reneged_q_doc} patients")
        print (f"Mean Balked Q Doctor : {self.mean_balked_q_doc} patients")

    # Method to run trial
    def run_trial(self):
        for run in range(g.number_of_runs):
            my_model = Model(run)
            my_model.run()
            
            ##NEW added doctor results to end of list of results to add for this
            # run
            self.df_trial_results.loc[run] = [my_model.mean_q_time_nurse,
                                              my_model.num_reneged_nurse,
                                              my_model.num_balked_nurse,
                                              my_model.mean_q_time_doctor,
                                              my_model.num_reneged_doctor,
                                              my_model.num_balked_doctor]

        self.calculate_means_over_trial()
        self.print_trial_results()

# Create new instance of Trial and run it
my_trial = Trial()
my_trial.run_trial()

