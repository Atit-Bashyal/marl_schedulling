

class PrimaryEquipment(object):
    def __init__(self, name, energydemand, rawprocessing, auxprocessing, inputresource, outputresource):
        self.name = name
        self.energydemand = energydemand  # Energy demand list  for different actions
        self.rawprocessing = rawprocessing  # list with input and output processing rate (for ech equipment input=output)
        self.auxprocessing = auxprocessing  # list with auxillary processing rate 
        self.inputresource = inputresource.strip().strip('"')  # The key for the input resource in the container
        self.outputresource = outputresource.strip().strip('"')  # The key for the output resource in the container

    def get_energy_demand(self, action):
        """
        Returns the energy demand for a given action.
        """
        return self.energydemand[action]
    
    def get_primary_production_value(self, containers, action):

        """
        Returns procced amounts of input and output resources
        """
        processed_amount = 0
        current_input_level = containers[self.inputresource]['value']
        available_input_material = max(current_input_level,0)
        current_output_level = containers[self.outputresource]['value']
        remaining_space = containers[self.outputresource]['max_cap'] - current_output_level
        

        if action == 0:
            if self.name == 'Reductionfurnace':
                new_amount = self.rawprocessing[action]
                if available_input_material != 0 :
                    if remaining_space != 0:
                        processed_amount = min(available_input_material, new_amount)
                        if remaining_space >= processed_amount:
                            return processed_amount
                        else:
                            processed_amount = remaining_space  
                            return processed_amount
                    else:
                        return processed_amount
                else:
                    return processed_amount
            else:
                return processed_amount
        
            
        
        else:
            new_amount = self.rawprocessing[action]

            if available_input_material != 0 :
                if remaining_space != 0:
                    processed_amount = min(available_input_material, new_amount)
                    if remaining_space >= processed_amount:
                        return processed_amount
                    else:
                        processed_amount = remaining_space  
                        return processed_amount
                else:
                    return processed_amount
            else:
                return processed_amount
        
    def get_primary_production_reward(self, containers, action):
        """
        Calculates the reward for processing raw materials into finished products.
        
        Args:
            containers (dict): Dictionary tracking input and output container states.
            action (int): Action index representing the processing level.

        Returns:
            float: The reward value based on the processing outcome.
        """
        current_input_level = containers[self.inputresource]['value']
        available_input_material = max(current_input_level,0)
        current_output_level = containers[self.outputresource]['value']
        remaining_space = containers[self.outputresource]['max_cap'] - current_output_level
        


        if action == 0:
            if available_input_material == 0:
                return 0 # No processing action taken; neutral reward
            else:
                return -1 # penalize idle state 
            
        
        else:
            self.new_amount = self.rawprocessing[action]
            processed_amount = min(available_input_material, self.new_amount)

            if available_input_material != 0 :
                if remaining_space >= processed_amount :
                        return 1 # Successful processing
                else:
                    return -1 # Penalize attempting to overfill
            else:
                return -1 # Penalize processing when no input is available
                    
    def get_water_value(self,containers,action):
        """
        Returns processed amount of auxillary resource.
        """
        self.consumed_amount_water  = self.auxprocessing['Coolwater'][action]
        self.available_input_water = max(containers['Coolwater']['value'],0)

        if self.consumed_amount_water > 0 and self.available_input_water > 0 :
            return - min(self.consumed_amount_water, self.available_input_water)
        else:
            return 0
        
    def get_nitrogen_value(self,containers,action):
        """
        Returns processed amount of auxillary resource.
        """
        self.consumed_amount_nitrogen = self.auxprocessing['Nitrogen'][action] 
        self.available_input_nitrogen = max(containers['Nitrogen']['value'],0)

        if self.consumed_amount_nitrogen > 0 and self.available_input_nitrogen > 0 :
            return - min(self.consumed_amount_nitrogen, self.available_input_nitrogen)
        else:
            return 0
        
    def run_equipment(self, containers, action):
        """
        Runs the primary equipment process based on the action provided.
        """

        # Determine the amount of energy needed to process at given action
        energy_demand = self.get_energy_demand(action)

        # Determine the amount processed resource the given action
        primary = self.get_primary_production_value(containers,action)
        prim_reward = self.get_primary_production_reward(containers,action)
        water = self.get_water_value(containers,action)
        nitrogen = self.get_nitrogen_value(containers,action)
        resource_processed = {self.inputresource : - primary , self.outputresource : primary,'Coolwater': water,'Nitrogen': nitrogen}


        return {'energy':energy_demand,'resource':resource_processed,'prod_rew':prim_reward}

    def reset(self):
        """
        add a reset function to clear or initialize any state variables if needed.
        """
        pass

class AuxiliaryEquipment(object):
    
    def __init__(self, name, energydemand, rawprocessing, auxprocessing, inputresource, outputresource):
        
        self.name = name
        self.energydemand = energydemand  # Energy demand for different actions
        self.rawprocessing = rawprocessing  # Amount of input resource needed for different actions
        self.auxprocessing = auxprocessing  # Auxiliary processing requirements for different actions
        self.inputresource = inputresource.strip().strip('"')   # Input resource key in the containers dictionary
        self.outputresource = outputresource.strip().strip('"')   # Output resource key in the containers dictionary

    def get_energy_demand(self, action):
        """
        Returns the energy demand for a given action.
        """
        return self.energydemand[action]
    
    def get_water_value(self,containers,action):
        
        """
        Returns reward to the agent for producting output resources
        """

        self.new_amount = self.auxprocessing['Coolwater'][action]
        self.remaining_space = containers['Coolwater']['max_cap'] - containers['Coolwater']['value']
   
        if self.remaining_space >= self.new_amount:
            return self.new_amount
        else:
            self.new_amount = 0    
            return self.new_amount
    
    def get_nitrogen_value(self,containers,action):
        
        """
        Returns reward to the agent for producting output resources
        """

        self.new_amount = self.auxprocessing['Nitrogen'][action]
        self.remaining_space = containers['Nitrogen']['max_cap'] - containers['Nitrogen']['value']
   
        if self.remaining_space >= self.new_amount:
            return self.new_amount
        else:
            self.new_amount = 0    
            return self.new_amount
    
    def get_nitrogen_production_reward(self,containers,action):
        
        """
        Returns reward to the agent for producting output resources
        """

        self.new_amount = self.auxprocessing['Nitrogen'][action]
        self.remaining_space = containers['Nitrogen']['max_cap'] - containers['Nitrogen']['value']
   
        # Reward logic
        if self.new_amount <= 0:
            # Neutral reward for no production
            return 0
        
        else:
            if self.remaining_space >= self.new_amount:
                # Reward for successful production
                return 1
        
            else:
                # Penalize overproduction that exceeds capacity
                return -1
        
    def get_water_production_reward(self,containers,action):
        
        """
        Returns reward to the agent for producting output resources
        """

        self.new_amount = self.auxprocessing['Coolwater'][action]
        self.remaining_space = containers['Coolwater']['max_cap'] - containers['Coolwater']['value']
   
        # Reward logic
        if self.new_amount <= 0:
            # Neutral reward for no production
            return 0
        
        else:
            if self.remaining_space >= self.new_amount:
                # Reward for successful production
                return 1
        
            else:
                # Penalize overproduction that exceeds capacity
                return -1

        
    def run_equipment(self, containers, action):
        """
        Runs the primary equipment process based on the action provided.
        """

        # Determine the amount of energy needed to process at given action
        energy_demand = self.get_energy_demand(action)

        water = self.get_water_value(containers,action)
        nitrogen = self.get_nitrogen_value(containers,action)
        prim_reward = self.get_water_production_reward(containers,action) + self.get_nitrogen_production_reward(containers,action)
        resource_processed = {'Coolwater': water, 'Nitrogen': nitrogen}

        return {'energy': energy_demand,'resource':resource_processed, 'prod_rew':prim_reward}

    def reset(self):
        """
        add a reset function to clear or initialize any state variables if needed.
        """
        pass

class EnergyStorageDevice(object):
    def __init__(self, capacity=300, discharge_efficiency=0.9, charge_efficiency=0.9, max_discharge_rate=50, max_charge_rate=50):
        """
        Initialize the Energy Storage Device.
        
        Args:
            name (str): Name of the energy storage device.
            capacity (float): Maximum energy storage capacity in kWh.
            discharge_efficiency (float): Efficiency of discharging (fraction, e.g., 0.9 for 90%).
            charge_efficiency (float): Efficiency of charging (fraction, e.g., 0.9 for 90%).
            max_discharge_rate (float): Maximum discharge rate in kWh per time step.
            max_charge_rate (float): Maximum charge rate in kWh per time step.
        """
        self.capacity = capacity
        self.discharge_efficiency = discharge_efficiency
        self.charge_efficiency = charge_efficiency
        self.max_discharge_rate = max_discharge_rate
        self.max_charge_rate = max_charge_rate
        self.current_energy = 0
    
    def take_action(self, action):
        """
        Takes an action to either charge or discharge the energy storage device.
        
        Args:
            action (float): Positive value for charging, negative value for discharging.
                           Value magnitude indicates the rate (kWh).
        
        Returns:
            dict: Contains the following keys:
                - 'energy_consumed': Energy consumed while charging (if charging).
                - 'energy_discharged': Energy discharged (if discharging).
                - 'energy_stored': Updated energy level in the storage device.
                - 'status': 'charging', 'discharging', or 'idle'.
        """
        if action == 1 :  # Charging
            charge_amount = self.max_charge_rate  # Limit by max charge rate
            energy_input = charge_amount * self.charge_efficiency  # Account for charging efficiency
            available_space = self.capacity - self.current_energy  # Space left in the battery
            actual_charge = min(energy_input, available_space)  # Ensure we don't exceed capacity
            self.current_energy += actual_charge  # Update stored energy
            return {
                'energy_consumed': actual_charge,
                'energy_discharged': 0.0,
                'energy_stored': self.current_energy,
                'status': 'charging'
            }
        
        if action == 2:  # Discharging
            discharge_amount = self.max_discharge_rate  # Limit by max discharge rate
            energy_output = discharge_amount / self.discharge_efficiency
            available_energy = self.current_energy # Energy currently in the device
            actual_discharge = min(energy_output, available_energy)  # Account for efficiency
            self.current_energy -= actual_discharge  # Update stored energy
            return {
                'energy_consumed': 0.0,
                'energy_discharged': actual_discharge,
                'energy_stored': self.current_energy,
                'status': 'discharging'
            }
        
        if action == 0: # Idle (action = 0)
            return {
                'energy_consumed': 0.0,
                'energy_discharged': 0.0,
                'energy_stored': self.current_energy,
                'status': 'idle'
            }

    def reset(self):
        """
        Resets the energy storage device to its initial state.
        """
        self.current_energy = 0.0

    def get_energy_level(self):
        """
        Returns the current energy level in the storage device.
        """
        return self.current_energy
