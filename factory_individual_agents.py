from equipments import PrimaryEquipment,AuxiliaryEquipment,EnergyStorageDevice
import ast
from prettytable import PrettyTable
from gymnasium.spaces import Discrete, MultiDiscrete , Box
import numpy as np

class Factory(object):
    def __init__(self, root, price_signal, pv_signal):
        #env and root of xml file
       
        self.root = root

        # DR signals
        self.PRICE = price_signal
        self.HOURLY_PRICE =[ i/1000 for i in self.PRICE ]
        self.pv_prod = pv_signal

        # factory as discrete events simulation
        self.resources_name_list = []
        self.equipment_name_list = []
        self.primary_equipment_name_list = []
        self.auxillary_equipment_name_list = []
        self.energy_storage_name_list = []
        
        self.consumption_dict = {}
        self.production_dict = {}
        self.container_dict = {}
        self.energy_storage_dict= {}
        self.current_timestep = 0


        self.containers = self.initialize_containers()
        self.equipments = self.initialize_equipments()
        self.energy_storage = self.initialize_energy_storage()

        self._init_factory_parameters()
        self.reset_production() 

        self.agent_action_space = {name:len([x for x in equipment.energydemand if x != 0]) for name,equipment in self.equipments.items()}
        
    def _init_factory_parameters(self):
            
        self.num_primary_resources = ast.literal_eval(self.root.find('info').find('NumPrimary').text)
        self.num_aux_resources = ast.literal_eval(self.root.find('info').find('NumAuxilary').text)
        self.num_nsc_primary = ast.literal_eval(self.root.find('info').find('NumNscPrimary').text)
        self.num_nsc_auxillary = ast.literal_eval(self.root.find('info').find('NumNscAuxilary').text)
        self.num_primary_sc_equip = self.num_primary_resources - self.num_nsc_primary
        self.num_aux_sc_equip = self.num_aux_resources - self.num_nsc_auxillary
        self.num_equipment_agent = self.num_primary_sc_equip + self.num_aux_sc_equip
        self.nsc = ast.literal_eval(self.root.find('info').find('NscPrimary').text)
        self.equipment_agents = [equipment for equipment in self.equipment_name_list if equipment not in self.nsc]
        
        self.primary_agents = [equipment for equipment in self.primary_equipment_name_list if equipment not in self.nsc]
     
        self.auxillary_agents = [equipment for equipment in self.auxillary_equipment_name_list if equipment not in self.nsc]
      
        self.num_resources = ast.literal_eval(self.root.find('info').find('Numcontainers').text)
        self.num_ess = ast.literal_eval(self.root.find('info').find('NumESS').text)
    
    def reset_env(self):
        self.resources_name_list = []
        self.equipment_name_list = []
        self.primary_equipment_name_list = []
        self.auxillary_equipment_name_list = []
        self.energy_storage_name_list = []
        
        self.consumption_dict = {}
        self.production_dict = {}
        self.container_dict = {}
        self.energy_storage_dict= {}
        self.current_timestep = 0

        self.containers = self.initialize_containers()
        self.equipments = self.initialize_equipments()
        self.energy_storage = self.initialize_energy_storage()
        
    def reset_production(self):
        self.production = dict.fromkeys(self.resources_name_list, 0)
        for energy_storage_device in self.energy_storage_name_list:
            self.energy_storage[energy_storage_device].reset()
    
    def initialize_containers(self):
        containers = {}
        # Parse root and create containers
        for container in self.root.find('Resources').findall('.//Container'):
            name = container.get('name')
            self.resources_name_list.append(name)
            initial_val = float(container.find('Initialval').text)
            min_cap = float(container.find('Mincap').text)
            max_cap = float(container.find('Maxcap').text)
            containers[name] = {'value' : initial_val,'min_cap': min_cap, 'max_cap': max_cap}

        return containers
           
    def initialize_equipments(self):
        equipment_dict = {}

        # Parse XML to extract attributes and create primary equipment instances
        for equipment in self.root.find('Equipments').findall('.//Equipment'):

            #equipment nam
            name = equipment.get('name')
            self.equipment_name_list.append(name)          
            # Energy demand for different actions
            energydemand = ast.literal_eval(equipment.find('Energydemand').text) 
            # Raw processing amounts for actions
            rawprocessing = ast.literal_eval(equipment.find('Rawprocessing').text) 
            # Auxiliary processing requirements 
            auxprocessing = ast.literal_eval(equipment.find('Auxilaryprocessing').text)  
            # Input resource name 
            inputresource = equipment.find('Inputresource').text 
       
            # Output resource name 
            outputresource = equipment.find('Outputresource').text

            # equipment type 
            type = equipment.find('Equipmentclass').text.strip().strip('"')
          
         
            if type == "Primary":
                self.primary_equipment_name_list.append(name)

                equipment_dict[name] = PrimaryEquipment(name=name,
                    energydemand=energydemand,
                    rawprocessing=rawprocessing,
                    auxprocessing=auxprocessing,
                    inputresource=inputresource,
                    outputresource=outputresource
                )
        
                
            else:
               self.auxillary_equipment_name_list.append(name)
               equipment_dict[name] = AuxiliaryEquipment(name=name,
                    energydemand=energydemand,
                    rawprocessing=rawprocessing,
                    auxprocessing=auxprocessing,
                    inputresource=inputresource,
                    outputresource=outputresource
                )  
                
        

        return equipment_dict
    
    def initialize_energy_storage(self):
        energy_storage = {}

        for ess in self.root.find('Energystorage').findall('.//Battery'):
            
            name = ess.get('name')         
            # Energy demand for different actions
            self.energy_storage_name_list.append(name)
            capacity = ast.literal_eval(ess.find('Storagecapicity').text) 
            # Raw processing amounts for actions
            discharge_efficiency = ast.literal_eval(ess.find('Dischargingefficiency').text) 
            # Auxiliary processing requirements 
            charge_efficiency = ast.literal_eval(ess.find('Chargingefficiency').text)  
            # Input resource name 
            max_discharge_rate = ast.literal_eval(ess.find('Dischargingrate').text) 
       
            # Output resource name 
            max_charge_rate = ast.literal_eval(ess.find('Chargingrate').text) 


            energy_storage[name] = EnergyStorageDevice(
                capacity = capacity,
                discharge_efficiency = discharge_efficiency,
                charge_efficiency= charge_efficiency,
                max_discharge_rate = max_discharge_rate,
                max_charge_rate = max_charge_rate) 
        return energy_storage

    def set_action_spaces(self):
        spaces = {}
        primary = {name:len([x for x in equipment.energydemand if x != 0]) for name,equipment in self.equipments.items()}
        for agent , space in primary.items():
            spaces[agent] = Discrete(space+1)
        for agent in self.energy_storage_name_list:
            spaces[agent] = Discrete(3)
        return spaces

    def run(self,action,mask=False):

        self.current_action = action
        self.current_action['Reductionfurnace'] = 0
        

        energy_spent = {}
        energy_storage = {}
        production_monitoring = {}
        production_rew = {}
     
       
        current_container_status = self.containers.copy()
        container_updates = []
        for equipment in self.equipment_name_list:
            run_param = self.equipments[equipment].run_equipment(current_container_status, self.current_action[equipment])
            energy_spent[equipment] = run_param['energy']
            resource_dictionary = run_param['resource']
            prod_reward_= run_param['prod_rew']
            container_updates.append(resource_dictionary)
            production = resource_dictionary[self.equipments[equipment].outputresource]
            production_monitoring[self.equipments[equipment].outputresource] = production  
            production_rew[equipment] = prod_reward_         
        for i in container_updates:
            for key,value in i.items():
                self.containers[key]['value'] += value

        for energy_storage_device in self.energy_storage_name_list:
            run_param = self.energy_storage[energy_storage_device].take_action(self.current_action[energy_storage_device])
            energy_storage[energy_storage_device] = run_param
            if run_param['energy_consumed']>0:
                production_rew[energy_storage_device] = 1
            else:
                production_rew[energy_storage_device] = 0

        
        production_reward = production_rew
        consumption_reward = self.get_consumption_reward(energy_spent,energy_storage)
        self.consumption_dict[self.current_timestep]  = energy_spent
        self.production_dict[self.current_timestep] = current_container_status
        self.energy_storage_dict[self.current_timestep] = energy_storage
        self.current_timestep += 1
        new_state = self.get_new_state()
        
        if mask==True:
            masks = self.create_actionmask()
            print(masks)
            new_dict = {}
            for key, value in new_state.items():
                # Create a new nested dictionary with 'initial' and 'new' keys
                new_dict[key] = {'observation': value, 'action_mask': masks[key]}
            new_state = new_dict



        return new_state , production_reward , consumption_reward


    def get_consumption_reward(self,energy_spent,energy_storage):
        consumption_dict = {}
        consumption_list = []
        for agent,energy in energy_spent.items():
            cost = - energy*(self.HOURLY_PRICE[self.current_timestep])
            consumption_dict[agent] = cost
            consumption_list.append(energy)
        for agent,energy in energy_storage.items():  
                cost_charging =  - energy['energy_consumed']*(self.HOURLY_PRICE[self.current_timestep])
                save_discharging = energy['energy_discharged']*(self.HOURLY_PRICE[self.current_timestep])
                consumption_dict[agent] = cost_charging + save_discharging


        return consumption_dict
 
    def get_new_state(self):
        
        new_state = {}
        external_state = [self.current_timestep,self.HOURLY_PRICE[self.current_timestep]]
        energy_storage_state = [ obj.current_energy for name , obj in self.energy_storage.items()]
        for equipment in self.primary_agents:
            input_state = self.containers[self.equipments[equipment].inputresource]['value']
            output_state = self.containers[self.equipments[equipment].outputresource]['value']
            combined_state = [input_state,output_state] + external_state + energy_storage_state
            new_state_ = [max(0.0, i) for i in combined_state]
            new_state[equipment] = np.array(new_state_)
        for equipment in self.auxillary_agents:
            output_state = self.containers[self.equipments[equipment].outputresource]['value']
            combined_state = [output_state] + external_state + energy_storage_state
            new_state_ = [max(0.0, i) for i in combined_state]
            new_state[equipment] = np.array(new_state_)
        for ess in self.energy_storage_name_list:
            combined_state = external_state + energy_storage_state
            new_state_ = [max(0.0, i) for i in combined_state]
            new_state[ess] = np.array(new_state_)

        return new_state
    
    def create_actionmask(self):
        mask = np.ones(1, dtype=int) 
        action_mask = {}
        # Generate a dictionary with the primary actions (length of non-zero energydemand)
        primary = {name: len([x for x in equipment.energydemand if x != 0]) for name, equipment in self.equipments.items()}

        for agent , space in primary.items():
            action_mask[agent] = mask 
        for agent in self.energy_storage_name_list:
            action_mask[agent] = mask 
        
        # Loop through each agent and its corresponding equipment
        for agent, space in primary.items():
            try:
                # Check if inputresource has a value of 0 in the container
                if self.containers[self.equipments[agent].inputresource]['value'] == 0:
                    mask[0] = 0  # Disable action (set to 0)
                    action_mask[agent] = mask  # Add to the action_mask
                if self.containers[self.equipments[agent].inputresource]['value'] == self.containers[self.equipments[agent].inputresource]['max_cap']:
                    mask[0] = 0  # Disable action (set to 0)
                    action_mask[agent] = mask
            except KeyError:
                # Handle case where self.equipments[agent].inputresource does not exist
                pass  # Do nothing if there is an error (e.g., no inputresource for this agent)
        for agent in self.energy_storage_name_list:
            mask = np.ones(1,dtype=int)
            action_mask[agent] = mask
        
        return action_mask


    def print_table(self):
        table_primary = PrettyTable()
        table_primary.field_names = ['Resource', 'Storage']
        d = {}
        for resource,container in self.containers.items():
            table_primary.add_row([resource,container])
            d[resource] = container
        for resource,object in self.energy_storage.items():
            table_primary.add_row([resource,object.current_energy])
            d[resource] = object.current_energy
        
        print(table_primary)
        return d
       


    # def get_new_state(self):
    #     factory_state = [container_level['value']  for name, container_level in self.containers.items()]
    #     external_state = [self.current_timestep,self.HOURLY_PRICE[self.current_timestep+1]]
    #     energy_storage_state = [ obj.current_energy for name , obj in self.energy_storage.items()]
    #     combined_state = factory_state + external_state + energy_storage_state
    #     new_state = [max(0.0, i) for i in combined_state]
    #     return np.array(new_state)