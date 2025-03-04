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

    def set_current_action(self,action_list):
        equipments_ = self.equipment_agents + self.energy_storage_name_list
        current_action = {}
        for i, equipment in enumerate(equipments_):
            current_action[equipment] = action_list[i]
        return current_action

    def run(self,action):

        self.current_action = self.set_current_action(action)
        self.current_action['Reductionfurnace'] = 0
        

        energy_spent = {}
        energy_storage = {}
        production_monitoring = {}
        production_rew = {}
     
       
        current_container_status = self.containers.copy()
        for equipment in self.equipment_name_list:
            run_param = self.equipments[equipment].run_equipment(current_container_status, self.current_action[equipment])
            energy_spent[equipment] = run_param['energy']
            resource_dictionary = run_param['resource']
            prod_reward_= run_param['prod_rew']
            production = resource_dictionary[self.equipments[equipment].outputresource]
            production_monitoring[self.equipments[equipment].outputresource] = production  
            production_rew[equipment] = prod_reward_         
            for key,value in resource_dictionary.items():
                self.containers[key]['value'] += value

        for energy_storage_device in self.energy_storage_name_list:
            run_param = self.energy_storage[energy_storage_device].take_action(self.current_action[energy_storage_device])
            energy_storage[energy_storage_device] = run_param
            if run_param['energy_consumed']>0:
                production_rew[energy_storage_device] = run_param['energy_consumed']
            else:
                production_rew[energy_storage_device] = 0

        
        production_reward = production_rew
        consumption_reward = self.get_consumption_reward(energy_spent,energy_storage)
        self.consumption_dict[self.current_timestep]  = energy_spent
        self.production_dict[self.current_timestep] = current_container_status
        self.energy_storage_dict[self.current_timestep] = energy_storage
        self.current_timestep += 1
        new_state = self.get_new_state()
        
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

        external_state = [self.current_timestep,self.HOURLY_PRICE[self.current_timestep]]

        energy_storage_state = [ obj.current_energy for name , obj in self.energy_storage.items()]

        containers_state = [obj['value'] for name , obj in self.containers.items()]

        combined_state =  containers_state + energy_storage_state + external_state
        new_state_ = [max(0.0, i) for i in combined_state]
        new_state = np.array(new_state_, dtype=np.float32)
        return new_state
    

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
       