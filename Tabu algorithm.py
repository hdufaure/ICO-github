#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt

# nb max de clients est 130
n_clients=50
def data_prep(n_clients):
    n_depot = n_clients+1
    #on importe les donnée sur les clients et le depot
    df=pd.read_excel("2_detail_table_customers.xls")
    df_depot=pd.read_excel('4_detail_table_depots.xls')
    #on extrait les informations qui nous intéressent
    Columns=["CUSTOMER_NUMBER","CUSTOMER_LATITUDE","CUSTOMER_LONGITUDE"]
    Data =( df[Columns]
           .rename(columns={"CUSTOMER_NUMBER":"number","CUSTOMER_LATITUDE":"x","CUSTOMER_LONGITUDE":"y"})
          )
    Data=Data.head(n_clients)
    depot = df_depot[['DEPOT_CODE','DEPOT_LATITUDE','DEPOT_LONGITUDE']].drop_duplicates()
    depot = depot.rename(columns={"DEPOT_CODE":"number of client/depot","DEPOT_LATITUDE":"x","DEPOT_LONGITUDE":"y"})
    Data.loc[len(Data)]=depot.iloc[0]
    return Data
def client_distances(Data):
    #définition d'une matrice U qui contient les distances entre les clienst/depot
    U=np.zeros((len(Data),len(Data)))
    for i in range(len(Data)):
        for j in range(i+1,len(Data)):
            U[i,j]=np.sqrt((Data.iloc[i,1]-Data.iloc[j,1])**2+(Data.iloc[i,2]-Data.iloc[j,2])**2)
            U[j,i]=U[i,j]
    return U
Data=data_prep(n_clients)
U=client_distances(Data)


# In[12]:


nbmax = 200  # Nombre maximal d'itérations
f_limite = 0  # Borne inférieure de la fonction objectif
taille_max_T = 10  # Taille maximale de la liste taboue

def fonction_objectif( L,M=U):
    time=0
    V=30
    for i in range(len(L)-1):
        time+=U[L[i]-1,L[i+1]-1]
    return time

def generer_voisins(solution,n_voisins=50):
    voisins = []
    n=len(solution)
    # Exemple : génération de voisins en inversant deux éléments de la solution
    #for i in range(1,len(solution)//2):
     #       voisin = solution[:]
     #       voisin[i], voisin[n-i-1] = voisin[n-i-1], voisin[i]
      #      voisins.append(voisin)
    for i in range(n_voisins):
            voisin = solution[1:-1]
            random.shuffle(voisin)
            voisin=[solution[0]]+voisin+[solution[0]]
            voisins.append(voisin)
    
    return voisins
def fonction_aspiration(voisins,T,A):
    # Vérifier si la solution est taboue et satisfait l'aspiration
    sol=voisins[0][:]
    for solution in voisins[1:]:
        if (solution in T)  and (fonction_objectif(solution)<fonction_objectif(sol)):
            sol=solution[:]
    
    #T.remove(sol)
    return sol
def initialiser_solution(n=n_clients+1):
    L=[n]
    aux=[i for i in range(1,n-1)]
    random.shuffle(aux)
    L.extend(aux)
    L.append(n)
    return L
def tabou( sol, nbmax):
    # Initialisation
    solution=sol
    if sol==None:
        solution = initialiser_solution()
    #print(solution)
    meilleure_solution = solution[:]
    nb_iter = 0
    T = []  # Liste taboue initialement vide
    meilleure_iteration = 0
    f_limite = 50

    # Initialisation de la fonction d'aspiration
    A = fonction_objectif(meilleure_solution)
    T.append(solution)
    # Processus itératif
    while (fonction_objectif(solution) > 1) and (nb_iter - meilleure_iteration < nbmax):
    
        nb_iter += 1
        voisins = generer_voisins(solution)
        
        meilleure_voisin = solution[:]
        #print(meilleure_voisin)
        test=False
        for voisin in voisins:
            #print(fonction_objectif(meilleure_voisin))
            if (voisin not in T) and (fonction_objectif(voisin) <= A) and (fonction_objectif(voisin) <= fonction_objectif(meilleure_voisin)) :
                meilleure_voisin = voisin[:]
                test=True
        
        if test==False:
            #print(True)
            # Si aucun voisin valide n'est trouvé, continuer avec la prochaine itération
            meilleure_voisin=fonction_aspiration(voisins,T,fonction_objectif(meilleure_voisin))    
        # Mettre à jour la meilleure solution trouvée
        if fonction_objectif(meilleure_voisin) < fonction_objectif(meilleure_solution):
            meilleure_solution = meilleure_voisin[:]
            meilleure_iteration = nb_iter
        # Mettre à jour la liste taboue
        T.append(meilleure_voisin)
        if len(T) > taille_max_T:
            T.pop(0)  # Supprimer le plus ancien élément de T
        # Mettre à jour la fonction d'aspiration
        A = fonction_objectif(meilleure_solution)
        # Mettre à jour la solution courante
        solution = meilleure_voisin[:]
    #print(A)
    return A,meilleure_solution


# In[13]:


nbmax = 300  # Nombre maximal d'itérations
taille_max_T = 10  # Taille maximale de la liste taboue

DEPOT = (Data.iloc[-1]['x'], Data.iloc[-1]['y'])
client_positions = list(zip(Data['x'], Data['y']))


# In[20]:


nb_iter=100
A,best_solution=tabou(None,nbmax)
for i in range(nb_iter):
    A_aux,best_solution_aux=tabou(best_solution,nbmax)
    if A>A_aux:
        best_solution=best_solution_aux[:]
        A=A_aux
A


# In[15]:


#new_cities_order = np.concatenate((np.array([Data.loc[best_solution[i]-1,['x','y']] for i in range(len(best_solution))]),np.array([Data.loc[n_clients,['x','y']]])))
# Plot the cities.
plt.scatter(Data['x'],Data['y'])
plt.scatter(*DEPOT, color='red', label='Dépôt')
# Plot the path.
X=[Data.loc[i-1,'x'] for i in best_solution]
Y=[Data.loc[i-1,'y'] for i in best_solution]
plt.plot(X,Y)
plt.show()


# In[28]:


from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector


# In[75]:


class OptTabuAgentCollab(Agent):
    def __init__(self, unique_id, model, collaboratif=True):
        super().__init__(unique_id, model)        
        self.sol = initialiser_solution()
        self.best=1 #inverse de la distance
        self.collaboratif=collaboratif
        self.voisins=generer_voisins(self.sol)
        self.T = []
        self.A = fonction_objectif(self.sol)
        self.meilleure_solution = self.sol[:]

    def contact(self):
        min=self.best
        for a in self.model.schedule.agents:
            if a.best<min:
                min=a.best
                best_agent=a
                
        if min<=self.best:
            #print(f"je ne suis pas le meilleure, l'agent {a.unique_id} a une distance de {min} qui est : {best_agent.best}")
            self.best=min
            #print(f"je suis maintenant aussi le meilleure, : {self.best}")
            
    #passser d'une itération à une autre
    def step(self):
        self.T.append(self.sol)
        if self.collaboratif==True:
            self.contact()
        meilleure_voisin = self.sol[:]
        #print(meilleure_voisin)
        test=False
        for voisin in self.voisins:
            #print(fonction_objectif(meilleure_voisin))
            if (voisin not in self.T) and (fonction_objectif(voisin) <= self.A) and (fonction_objectif(voisin) <= fonction_objectif(meilleure_voisin)) :
                meilleure_voisin = voisin[:]
                test=True
        
        if test==False:
            #print(True)
            # Si aucun voisin valide n'est trouvé, continuer avec la prochaine itération
            meilleure_voisin=fonction_aspiration(self.voisins,self.T,fonction_objectif(meilleure_voisin))    
        # Mettre à jour la meilleure solution trouvée
        if fonction_objectif(meilleure_voisin) < fonction_objectif(self.meilleure_solution):
            self.meilleure_solution = meilleure_voisin[:]
        # Mettre à jour la liste taboue
        self.T.append(meilleure_voisin)
        if len(self.T) > taille_max_T:
            self.T.pop(0)  # Supprimer le plus ancien élément de T
        # Mettre à jour la fonction d'aspiration
        self.A = fonction_objectif(self.meilleure_solution)
        # Mettre à jour la solution courante
        
        self.sol = meilleure_voisin[:]
        #print(f"je suis l'agent collab d'id {self.unique_id} et ma solution est {self.sol}")


# In[76]:


class OptTabuAgentNonCollab(Agent):
    def __init__(self, unique_id, model, collaboratif=False):
        super().__init__(unique_id, model)        
        self.sol = initialiser_solution()
        self.best=1 #inverse de la distance
        self.collaboratif=collaboratif
        self.voisins=generer_voisins(self.sol)
        self.T = []
        self.meilleure_solution = self.sol[:]
        self.A = fonction_objectif(self.sol)

    def contact(self):
        min=self.best
        for a in self.model.schedule.agents:
            if a.best<min:
                min=a.best
                best_agent=a
                
        if min<=self.best:
            print(f"je ne suis pas le meilleure, l'agent {a.unique_id} a une distance de {min} qui est : {best_agent.best}")
            self.best=min
            print(f"je suis maintenant aussi le meilleure, : {self.best}")
            
    #passser d'une itération à une autre
    def step(self):
        self.T.append(self.sol)
        if self.collaboratif==True:
            self.contact()
        meilleure_voisin = self.sol[:]
        #print(meilleure_voisin)
        test=False
        for voisin in self.voisins:
            #print(fonction_objectif(meilleure_voisin))
            if (voisin not in self.T) and (fonction_objectif(voisin) <= self.A) and (fonction_objectif(voisin) <= fonction_objectif(meilleure_voisin)) :
                meilleure_voisin = voisin[:]
                test=True
        
        if test==False:
            #print(True)
            # Si aucun voisin valide n'est trouvé, continuer avec la prochaine itération
            meilleure_voisin=fonction_aspiration(self.voisins,self.T,fonction_objectif(meilleure_voisin))    
        # Mettre à jour la meilleure solution trouvée
        if fonction_objectif(meilleure_voisin) < fonction_objectif(self.meilleure_solution):
            self.meilleure_solution = meilleure_voisin[:]
            meilleure_iteration = nb_iter
        # Mettre à jour la liste taboue
        self.T.append(meilleure_voisin)
        if len(self.T) > taille_max_T:
            self.T.pop(0)  # Supprimer le plus ancien élément de T
        # Mettre à jour la fonction d'aspiration
        self.A = fonction_objectif(self.meilleure_solution)
        # Mettre à jour la solution courante
        
        self.sol = meilleure_voisin[:]
        #print(f"je suis l'agent non collab d'id {self.unique_id} et ma solution est {self.sol}")
        


# In[84]:


class TabuModel(Model):
    def fonction_objectif(self,sol,M=U):
        time=0
        V=30
        for i in range(len(sol)-1):
            time+=U[sol[i]-1,sol[i+1]-1]
        return time
    
    #le constructeur
    def __init__(self,NC=10, NNC=10):
        self.num_agents_collab=NC
        self.num_agents_non_collab=NNC
        super().__init__()
        self.schedule=SimultaneousActivation(self) #ajout chef orchestre
        
        #ajout d'un datacollector
        self.datacollector = DataCollector(model_reporters={"Temps calculé total pour le trajet": self.fonction_objectif})        

        #ajout des premiers agents au SMA
        for i in range(NC):
            a = OptTabuAgentCollab(i, self)
            self.schedule.add(a)
            
        for i in range(NNC):
            a = OptTabuAgentNonCollab(i+NC, self)
            self.schedule.add(a)

    def step(self):
        self.schedule.step() 
        self.datacollector.collect(self)

#test du modèle
steps=10
monSMA = TabuModel(10,10)
for i in range(steps):
    print(f"l'instant {i+1}")
    monSMA.step()
    
data = monSMA.datacollector.get_model_vars_dataframe()

# Affiche la courbe
import matplotlib.pyplot as plt
plt.plot(data)


# In[80]:


data


# In[ ]:





# In[ ]:


class OptimisationCollaborativeModel(Model):
    """A model for infection spread."""

    def __init__(self, population, N=6, popSize=100, eliteSize=20, mutationRate=0.01):

        self.population = population
        self.num_agents = N
        self.popSize = popSize
        self.eliteSize = eliteSize
        self.mutationRate = mutationRate
        
        
        
        #The scheduler
        self.schedule = SimultaneousActivation(self)
        
        
        
        
        
        # Create agents
        for i in range(int(self.num_agents/2)):
            a = OptGenAgent(i, self)
            self.schedule.add(a)
            
        for i in range(int(self.num_agents/2),self.num_agents):
            a = OptGenAgent(i, self,True)
            self.schedule.add(a)            
        
        self.datacollector = DataCollector(
            #model_reporters={"TheGlobalBest": compute_global_best},
            agent_reporters={"Best": lambda a:a.best})

            
    
        
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        



generations=500


cityList = []

for i in range(0,25):
    cityList.append(Point(x=int(random.random() * 200), y=int(random.random() * 200)))
    
model = OptimisationCollaborativeModel(cityList)

for i in range(generations):
    print(f"Génération n{i+1}")
    model.step()  
    

agent_state = model.datacollector.get_agent_vars_dataframe()
print(agent_state)
res=agent_state.unstack()
print(res)
res.plot()
print("la meilleure valeur trouvée : ")
print(res.min().min())

